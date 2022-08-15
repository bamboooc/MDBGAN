import torch
import torch.nn as nn
import numpy as np
import math
import copy
import torch.nn.functional as F
from MDB.imresize import imresize, imresize_to_shape
from MDB import rbfl

def weights_init(m):
    classname = m.__class__.__name__
    if classname.find('Conv2d') != -1:
        m.weight.data.normal_(0.0, 0.02)
    elif classname.find('Norm') != -1:
        m.weight.data.normal_(1.0, 0.02)
        m.bias.data.fill_(0)


class ContentBranch(nn.Module):
    def __init__(self, channel):
        super(ContentBranch, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(nn.Conv2d(channel, channel // 16, 1, bias=False),
                                nn.ReLU(),
                                nn.Conv2d(channel // 16, channel, 1, bias=False))
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        temp = x
        avg_out = self.fc(self.avg_pool(x))
        return self.sigmoid(avg_out) * temp

class LayoutBranch(nn.Module):
    def __init__(self, kernel_size=1):
        super(LayoutBranch, self).__init__()
        self.conv1 = nn.Conv2d(64, 1, kernel_size, padding=kernel_size // 2, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        temp = x
        x = self.conv1(x)
        print("lblb-x",x.shape)
        # print("temp", temp.shape)
        return self.sigmoid(x) * temp


def get_activation(opt):
    activations = {"lrelu": nn.LeakyReLU(opt.lrelu_alpha, inplace=True),
                   "elu": nn.ELU(alpha=1.0, inplace=True),
                   "prelu": nn.PReLU(num_parameters=1, init=0.25),
                   "selu": nn.SELU(inplace=True)
                   }
    return activations[opt.activation]


def uplbmple(x, size):
    # https://blog.csdn.net/moshiyaofei/article/details/102243913
    x_up = torch.nn.functional.interpolate(x, size=size, mode='trilinear', align_corners=True)
    return x_up


class ConvBlock(nn.Sequential):
    def __init__(self, in_channel, out_channel, ker_size, padd, opt, generator=False):
        super(ConvBlock, self).__init__()
        self.add_module('conv', nn.Conv2d(in_channel, out_channel, kernel_size=ker_size, stride=1, padding=padd))
        if generator and opt.batch_norm:
            self.add_module('norm', nn.BatchNorm2d(out_channel))
        self.add_module(opt.activation, get_activation(opt))

class GeneratorBlock(nn.Module):
    def __init__(self, in_channel, out_channel,dilation=1,stride=1,bn=True):
        super(GeneratorBlock, self).__init__()
        self.liner0 = ConvBlock(in_channel, out_channel, ker_size=3, padding=1, stride=stride, bn=bn)
        self.branch0 = nn.Sequential(
            ConvBlock(out_channel, out_channel, ker_size=3, padding=2, stride=stride,dilation=2,bn=bn)
        )
        self.branch1 = nn.Sequential(
            ConvBlock(out_channel, out_channel, ker_size=3, padding=1, stride=stride,dilation=1,bn=bn),
            ConvBlock(out_channel, out_channel, ker_size=3, padding=3, stride=stride, dilation=3, bn=bn)
        )
        self.liner1 = ConvBlock2D(out_channel*3, out_channel, ker_size=3, padding=1, stride=stride, bn=bn)

    def forward(self,x):
        init = self.liner0(x)
        x1 = self.branch0(init)
        x2 = self.branch1(init)
        xx1 = torch.cat((init,x1,x2),1)
        out = self.liner1(xx1)
        return out

class Discriminator(nn.Module):
    def __init__(self, opt):
        super(Discriminator, self).__init__()

        self.opt = opt
        N = int(opt.nfc)
        M = N

        self.head = ConvBlock(opt.nc_im, N, opt.ker_size, opt.padd_size, opt)
        self.body = nn.Sequential()

        for i in range(opt.num_layer):
            # block = ConvBlock(N, N, opt.ker_size, opt.padd_size, opt)
            block = GeneratorBlock(N,N)
            self.body.add_module('block%d' % (i), block)

        self.cb = ChannelAttention(out_planes=64)
        self.lb = SpatialAttention()
        # self.tail = nn.Sequential(ConvBlock(N, N, opt.ker_size, opt.padd_size, opt),ConvBlock(N, N, opt.ker_size, opt.padd_size, opt),nn.Conv2d(N, 1, kernel_size=opt.ker_size, padding=opt.padd_size))
        self.tail = nn.Conv2d(N, 1, kernel_size=opt.ker_size, padding=opt.padd_size)

        # self.rbf=rbf.BasicConv(M)



    def forward(self, x):
        head = self.head(x)
        body = self.body(head)
        cb = self.cb(body)
        print("cb",cb.shape)
        out1 = self.tail(cb)
        lb = self.lb(body)
        print("lb",lb.shape)
        out2 = self.tail(lb)
        # out = out1 + out2
        return out1,out2

class GrowingGenerator(nn.Module):
    def __init__(self, opt):
        super(GrowingGenerator, self).__init__()

        self.opt = opt
        N = int(opt.nfc)

        self._pad = nn.ConstantPad2d(1,0)
        self._pad_block = nn.ConstantPad2d(opt.num_layer-1,0) if opt.train_mode == "generation"\
                                                           or opt.train_mode == "animation" \
                                                        else nn.ConstantPad2d(opt.num_layer,0)

        self.head = ConvBlock(opt.nc_im, N, opt.ker_size, opt.padd_size, opt, generator=True)

        #

        self.body = torch.nn.ModuleList([])
        _first_stage = nn.Sequential()
        for i in range(1):
            block = GeneratorBlock(N,N)
            _first_stage.add_module('block%d'%(i),block)
        self.body.append(_first_stage)
        self.tail = nn.Sequential(
            nn.Conv2d(N, opt.nc_im, kernel_size=opt.ker_size, padding=opt.padd_size),
            nn.Tanh())

    def init_next_stage(self):
        self.body.append(copy.deepcopy(self.body[-1]))

    def forward(self, noise, real_shapes, noise_amp):
        x = self.head(self._pad(noise[0]))

        # we do some uplbmpling for training models for unconditional generation to increase
        # the image diversity at the edges of generated images
        if self.opt.train_mode == "generation" or self.opt.train_mode == "animation":
            x = uplbmple(x, size=[x.shape[2] + 2, x.shape[3] + 2, x.shape[4] + 2])
            # print(x.shape)
        x = self._pad_block(x)
        x_prev_out = self.body[0](x)

        for idx, block in enumerate(self.body[1:], 1):
            if self.opt.train_mode == "generation" or self.opt.train_mode == "animation":
                x_prev_out_1 = uplbmple(x_prev_out, size=[real_shapes[idx][2], real_shapes[idx][3], real_shapes[idx][4]])
                x_prev_out_2 = uplbmple(x_prev_out, size=[real_shapes[idx][2] + self.opt.num_layer*2,
                                                          real_shapes[idx][3] + self.opt.num_layer*2,
                                                          real_shapes[idx][4] + self.opt.num_layer*2])
                x_prev = block(x_prev_out_2 + noise[idx] * noise_amp[idx])
            else:
                x_prev_out_1 = uplbmple(x_prev_out, size=real_shapes[idx][2:])
                x_prev = block(self._pad_block(x_prev_out_1+noise[idx]*noise_amp[idx]))
            x_prev_out = x_prev + x_prev_out_1
        out = self.tail(self._pad(x_prev_out))
        return out
