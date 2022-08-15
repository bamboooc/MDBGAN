import datetime
import dateutil.tz
import os
import os.path as osp
from shutil import copyfile, copytree
import glob
import time
import random
import torch

from MDB.config import get_arguments
import MDB.functions3D as functions


# noinspection PyInterpreter
if __name__ == '__main__':
    parser = get_arguments()
    parser.add_argument('--input_name', help='input image name for training', required=True)
    parser.add_argument('--naive_img', help='naive input image  (harmonization or editing)', default="")
    parser.add_argument('--gpu', type=int, help='which GPU to use', default=0)
    parser.add_argument('--train_mode', default='generation',
                        choices=['generation', 'retarget', 'harmonization', 'editing', 'animation'],
                        help="generation, retarget, harmonization, editing, animation")
    parser.add_argument('--lr_scale', type=float, help='scaling of learning rate for lower stages', default=0.1)
    parser.add_argument('--train_stages', type=int, help='how many stages to use for training', default=6)

    parser.add_argument('--fine_tune', action='store_true', help='whether to fine tune on a given image', default=0)
    parser.add_argument('--model_dir', help='model to be used for fine tuning (harmonization or editing)', default="")

    opt = parser.parse_args()
    opt = functions.post_config(opt)


    if not os.path.exists(opt.input_name):
        print("Image does not exist: {}".format(opt.input_name))
        print("Please specify a valid image.")
        exit()

    if torch.cuda.is_available():
        torch.cuda.set_device(opt.gpu)

    if opt.train_mode == "generation" or opt.train_mode == "retarget" or opt.train_mode == "animation":
        if opt.train_mode == "animation":
            opt.min_size = 20
        from MDB.training_generation import *
    elif opt.train_mode == "harmonization" or opt.train_mode == "editing":
        if opt.fine_tune:
            if opt.model_dir == "":
                print("Model for fine tuning not specified.")
                print("Please use --model_dir to define model location.")
                exit()
            else:
                if not os.path.exists(opt.model_dir):
                    print("Model does not exist: {}".format(opt.model_dir))
                    print("Please specify a valid model.")
                    exit()
            if not os.path.exists(opt.naive_img):
                print("Image for harmonization/editing not found: {}".format(opt.naive_img))
                exit()

    dir2save = functions.generate_dir2save(opt)

    if osp.exists(dir2save):
        print('Trained model already exist: {}'.format(dir2save))
        exit()

    # create log dir
    try:
        os.makedirs(dir2save)
    except OSError:
        pass


    # train model
    print("Training model ({})".format(dir2save))
    start = time.time()
    train(opt)
    end = time.time()
    elapsed_time = end - start
    print("Time for training: {} seconds".format(elapsed_time))
