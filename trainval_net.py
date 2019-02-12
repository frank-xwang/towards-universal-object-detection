# --------------------------------------------------------
# Pytorch multi-GPU Faster R-CNN
# Licensed under The MIT License [see LICENSE for details]
# Written by Jiasen Lu, Jianwei Yang, based on code from Ross Girshick
# --------------------------------------------------------
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import _init_paths
import os
import sys
import numpy as np
import argparse
import pprint
import pdb
import time

import torch
from torch.autograd import Variable
import torch.nn as nn
import torch.optim as optim

import torchvision.transforms as transforms
from torch.utils.data.sampler import Sampler

from roi_data_layer.roidb import combined_roidb
from roi_data_layer.roibatchLoader import roibatchLoader
from model.utils.config import cfg, cfg_from_file, cfg_from_list, get_output_dir
from model.utils.net_utils import weights_normal_init, save_net, load_net, \
      adjust_learning_rate, save_checkpoint, clip_gradient

from model.faster_rcnn.vgg16 import vgg16
from model.faster_rcnn.resnet import resnet
from model.faster_rcnn.SEResNet import se_resnet

def parse_args():
  """
  Parse input arguments
  """
  parser = argparse.ArgumentParser(description='Train a Fast R-CNN network')
  parser.add_argument('--dataset', dest='dataset',
                      help='training dataset',
                      default='pascal_voc', type=str)
  parser.add_argument('--net', dest='net',
                    help='vgg16, res101',
                    default='vgg16', type=str)
  parser.add_argument('--start_epoch', dest='start_epoch',
                      help='starting epoch',
                      default=1, type=int)
  parser.add_argument('--epochs', dest='max_epochs',
                      help='number of epochs to train',
                      default=20, type=int)
  parser.add_argument('--disp_interval', dest='disp_interval',
                      help='number of iterations to display',
                      default=100, type=int)
  parser.add_argument('--checkpoint_interval', dest='checkpoint_interval',
                      help='number of iterations to display',
                      default=10000, type=int)

  parser.add_argument('--save_dir', dest='save_dir',
                      help='directory to save models', default="models",
                      nargs=argparse.REMAINDER)
  parser.add_argument('--nw', dest='num_workers',
                      help='number of worker to load data',
                      default=0, type=int)
  parser.add_argument('--cuda', dest='cuda',
                      help='whether use CUDA',
                      action='store_true')
  parser.add_argument('--ls', dest='large_scale',
                      help='whether use large imag scale',
                      action='store_true')                      
  parser.add_argument('--mGPUs', dest='mGPUs',
                      help='whether use multiple GPUs',
                      action='store_true')
  parser.add_argument('--bs', dest='batch_size',
                      help='batch_size',
                      default=1, type=int)
  parser.add_argument('--cag', dest='class_agnostic',
                      help='whether perform class_agnostic bbox regression',
                      action='store_true')

  # config optimization
  parser.add_argument('--o', dest='optimizer',
                      help='training optimizer',
                      default="sgd", type=str)
  parser.add_argument('--lr', dest='lr',
                      help='starting learning rate',
                      default=0.001, type=float)
  parser.add_argument('--lr_decay_step', dest='lr_decay_step',
                      help='step to do learning rate decay, unit is epoch',
                      default=5, type=int)
  parser.add_argument('--lr_decay_gamma', dest='lr_decay_gamma',
                      help='learning rate decay ratio',
                      default=0.1, type=float)

  # set training session
  parser.add_argument('--s', dest='session',
                      help='training session',
                      default=1, type=int)

  # resume trained model
  parser.add_argument('--r', dest='resume',
                      help='resume checkpoint or not',
                      default=False, type=bool)
  parser.add_argument('--checksession', dest='checksession',
                      help='checksession to load model',
                      default=1, type=int)
  parser.add_argument('--checkepoch', dest='checkepoch',
                      help='checkepoch to load model',
                      default=1, type=int)
  parser.add_argument('--checkpoint', dest='checkpoint',
                      help='checkpoint to load model',
                      default=0, type=int)
  # log and diaplay
  parser.add_argument('--use_tfboard', dest='use_tfboard',
                      help='whether use tensorflow tensorboard',
                      default=False, type=bool)
  parser.add_argument('--fix_bn', dest='fix_bn',
                      help='whether fix batch norm layer',
                      default="False", type=str)
  parser.add_argument('--USE_FLIPPED', dest='USE_FLIPPED',
                      help='whether use flipped images',
                      default='True', type=str)
  parser.add_argument('--DATA_DIR', dest='DATA_DIR',
                      help='path to DATA_DIR',
                      default="/home/xuw080/data4/universal_model/data/", type=str)
  args = parser.parse_args()
  return args


def load_state_dict(model, state_dict):      
    # load weight and bias of bn params for each datasets
    # for k, v in state_dict.items():
    #     print(k)
    # # print(v.shape)
    # for k in model.state_dict():
    #     print(k)
    new_state_dict = {}

    for k, v in model.state_dict().items():
        if 'RCNN_cls_score' in k or 'RCNN_bbox_pred' in k:
          new_state_dict[k] = v
        elif k in state_dict:
            new_state_dict[k] = state_dict[k]
    
    model.load_state_dict(new_state_dict)


class sampler(Sampler):
  def __init__(self, train_size, batch_size):
    self.num_data = train_size
    self.num_per_batch = int(train_size / batch_size)
    self.batch_size = batch_size
    self.range = torch.arange(0,batch_size).view(1, batch_size).long()
    self.leftover_flag = False
    if train_size % batch_size:
      self.leftover = torch.arange(self.num_per_batch*batch_size, train_size).long()
      self.leftover_flag = True

  def __iter__(self):
    rand_num = torch.randperm(self.num_per_batch).view(-1,1) * self.batch_size
    self.rand_num = rand_num.expand(self.num_per_batch, self.batch_size) + self.range

    self.rand_num_view = self.rand_num.view(-1)

    if self.leftover_flag:
      self.rand_num_view = torch.cat((self.rand_num_view, self.leftover),0)

    return iter(self.rand_num_view)

  def __len__(self):
    return self.num_data

if __name__ == '__main__':

  args = parse_args()

  print('Called with args:')
  print(args)
  cfg.VGG_ORIGIN = True
  cfg.sample_mode = 'random'
  cfg.DEBUG = False # set as True if debug whether 'people' is ignored
  cfg.filter_empty = True
  cfg.ignore_people = False
  cfg.use_coco_igonore = True
  cfg.random_resize = False

  if args.use_tfboard:
    from model.utils.logger import Logger
    # Set the logger
    logger = Logger('./logs')

  if args.dataset == "pascal_voc":
      args.imdb_name = "voc_2007_trainval"
      args.imdbval_name = "voc_2007_test"
      cfg.dataset = args.dataset
      cfg.TRAIN.SCALES=(600,)
      # cfg.TRAIN.BATCH_SIZE = 128
      # cfg.TRAIN.RPN_BATCHSIZE = 128
      cfg.TRAIN.USE_FLIPPED = True
      cfg.imdb_name = args.imdb_name
      cfg.filter_empty = True
      cfg.POOLING_SIZE_H = 7
      cfg.POOLING_SIZE_W = 7
      #args.set_cfgs = ['ANCHOR_SCALES', '[2.72, 3.81, 5.45, 7.64, 10.9, 15.27, 21.8, 32]', 'ANCHOR_RATIOS', '[2]', 'MAX_NUM_GT_BOXES', '20']
      args.set_cfgs = ['ANCHOR_SCALES', '[8, 16, 32]', 'ANCHOR_RATIOS', '[0.5, 1, 2]', 'MAX_NUM_GT_BOXES', '20']
      args.set_cfgs = ['ANCHOR_SCALES', '[0.75, 1, 1.5, 2, 3, 4, 6, 8, 12, 16, 24, 30]', 'ANCHOR_RATIOS', '[0.5, 1, 2]', 'MAX_NUM_GT_BOXES', '20']
      #args.set_cfgs = ['ANCHOR_SCALES', '[8, 16, 32]', 'ANCHOR_RATIOS', '[0.5,1,2]', 'MAX_NUM_GT_BOXES', '20']
      
  elif args.dataset == "pascal_voc_0712":
      args.imdb_name = "voc_2007_trainval+voc_2012_trainval"
      args.imdbval_name = "voc_2007_test"
      cfg.dataset = args.dataset
      cfg.TRAIN.SCALES=(600,)
      cfg.TRAIN.USE_FLIPPED = True
      cfg.imdb_name = args.imdb_name
      cfg.POOLING_SIZE_H = 7
      cfg.POOLING_SIZE_W = 7
      args.set_cfgs = ['ANCHOR_SCALES', '[4, 8, 16, 32]', 'ANCHOR_RATIOS', '[0.5,1,2]', 'MAX_NUM_GT_BOXES', '20']
      args.set_cfgs = ['ANCHOR_SCALES', '[0.75, 1, 1.5, 2, 3, 4, 6, 8, 12, 16, 24, 30]', 'ANCHOR_RATIOS', '[0.5, 1, 2]', 'MAX_NUM_GT_BOXES', '30']

  elif args.dataset == "Kitchen":
      args.imdb_name = "kitchen_train"
      args.imdbval_name = "kitchen_test"
      cfg.dataset = args.dataset
      cfg.TRAIN.SCALES=(1024,)
      cfg.TRAIN.USE_FLIPPED = True
      cfg.imdb_name = args.imdb_name
      cfg.POOLING_SIZE_H = 7
      cfg.POOLING_SIZE_W = 7
      args.set_cfgs = ['ANCHOR_SCALES', '[4, 8, 16, 32]', 'ANCHOR_RATIOS', '[0.5,1,2]', 'MAX_NUM_GT_BOXES', '20']
      args.set_cfgs = ['ANCHOR_SCALES', '[0.75, 1, 1.5, 2, 3, 4, 6, 8, 12, 16, 24, 30]', 'ANCHOR_RATIOS', '[0.5, 1, 2]', 'MAX_NUM_GT_BOXES', '30']

  elif args.dataset == "coco":
      args.imdb_name = "coco_2014_train+coco_2014_valminusminival"
      # args.imdb_name = "coco_2014_valminusminival"
      cfg.dataset = args.dataset
      cfg.imdb_name = args.imdb_name
      cfg.TRAIN.USE_FLIPPED = True
      cfg.TRAIN.SCALES=(600,)
      cfg.POOLING_SIZE_H = 7
      cfg.POOLING_SIZE_W = 7
      args.imdbval_name = "coco_2014_minival"
      args.set_cfgs = ['ANCHOR_SCALES', '[4, 8, 16, 32]', 'ANCHOR_RATIOS', '[0.5,1,2]', 'MAX_NUM_GT_BOXES', '50']

  elif args.dataset == "imagenet":
      args.imdb_name = "imagenet_train"
      args.imdbval_name = "imagenet_val"
      cfg.dataset = args.dataset
      cfg.imdb_name = args.imdb_name
      cfg.TRAIN.USE_FLIPPED = True
      cfg.POOLING_SIZE_H = 7
      cfg.POOLING_SIZE_W = 7
      args.set_cfgs = ['ANCHOR_SCALES', '[4, 8, 16, 32]', 'ANCHOR_RATIOS', '[0.5,1,2]', 'MAX_NUM_GT_BOXES', '30']

  elif args.dataset == "caltech":
      args.imdb_name = "caltech_train"
      args.imdbval_name = "caltech_val"
      cfg.dataset = args.dataset
      cfg.imdb_name = args.imdb_name
      cfg.TRAIN.USE_FLIPPED = True 
      # cfg.TRAIN.RPN_BATCHSIZE = 32
      # cfg.TRAIN.BATCH_SIZE = 64
      cfg.RPN_POSITIVE_OVERLAP = 0.5
      cfg.TRAIN.RPN_NMS_THRESH = 0.65
      cfg.POOLING_SIZE_H = 7
      cfg.POOLING_SIZE_W = 7
      cfg.TRAIN.FG_THRESH = 0.45
      cfg.TRAIN.SCALES=(720,)
      cfg.sample_mode = 'bootstrap' # use bootstrap or ramdom as sampling method
      cfg.VGG_ORIGIN = False # whether use vgg original classification layers
      cfg.TRAIN.USE_ALL_GT = True # choose true if want to exclude all proposals overlap with 'people' larger than 0.3
      cfg.ignore_people = False # ignore people, all proposals overlap with 'people' larger than 0.3 will be igonored
      cfg.use_coco_igonore = True
      cfg.filter_empty = True # whether filter 0 gt images
      cfg.DEBUG = False # set as True if debug whether 'people' is ignored
      ## scales*11 is the new_width, new_width*ratio is new height
      # 30, 42, 60, 84, 120, 168, 240 width
      args.set_cfgs = ['ANCHOR_SCALES', '[0.75, 1, 1.5, 2, 3, 4, 6, 8, 12, 16, 24, 30]', 'ANCHOR_RATIOS', '[1, 2]', 'MAX_NUM_GT_BOXES', '30']

  elif args.dataset == "KITTIVOC":
      args.imdb_name = "kittivoc_train"
      args.imdbval_name = "kittivoc_val"
      cfg.TRAIN.USE_FLIPPED = True
      # cfg.TRAIN.BATCH_SIZE = 256
      # cfg.TRAIN.RPN_BATCHSIZE = 256
      cfg.RPN_POSITIVE_OVERLAP = 0.5
      cfg.TRAIN.RPN_NMS_THRESH = 0.65
      cfg.TRAIN.FG_THRESH = 0.45
      cfg.POOLING_SIZE_H = 7
      cfg.POOLING_SIZE_W = 7
      cfg.dataset = args.dataset
      cfg.imdb_name = args.imdb_name
      cfg.TRAIN.SCALES=(576,)
      ## scales*11 is the new_width, new_width*ratio is new height
      # 30, 42, 60, 84, 120, 168, 240, 355
      args.set_cfgs = ['ANCHOR_SCALES', '[2.72, 3.81, 5.45, 7.64, 10.9, 15.27, 21.8, 32]', 'ANCHOR_RATIOS', '[2]', 'MAX_NUM_GT_BOXES', '20']
      args.set_cfgs = ['ANCHOR_SCALES', '[0.75, 1, 1.5, 2, 3, 4, 6, 8, 12, 16, 24, 30]', 'ANCHOR_RATIOS', '[0.5, 1, 2]', 'MAX_NUM_GT_BOXES', '50']

  elif args.dataset == "widerface":
      args.imdb_name = "widerface_train"
      args.imdbval_name = "widerface_val"
      cfg.TRAIN.USE_FLIPPED = True
      # cfg.TRAIN.RPN_BATCHSIZE = 256
      # cfg.TRAIN.BATCH_SIZE = 256
      cfg.RPN_POSITIVE_OVERLAP = 0.5
      cfg.TRAIN.RPN_NMS_THRESH = 0.65
      cfg.POOLING_SIZE_H = 7
      cfg.POOLING_SIZE_W = 7
      cfg.TRAIN.FG_THRESH = 0.45
      cfg.dataset = args.dataset
      cfg.imdb_name = args.imdb_name
      cfg.TRAIN.SCALES=(600,)
      cfg.sample_mode = 'bootstrap' # use bootstrap or ramdom as sampling method
      cfg.VGG_ORIGIN = True # whether use vgg original classification layers
      cfg.TRAIN.USE_ALL_GT = True # choose true if want to exclude all proposals overlap with 'people' larger than 0.3
      cfg.filter_empty = False # whether filter 0 gt images
      cfg.DEBUG = False # set as True if debug whether 'people' is ignored
      args.set_cfgs = ['ANCHOR_SCALES', '[0.75, 1, 1.5, 2, 3, 4, 6, 8, 12, 16, 24, 30]', 'ANCHOR_RATIOS', '[0.5,1,2]', 'MAX_NUM_GT_BOXES', '300']

  elif args.dataset == "citypersons":
      args.imdb_name = "citypersons_train"
      args.imdbval_name = "citypersons_val"
      cfg.TRAIN.USE_FLIPPED = True
      # cfg.TRAIN.RPN_BATCHSIZE = 128
      # cfg.TRAIN.BATCH_SIZE = 128
      cfg.RPN_POSITIVE_OVERLAP = 0.7
      cfg.TRAIN.RPN_NMS_THRESH = 0.65
      cfg.POOLING_SIZE_H = 7
      cfg.POOLING_SIZE_W = 7
      cfg.TRAIN.FG_THRESH = 0.45
      cfg.dataset = args.dataset
      cfg.imdb_name = args.imdb_name
      cfg.TRAIN.SCALES=(1024,) # up-sampling 2x is the best, we choose 1.6x this time. #  1331=1.6x
      cfg.sample_mode = 'bootstrap' # use bootstrap or ramdom as sampling method
      cfg.VGG_ORIGIN = False # whether use vgg original classification layers
      cfg.TRAIN.USE_ALL_GT = True # choose true if want to exclude all proposals overlap with 'people' larger than 0.3
      cfg.filter_empty = False # whether filter 0 gt images
      args.optimizer =="adam"
      cfg.DEBUG = False # set as True if debug whether 'people' is ignored
      ## scales*11 is the new_width, new_width*ratio is new height
      # 30, 42, 60, 84, 120, 168, 240 width
      args.set_cfgs = ['ANCHOR_SCALES', '[1, 2, 4, 6, 8, 12, 16, 24, 30]', 'ANCHOR_RATIOS', '[2]', 'MAX_NUM_GT_BOXES', '30']
      #args.set_cfgs = ['ANCHOR_SCALES', '[0.75, 1, 1.5, 2, 3, 4, 6, 8, 12, 16, 24, 30]', 'ANCHOR_RATIOS', '[2]', 'MAX_NUM_GT_BOXES', '30']

  elif args.dataset == "vg":
      # train sizes: train, smalltrain, minitrain
      # train scale: ['150-50-20', '150-50-50', '500-150-80', '750-250-150', '1750-700-450', '1600-400-20']
      args.imdb_name = "vg_150-50-50_minitrain"
      args.imdbval_name = "vg_150-50-50_minival"
      cfg.TRAIN.USE_FLIPPED = True
      cfg.dataset = args.dataset
      cfg.imdb_name = args.imdb_name
      args.set_cfgs = ['ANCHOR_SCALES', '[4, 8, 16, 32]', 'ANCHOR_RATIOS', '[0.5,1,2]', 'MAX_NUM_GT_BOXES', '50']

  elif args.dataset == "clipart" or args.dataset == "comic" or args.dataset == "watercolor" or args.dataset == 'cross_domain':
      if args.dataset == "clipart" :
        args.imdb_name = "clipart_train"
        args.imdbval_name = "clipart_test"
      elif args.dataset == "comic" :
        args.imdb_name = "comic_train"
        args.imdbval_name = "comic_test"
      elif args.dataset == "watercolor" :
        args.imdb_name = "watercolor_train"
        args.imdbval_name = "watercolor_test"
      elif args.dataset == 'cross_domain':
        args.imdb_name = "watercolor_train+clipart_train+comic_train"
        args.imdbval_name = "watercolor_test"
      # not_resume = False
      # if not args.resume:
      #   args.resume=True
      #   not_resume=True

      cfg.dataset = args.dataset
      cfg.TRAIN.SCALES=(600,)
      # cfg.TRAIN.BATCH_SIZE = 256
      # cfg.TRAIN.RPN_BATCHSIZE = 256
      cfg.RPN_POSITIVE_OVERLAP = 0.7
      cfg.TRAIN.FG_THRESH = 0.3
      cfg.TRAIN.USE_FLIPPED = True
      cfg.imdb_name = args.imdb_name
      cfg.filter_empty = True
      cfg.POOLING_SIZE_H = 7
      cfg.POOLING_SIZE_W = 7
      #args.set_cfgs = ['ANCHOR_SCALES', '[2.72, 3.81, 5.45, 7.64, 10.9, 15.27, 21.8, 32]', 'ANCHOR_RATIOS', '[2]', 'MAX_NUM_GT_BOXES', '20']
      args.set_cfgs = ['ANCHOR_SCALES', '[8, 16, 32]', 'ANCHOR_RATIOS', '[0.5, 1, 2]', 'MAX_NUM_GT_BOXES', '20']
      args.set_cfgs = ['ANCHOR_SCALES', '[0.75, 1, 1.5, 2, 3, 4, 6, 8, 12, 16, 24, 30]', 'ANCHOR_RATIOS', '[0.5, 1, 2]', 'MAX_NUM_GT_BOXES', '30']

  elif args.dataset == "KAISTVOC":
      args.imdb_name = "kaist_train"
      args.imdbval_name = "kaist_test"
      cfg.dataset = args.dataset
      cfg.imdb_name = args.imdb_name
      cfg.TRAIN.USE_FLIPPED = True 
      cfg.TRAIN.RPN_BATCHSIZE = 32
      cfg.RPN_POSITIVE_OVERLAP = 0.5
      cfg.TRAIN.RPN_NMS_THRESH = 0.65
      cfg.POOLING_SIZE_H = 8
      cfg.POOLING_SIZE_W = 4
      cfg.TRAIN.FG_THRESH = 0.45
      cfg.TRAIN.SCALES=(800,)
      cfg.sample_mode = 'bootstrap' # use bootstrap or ramdom as sampling method
      cfg.VGG_ORIGIN = False # whether use vgg original classification layers
      cfg.TRAIN.USE_ALL_GT = True # choose true if want to exclude all proposals overlap with 'people' larger than 0.3
      cfg.ignore_people = False # ignore people, all proposals overlap with 'people' larger than 0.3 will be igonored
      cfg.filter_empty = False # whether filter 0 gt images
      cfg.DEBUG = False # set as True if debug whether 'people' is ignored
      ## scales*11 is the new_width, new_width*ratio is new height
      # 30, 42, 60, 84, 120, 168, 240 width
      args.set_cfgs = ['ANCHOR_SCALES', '[2.72, 3.81, 5.45, 7.64, 10.9, 15.27, 21.8, 32]', 'ANCHOR_RATIOS', '[1, 2]', 'MAX_NUM_GT_BOXES', '20']
  
  elif args.dataset == "dota":
      args.imdb_name = "dota_train"
      args.imdbval_name = "dota_val"
      cfg.dataset = args.dataset
      cfg.TRAIN.SCALES=(600,)
      cfg.TRAIN.RPN_BATCHSIZE = 128 # SHOULD CHANGE ABOVE BUILDING MODEL BLOCKS
      cfg.TRAIN.BATCH_SIZE = 128   # SHOULD CHANGE ABOVE BUILDING MODEL BLOCKS
      cfg.TRAIN.USE_FLIPPED = True
      cfg.imdb_name = args.imdb_name
      cfg.POOLING_SIZE_H = 7
      cfg.POOLING_SIZE_W = 7
      args.set_cfgs = ['ANCHOR_SCALES', '[4, 8, 16, 32]', 'ANCHOR_RATIOS', '[0.5,1,2]', 'MAX_NUM_GT_BOXES', '20']
      args.set_cfgs = ['ANCHOR_SCALES', '[0.75, 1, 1.5, 2, 3, 4, 6, 8, 12, 16, 24, 30]', 'ANCHOR_RATIOS', '[0.5, 1, 2]', 'MAX_NUM_GT_BOXES', '100']

  elif args.dataset == "deeplesion":
      args.imdb_name = "deeplesion_trainval"
      args.imdbval_name = "deeplesion_test"
      cfg.dataset = args.dataset
      cfg.TRAIN.SCALES=(512,)
      cfg.TRAIN.RPN_BATCHSIZE = 128 # SHOULD CHANGE ABOVE BUILDING MODEL BLOCKS
      cfg.TRAIN.BATCH_SIZE = 64   # SHOULD CHANGE ABOVE BUILDING MODEL BLOCKS
      cfg.TRAIN.USE_FLIPPED = False
      cfg.imdb_name = args.imdb_name
      cfg.POOLING_SIZE_H = 7
      cfg.POOLING_SIZE_W = 7
      args.set_cfgs = ['ANCHOR_SCALES', '[4, 8, 16, 32]', 'ANCHOR_RATIOS', '[0.5,1,2]', 'MAX_NUM_GT_BOXES', '20']
      args.set_cfgs = ['ANCHOR_SCALES', '[1, 2, 4, 8, 16, 24, 32, 48, 96]', 'ANCHOR_RATIOS', '[0.5, 1, 2]', 'MAX_NUM_GT_BOXES', '10']  

  elif args.dataset == "LISA":
    args.imdb_name = "LISA_train"
    args.imdbval_name = "LISA_test"
    cfg.TRAIN.USE_FLIPPED = True
    # cfg.TRAIN.BATCH_SIZE = 256
    # cfg.TRAIN.RPN_BATCHSIZE = 256
    cfg.RPN_POSITIVE_OVERLAP = 0.5
    cfg.TRAIN.RPN_NMS_THRESH = 0.65
    cfg.TRAIN.FG_THRESH = 0.45
    cfg.POOLING_SIZE_H = 7
    cfg.POOLING_SIZE_W = 7
    cfg.dataset = args.dataset
    cfg.imdb_name = args.imdb_name
    cfg.TRAIN.SCALES=(800,)
    ## scales*11 is the new_width, new_width*ratio is new height
    # 30, 42, 60, 84, 120, 168, 240, 355
    args.set_cfgs = ['ANCHOR_SCALES', '[0.75, 1, 1.5, 2, 3, 4, 6, 8, 12, 16, 24, 30]', 'ANCHOR_RATIOS', '[0.5, 1, 2]', 'MAX_NUM_GT_BOXES', '50']
  ### CONFIG FILES
  args.cfg_file = "cfgs/{}_ls.yml".format(args.net) if args.large_scale else "cfgs/{}.yml".format(args.net)

  if args.cfg_file is not None:
    cfg_from_file(args.cfg_file)
  if args.set_cfgs is not None:
    cfg_from_list(args.set_cfgs)

  print('Using config:')
  pprint.pprint(cfg)
  np.random.seed(cfg.RNG_SEED)
  cfg.Only_FinetuneBN = False
  cfg.reinit_rpn = False
  cfg.nums = 0
  cfg.bnn = 0

  cfg.fix_bn = args.fix_bn == "True"
  if cfg.fix_bn: print("INFO: Fix batch normalization layers")
  else: print("INFO: Do not fix batch normalization layers")

  #torch.backends.cudnn.benchmark = True
  if torch.cuda.is_available() and not args.cuda:
    print("WARNING: You have a CUDA device, so you should probably run with --cuda")
  cfg.DATA_DIR = args.DATA_DIR

  # train set
  # -- Note: Use validation set and disable the flipped to enable faster loading.
  cfg.TRAIN.USE_FLIPPED = args.USE_FLIPPED == 'True'
  cfg.USE_GPU_NMS = args.cuda
  print('loading datasets:',args.imdb_name)
  imdb, roidb, ratio_list, ratio_index = combined_roidb(args.imdb_name)
  train_size = len(roidb)

  print('{:d} roidb entries'.format(len(roidb)))

  output_dir = args.save_dir + "/" + args.net + "/" + args.dataset
  if not os.path.exists(output_dir):
    os.makedirs(output_dir)

  sampler_batch = sampler(train_size, args.batch_size)

  dataset = roibatchLoader(roidb, ratio_list, ratio_index, args.batch_size, \
                           imdb.num_classes, datasets_name=args.dataset,training=True)

  dataloader = torch.utils.data.DataLoader(dataset, batch_size=args.batch_size,
                            sampler=sampler_batch, num_workers=args.num_workers)

  # initilize the tensor holder here.
  im_data = torch.FloatTensor(1)
  im_info = torch.FloatTensor(1)
  num_boxes = torch.LongTensor(1)
  gt_boxes = torch.FloatTensor(1)

  # ship to cuda
  if args.cuda:
    im_data = im_data.cuda()
    im_info = im_info.cuda()
    num_boxes = num_boxes.cuda()
    gt_boxes = gt_boxes.cuda()

  # make variable
  im_data = Variable(im_data)
  im_info = Variable(im_info)
  num_boxes = Variable(num_boxes)
  gt_boxes = Variable(gt_boxes)

  if args.cuda:
    cfg.CUDA = True

  # initilize the network here.
  if args.dataset == "KITTIVOC":
    cfg.TRAIN.RPN_BATCHSIZE = 256 # num of rois fot training rpn
    cfg.TRAIN.BATCH_SIZE = 128 # num of rois for training rcnn
  if args.dataset == "caltech":
    cfg.TRAIN.RPN_BATCHSIZE = 64 # num of rois fot training rpn
    cfg.TRAIN.BATCH_SIZE = 32 # num of rois for training rcnn
  if args.dataset == "KAISTVOC":
    cfg.TRAIN.RPN_BATCHSIZE = 128 # num of rois fot training rpn
    cfg.TRAIN.BATCH_SIZE = 32 # num of rois for training rcnn
  if args.dataset == 'citypersons':
    cfg.TRAIN.RPN_BATCHSIZE = 128 # num of rois fot training rpn
    cfg.TRAIN.BATCH_SIZE = 128   # num of rois for training rcnn
  if args.dataset == 'dota':
    cfg.TRAIN.RPN_BATCHSIZE = 128 # num of rois fot training rpn
    cfg.TRAIN.BATCH_SIZE = 128   # num of rois for training rcnn
  if args.dataset == 'deeplesion':
    cfg.TRAIN.RPN_BATCHSIZE = 64 # num of rois fot training rpn
    cfg.TRAIN.BATCH_SIZE = 32   # num of rois for training rcnn
  if args.dataset == 'LISA':
    cfg.TRAIN.RPN_BATCHSIZE = 64 # num of rois fot training rpn
    cfg.TRAIN.BATCH_SIZE = 32   # num of rois for training rcnn
  if args.net == 'vgg16':
    fasterRCNN = vgg16(imdb.classes, pretrained=True, class_agnostic=args.class_agnostic, batch_norm=False, rpn_batchsize=cfg.TRAIN.RPN_BATCHSIZE)
  elif args.net == 'vgg16_bn':
    fasterRCNN = vgg16(imdb.classes, pretrained=True, class_agnostic=args.class_agnostic, batch_norm=True, rpn_batchsize=cfg.TRAIN.RPN_BATCHSIZE)
  elif args.net == 'res101':
    fasterRCNN = resnet(imdb.classes, 101, pretrained=True, class_agnostic=args.class_agnostic,rpn_batchsize=cfg.TRAIN.RPN_BATCHSIZE)
  elif args.net == 'res50':
    fasterRCNN = resnet(imdb.classes, 50, pretrained=True, class_agnostic=args.class_agnostic,rpn_batchsize=cfg.TRAIN.RPN_BATCHSIZE)
  elif args.net == 'se_res50':
    fasterRCNN = se_resnet(imdb.classes, 50, pretrained=True, class_agnostic=args.class_agnostic,rpn_batchsize=cfg.TRAIN.RPN_BATCHSIZE)
  elif args.net == 'res34':
    fasterRCNN = resnet(imdb.classes, 34, pretrained=True, class_agnostic=args.class_agnostic,rpn_batchsize=cfg.TRAIN.RPN_BATCHSIZE)
  elif args.net == 'res18':
    cfg.RESNET.FIXED_BLOCKS = 0
    fasterRCNN = resnet(imdb.classes, 18, pretrained=True, class_agnostic=args.class_agnostic,rpn_batchsize=cfg.TRAIN.RPN_BATCHSIZE)
  elif args.net == 'res152':
    fasterRCNN = resnet(imdb.classes, 152, pretrained=True, class_agnostic=args.class_agnostic,rpn_batchsize=cfg.TRAIN.RPN_BATCHSIZE)
  elif args.net == 'vgg_bn':
    fasterRCNN = vgg16(imdb.classes, pretrained=True, class_agnostic=args.class_agnostic, batch_norm=True, rpn_batchsize=cfg.TRAIN.RPN_BATCHSIZE)
  else:
    print("network is not defined")
    pdb.set_trace()

  fasterRCNN.create_architecture()

  lr = cfg.TRAIN.LEARNING_RATE
  lr = args.lr
  #print('=============', cfg.TRAIN.FG_FRACTION)
  #tr_momentum = cfg.TRAIN.MOMENTUM
  #tr_momentum = args.momentum

  params = []
  for key, value in dict(fasterRCNN.named_parameters()).items():
    if value.requires_grad:
      if 'bias' in key:
        params += [{'params':[value],'lr':lr*(cfg.TRAIN.DOUBLE_BIAS + 1), \
                'weight_decay': cfg.TRAIN.BIAS_DECAY and cfg.TRAIN.WEIGHT_DECAY or 0}]
      else:
        params += [{'params':[value],'lr':lr, 'weight_decay': cfg.TRAIN.WEIGHT_DECAY}]

  if args.optimizer == "adam":
    lr = lr * 0.1
    optimizer = torch.optim.Adam(params)

  elif args.optimizer == "sgd":
    optimizer = torch.optim.SGD(params, momentum=cfg.TRAIN.MOMENTUM)

  if args.resume:
    if (args.dataset == "clipart" or args.dataset == "comic" or args.dataset == "watercolor" or args.dataset == 'cross_domain')\
      and not_resume:
      load_name = '/home/xuw080/data4/universal_model/models/res50/pascal_voc_0712/faster_rcnn_11_12_2067.pth'
      args.max_epochs = 12 + args.max_epochs 
    else:
      load_name = os.path.join(output_dir,
        'faster_rcnn_{}_{}_{}.pth'.format(args.checksession, args.checkepoch, args.checkpoint))
    print("loading checkpoint %s" % (load_name))
    checkpoint = torch.load(load_name)
    args.session = checkpoint['session']
    args.start_epoch = checkpoint['epoch']
    # if args.dataset == "clipart" or args.dataset == "comic" or args.dataset == "watercolor" or args.dataset == 'cross_domain':
    #   load_state_dict(fasterRCNN, checkpoint['model'])
    # else:
    fasterRCNN.load_state_dict(checkpoint['model'])
    optimizer.load_state_dict(checkpoint['optimizer'])
    lr = optimizer.param_groups[0]['lr']
    # if (args.dataset == "clipart" or args.dataset == "comic" or args.dataset == "watercolor")\
    #   and not_resume:
    print('Resumed lr is: ', lr)
    if 'pooling_mode' in checkpoint.keys():
      print('loading faster-rcnn based on: %s' %(checkpoint['pooling_mode']))
      cfg.POOLING_MODE = checkpoint['pooling_mode']
    print("loaded checkpoint %s" % (load_name))
  if cfg.sample_mode == 'bootstrap':
    print("INFO: using bootstrap as smapling mode")

  if args.mGPUs:
    fasterRCNN = nn.DataParallel(fasterRCNN)

  if args.cuda:
    fasterRCNN.cuda()

  iters_per_epoch = int(train_size / args.batch_size)
  
  adjust_learning_rate(optimizer, args.lr/lr)
  lr = args.lr
  print('Learning rate is: ', lr)
  for epoch in range(args.start_epoch, args.max_epochs + 1):
    # setting to train mode
    fasterRCNN.train()
    loss_temp = 0
    start = time.time()

    if epoch % (args.lr_decay_step + 1) == 0:
        adjust_learning_rate(optimizer, args.lr_decay_gamma)
        lr *= args.lr_decay_gamma

    data_iter = iter(dataloader)
    for step in range(iters_per_epoch):
      cfg.new_iter = True
      data = next(data_iter)
      #print('original size is: ',im_data.data.shape)
      im_data.data.resize_(data[0].size()).copy_(data[0])
      im_info.data.resize_(data[1].size()).copy_(data[1])
      gt_boxes.data.resize_(data[2].size()).copy_(data[2])
      num_boxes.data.resize_(data[3].size()).copy_(data[3])
      #print('after resize size is: ', im_data.data.shape)

      fasterRCNN.zero_grad()
      rois, cls_prob, bbox_pred, \
      rpn_loss_cls, rpn_loss_box, \
      RCNN_loss_cls, RCNN_loss_bbox, \
      rois_label = fasterRCNN(im_data, im_info, gt_boxes, num_boxes)

      loss = rpn_loss_cls.mean() + rpn_loss_box.mean() \
           + RCNN_loss_cls.mean() + RCNN_loss_bbox.mean()
      loss_temp += loss.data[0]

      cfg.current_num = cfg.TRAIN.RPN_BATCHSIZE
      cfg.new_iter = True

      # backward
      optimizer.zero_grad()
      loss.backward()
      if args.net == "vgg16":
          clip_gradient(fasterRCNN, 10.)
      optimizer.step()

      if step % args.disp_interval == 0:
        end = time.time()
        if step > 0:
          loss_temp /= args.disp_interval

        if args.mGPUs:
          loss_rpn_cls = rpn_loss_cls.mean().data[0]
          loss_rpn_box = rpn_loss_box.mean().data[0]
          loss_rcnn_cls = RCNN_loss_cls.mean().data[0]
          loss_rcnn_box = RCNN_loss_bbox.mean().data[0]
          fg_cnt = torch.sum(rois_label.data.ne(0))
          bg_cnt = rois_label.data.numel() - fg_cnt
        else:
          loss_rpn_cls = rpn_loss_cls.data[0]
          loss_rpn_box = rpn_loss_box.data[0]
          loss_rcnn_cls = RCNN_loss_cls.data[0]
          loss_rcnn_box = RCNN_loss_bbox.data[0]
          fg_cnt = torch.sum(rois_label.data.ne(0))
          bg_cnt = rois_label.data.numel() - fg_cnt

        print("[session %d][epoch %2d][iter %4d/%4d] loss: %.4f, lr: %.2e" \
                                % (args.session, epoch, step, iters_per_epoch, loss_temp, lr))
        print("\t\t\tfg/bg=(%d/%d), time cost: %f" % (fg_cnt, bg_cnt, end-start))
        print("\t\t\trpn_cls: %.4f, rpn_box: %.4f, rcnn_cls: %.4f, rcnn_box %.4f" \
                      % (loss_rpn_cls, loss_rpn_box, loss_rcnn_cls, loss_rcnn_box))
        if args.use_tfboard:
          info = {
            'loss': loss_temp,
            'loss_rpn_cls': loss_rpn_cls,
            'loss_rpn_box': loss_rpn_box,
            'loss_rcnn_cls': loss_rcnn_cls,
            'loss_rcnn_box': loss_rcnn_box
          }
          for tag, value in info.items():
            logger.scalar_summary(tag, value, step)

        loss_temp = 0

    if (epoch % 1 == 0) or (epoch == args.max_epochs - 1):
      if args.mGPUs:
        save_name = os.path.join(output_dir, 'faster_rcnn_{}_{}_{}.pth'.format(args.session, epoch, step))
        save_checkpoint({
          'session': args.session,
          'epoch': epoch + 1,
          'model': fasterRCNN.module.state_dict(),
          'optimizer': optimizer.state_dict(),
          'pooling_mode': cfg.POOLING_MODE,
          'class_agnostic': args.class_agnostic,
        }, save_name)
      else:
        save_name = os.path.join(output_dir, 'faster_rcnn_{}_{}_{}.pth'.format(args.session, epoch, step))
        save_checkpoint({
          'session': args.session,
          'epoch': epoch + 1,
          'model': fasterRCNN.state_dict(),
          'optimizer': optimizer.state_dict(),
          'pooling_mode': cfg.POOLING_MODE,
          'class_agnostic': args.class_agnostic,
        }, save_name)
      print('save model: {}'.format(save_name))

    end = time.time()
    print(end - start)
