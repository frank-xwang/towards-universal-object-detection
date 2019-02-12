# --------------------------------------------------------
# Tensorflow Faster R-CNN
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
import cv2
import torch
from torch.autograd import Variable
import torch.nn as nn
import torch.optim as optim
import pickle
from roi_data_layer.roidb import combined_roidb
from roi_data_layer.roibatchLoader import roibatchLoader
from model.utils.config import cfg, cfg_from_file, cfg_from_list, get_output_dir
from model.rpn.bbox_transform import clip_boxes
from model.nms.nms_wrapper import nms
from model.rpn.bbox_transform import bbox_transform_inv
from model.utils.net_utils import save_net, load_net, vis_detections
from model.faster_rcnn.vgg16 import vgg16
from model.faster_rcnn.resnet import resnet
from model.faster_rcnn.SEResNet import se_resnet

import pdb

try:
    xrange          # Python 2
except NameError:
    xrange = range  # Python 3


def parse_args():
  """
  Parse input arguments
  """
  parser = argparse.ArgumentParser(description='Train a Fast R-CNN network')
  parser.add_argument('--dataset', dest='dataset',
                      help='training dataset',
                      default='pascal_voc', type=str)
  parser.add_argument('--TEST_MODEL', dest='TEST_MODEL',
                      help='testing model',
                      default='pascal_voc', type=str)
  parser.add_argument('--cfg', dest='cfg_file',
                      help='optional config file',
                      default='cfgs/vgg16.yml', type=str)
  parser.add_argument('--net', dest='net',
                      help='vgg16, res50, res101, res152',
                      default='res101', type=str)
  parser.add_argument('--set', dest='set_cfgs',
                      help='set config keys', default=None,
                      nargs=argparse.REMAINDER)
  parser.add_argument('--load_dir', dest='load_dir',
                      help='directory to load models', default="models",
                      type=str)
  parser.add_argument('--cuda', dest='cuda',
                      help='whether use CUDA',
                      action='store_true')
  parser.add_argument('--ls', dest='large_scale',
                      help='whether use large imag scale',
                      action='store_true')
  parser.add_argument('--mGPUs', dest='mGPUs',
                      help='whether use multiple GPUs',
                      action='store_true')
  parser.add_argument('--cag', dest='class_agnostic',
                      help='whether perform class_agnostic bbox regression',
                      action='store_true')
  parser.add_argument('--parallel_type', dest='parallel_type',
                      help='which part of model to parallel, 0: all, 1: model before roi pooling',
                      default=0, type=int)
  parser.add_argument('--checksession', dest='checksession',
                      help='checksession to load model',
                      default=1, type=int)
  parser.add_argument('--checkepoch', dest='checkepoch',
                      help='checkepoch to load network',
                      default=1, type=int)
  parser.add_argument('--checkpoint', dest='checkpoint',
                      help='checkpoint to load network',
                      default=10021, type=int)
  parser.add_argument('--bs', dest='batch_size',
                      help='batch_size',
                      default=1, type=int)
  parser.add_argument('--vis', dest='vis',
                      help='visualization mode',
                      action='store_true')
  args = parser.parse_args()
  return args

lr = cfg.TRAIN.LEARNING_RATE
momentum = cfg.TRAIN.MOMENTUM
weight_decay = cfg.TRAIN.WEIGHT_DECAY

if __name__ == '__main__':

  args = parse_args()

  print('Called with args:')
  print(args)
  cfg.VGG_ORIGIN = True
  if torch.cuda.is_available() and not args.cuda:
    print("WARNING: You have a CUDA device, so you should probably run with --cuda")

  np.random.seed(cfg.RNG_SEED)
  if args.dataset == "pascal_voc":
      args.imdb_name = "voc_2007_trainval"
      args.imdbval_name = "voc_2007_test"
      cfg.POOLING_SIZE_H = 7
      cfg.POOLING_SIZE_W = 7
      cfg.dataset = args.dataset
      #args.set_cfgs = ['ANCHOR_SCALES', '[8, 16, 32]', 'ANCHOR_RATIOS', '[0.5,1,2]']
      args.set_cfgs = ['ANCHOR_SCALES', '[8, 16, 32]', 'ANCHOR_RATIOS', '[0.5, 1, 2]']
      args.set_cfgs = ['ANCHOR_SCALES', '[0.75, 1, 1.5, 2, 3, 4, 6, 8, 12, 16, 24, 30]', 'ANCHOR_RATIOS', '[0.5, 1, 2]', 'MAX_NUM_GT_BOXES', '30']
  elif args.dataset == "pascal_voc_0712":
      args.imdb_name = "voc_2007_trainval+voc_2012_trainval"
      args.imdbval_name = "voc_2007_test"
      cfg.POOLING_SIZE_H = 7
      cfg.POOLING_SIZE_W = 7
      cfg.dataset = args.dataset
      args.set_cfgs = ['ANCHOR_SCALES', '[8, 16, 32]', 'ANCHOR_RATIOS', '[0.5,1,2]']
      args.set_cfgs = ['ANCHOR_SCALES', '[0.75, 1, 1.5, 2, 3, 4, 6, 8, 12, 16, 24, 30]', 'ANCHOR_RATIOS', '[0.5, 1, 2]', 'MAX_NUM_GT_BOXES', '30']
  elif args.dataset == "coco":
      args.imdb_name = "coco_2014_train+coco_2014_valminusminival"
      args.imdbval_name = "coco_2014_minival"
      cfg.POOLING_SIZE_H = 7
      cfg.POOLING_SIZE_W = 7
      cfg.dataset = args.dataset
      args.set_cfgs = ['ANCHOR_SCALES', '[4, 8, 16, 32]', 'ANCHOR_RATIOS', '[0.5,1,2]', 'MAX_NUM_GT_BOXES', '50']
      # args.set_cfgs = ['ANCHOR_SCALES', '[4, 8, 16, 32]', 'ANCHOR_RATIOS', '[0.5,1,2]']
  elif args.dataset == "imagenet":
      args.imdb_name = "imagenet_train"
      args.imdbval_name = "imagenet_val"
      cfg.POOLING_SIZE_H = 7
      cfg.POOLING_SIZE_W = 7
      args.set_cfgs = ['ANCHOR_SCALES', '[8, 16, 32]', 'ANCHOR_RATIOS', '[0.5,1,2]']
  elif args.dataset == "vg":
      args.imdb_name = "vg_150-50-50_minitrain"
      args.imdbval_name = "vg_150-50-50_minival"
      cfg.POOLING_SIZE_H = 7
      cfg.POOLING_SIZE_W = 7
      args.set_cfgs = ['ANCHOR_SCALES', '[4, 8, 16, 32]', 'ANCHOR_RATIOS', '[0.5,1,2]']
  elif args.dataset == "caltech":
      args.imdb_name = "caltech_test"
      args.imdbval_name = "caltech_test"      
      cfg.TRAIN.RPN_BATCHSIZE = 256
      cfg.TRAIN.BATCH_SIZE = 32
      cfg.RPN_POSITIVE_OVERLAP = 0.5
      cfg.TRAIN.RPN_NMS_THRESH = 0.7
      cfg.TRAIN.FG_THRESH = 0.5
      cfg.imdb_name = "caltech_test"
      cfg.POOLING_SIZE_H = 7
      cfg.POOLING_SIZE_W = 7
      cfg.TEST.SCALES=(720,)
      cfg.dataset = args.dataset
      cfg.VGG_ORIGIN = False
      ## scales*11 is the new_width, new_width*ratio is new height
      # 30, 42, 60, 84, 120, 168, 240
      args.set_cfgs = ['ANCHOR_SCALES', '[2.72, 3.81, 5.45, 7.64, 10.9, 15.27, 21.8, 32]', 'ANCHOR_RATIOS', '[1,2]', 'MAX_NUM_GT_BOXES', '20']
      args.set_cfgs = ['ANCHOR_SCALES', '[0.75, 1, 1.5, 2, 3, 4, 6, 8, 12, 16, 24, 30]', 'ANCHOR_RATIOS', '[1, 2]', 'MAX_NUM_GT_BOXES', '30']
      
  elif args.dataset == "KITTIVOC":
      args.imdb_name = "kittivoc_test"
      args.imdbval_name = "kittivoc_test"
      cfg.TEST.FG_THRESH = 0.5
      cfg.imdb_name = "kittivoc_test"
      cfg.POOLING_SIZE_H = 7
      cfg.POOLING_SIZE_W = 7
      cfg.TEST.SCALES=(576,)
      cfg.dataset = args.dataset
      ## scales*11 is the new_width, new_width*ratio is new height
      # 30, 42, 60, 84, 120, 168, 240, 355
      args.set_cfgs = ['ANCHOR_SCALES', '[2.72, 3.81, 5.45, 7.64, 10.9, 15.27, 21.8, 32]', 'ANCHOR_RATIOS', '[2]', 'MAX_NUM_GT_BOXES', '20']
      args.set_cfgs = ['ANCHOR_SCALES', '[0.75, 1, 1.5, 2, 3, 4, 6, 8, 12, 16, 24, 30]', 'ANCHOR_RATIOS', '[0.5, 1, 2]', 'MAX_NUM_GT_BOXES', '50']

  elif args.dataset == "widerface":
      args.imdb_name = "widerface_val"
      args.imdbval_name = "widerface_val"
      cfg.POOLING_SIZE_H = 7
      cfg.POOLING_SIZE_W = 7
      cfg.dataset = args.dataset
      cfg.imdb_name = "widerface_train"
      cfg.TEST.SCALES=(600,)
      cfg.sample_mode = 'bootstrap' # use bootstrap or ramdom as sampling method
      cfg.VGG_ORIGIN = True # whether use vgg original classification layers
      cfg.TRAIN.USE_ALL_GT = True # choose true if want to exclude all proposals overlap with 'people' larger than 0.3
      cfg.filter_empty = False # whether filter 0 gt images
      cfg.DEBUG = False # set as True if debug whether 'people' is ignored
      ## scales*11 is the new_width, new_width*ratio is new height
      # 30, 42, 60, 84, 120, 168, 240 width
      args.set_cfgs = ['ANCHOR_SCALES', '[0.75, 1, 1.5, 2, 3, 4, 6, 8, 12, 16, 24, 30]', 'ANCHOR_RATIOS', '[1]', 'MAX_NUM_GT_BOXES', '300']

  elif args.dataset == "citypersons":
      args.imdb_name = "citypersons_val"
      args.imdbval_name = "citypersons_val"
      cfg.POOLING_SIZE_H = 8
      cfg.POOLING_SIZE_W = 4
      cfg.dataset = args.dataset
      cfg.imdb_name = "citypersons_train"
      cfg.TEST.RPN_NMS_THRESH = 0.7
      cfg.TEST.NMS = 0.5 # original is 0.3, 0.5 is the same as official code
      cfg.TEST.SCALES=(1331,)
      cfg.sample_mode = 'ramdom' # use bootstrap or ramdom as sampling method
      cfg.VGG_ORIGIN = False # whether use vgg original classification layers
      cfg.TRAIN.USE_ALL_GT = True # choose true if want to exclude all proposals overlap with 'people' larger than 0.3
      cfg.filter_empty = False # whether filter 0 gt images
      cfg.DEBUG = False # set as True if debug whether 'people' is ignored
      ## scales*11 is the new_width, new_width*ratio is new height
      # 30, 42, 60, 84, 120, 168, 240 width
      args.set_cfgs = ['ANCHOR_SCALES', '[2, 4, 8, 10, 12, 16, 20, 24, 28, 32]', 'ANCHOR_RATIOS', '[1, 2]', 'MAX_NUM_GT_BOXES', '50']
      args.set_cfgs = ['ANCHOR_SCALES', '[0.75, 1, 1.5, 2, 3, 4, 6, 8, 12, 16, 24, 30]', 'ANCHOR_RATIOS', '[2]', 'MAX_NUM_GT_BOXES', '50']
      args.set_cfgs = ['ANCHOR_SCALES', '[1, 2, 4, 6, 8, 12, 16, 24, 30]', 'ANCHOR_RATIOS', '[2]', 'MAX_NUM_GT_BOXES', '30']

  elif args.dataset == "clipart" or args.dataset == "comic" or args.dataset == "watercolor":
      if args.dataset == "clipart" :
        args.imdb_name = "clipart_train"
        args.imdbval_name = "clipart_test"
      elif args.dataset == "comic" :
        args.imdb_name = "comic_train"
        args.imdbval_name = "comic_test"
      elif args.dataset == "watercolor" :
        args.imdb_name = "watercolor_train"
        args.imdbval_name = "watercolor_test"

      cfg.dataset = args.dataset
      cfg.TRAIN.SCALES=(600,)
      cfg.TRAIN.BATCH_SIZE = 256
      cfg.TRAIN.RPN_BATCHSIZE = 256
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
      cfg.TRAIN.BATCH_SIZE =128
      cfg.RPN_POSITIVE_OVERLAP = 0.5
      cfg.TEST.RPN_NMS_THRESH = 0.65
      cfg.POOLING_SIZE_H = 8
      cfg.POOLING_SIZE_W = 4
      cfg.TRAIN.FG_THRESH = 0.45
      cfg.TEST.SCALES=(720,)
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
      cfg.TRAIN.USE_FLIPPED = True
      cfg.imdb_name = args.imdb_name
      cfg.POOLING_SIZE_H = 7
      cfg.POOLING_SIZE_W = 7
      args.set_cfgs = ['ANCHOR_SCALES', '[4, 8, 16, 32]', 'ANCHOR_RATIOS', '[0.5,1,2]', 'MAX_NUM_GT_BOXES', '20']
      args.set_cfgs = ['ANCHOR_SCALES', '[0.75, 1, 1.5, 2, 3, 4, 6, 8, 12, 16, 24, 30]', 'ANCHOR_RATIOS', '[0.5, 1, 2]', 'MAX_NUM_GT_BOXES', '100']

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

  elif args.dataset == "deeplesion":
        args.imdb_name = "deeplesion_trainval"
        args.imdbval_name = "deeplesion_test"
        cfg.dataset = args.dataset
        cfg.TRAIN.SCALES=(512,)
        cfg.TRAIN.RPN_BATCHSIZE = 128 # SHOULD CHANGE ABOVE BUILDING MODEL BLOCKS
        cfg.TRAIN.BATCH_SIZE = 64   # SHOULD CHANGE ABOVE BUILDING MODEL BLOCKS
        cfg.TRAIN.USE_FLIPPED = True
        cfg.imdb_name = args.imdb_name
        cfg.POOLING_SIZE_H = 7
        cfg.POOLING_SIZE_W = 7
        cfg.TEST.RPN_MIN_SIZE = 16 # original is 16
        cfg.TEST.RPN_NMS_THRESH = 0.7 # original is 0.7
        args.set_cfgs = ['ANCHOR_SCALES', '[4, 8, 16, 32]', 'ANCHOR_RATIOS', '[0.5,1,2]', 'MAX_NUM_GT_BOXES', '20']
        args.set_cfgs = ['ANCHOR_SCALES', '[1, 2, 4, 8, 16, 24, 32, 48, 96]', 'ANCHOR_RATIOS', '[0.5, 1, 2]', 'MAX_NUM_GT_BOXES', '10']  

  elif args.dataset == "LISA":
    args.imdb_name = "LISA_train"
    args.imdbval_name = "LISA_test"
    cfg.TRAIN.USE_FLIPPED = True
    # cfg.TRAIN.BATCH_SIZE = 256
    # cfg.TRAIN.RPN_BATCHSIZE = 256
    cfg.POOLING_SIZE_H = 7
    cfg.POOLING_SIZE_W = 7
    cfg.dataset = args.dataset
    cfg.imdb_name = args.imdb_name
    cfg.TRAIN.SCALES=(800,)
    ## scales*11 is the new_width, new_width*ratio is new height
    # 30, 42, 60, 84, 120, 168, 240, 355
    args.set_cfgs = ['ANCHOR_SCALES', '[0.75, 1, 1.5, 2, 3, 4, 6, 8, 12, 16, 24, 30]', 'ANCHOR_RATIOS', '[0.5, 1, 2]', 'MAX_NUM_GT_BOXES', '50']
 
  args.cfg_file = "cfgs/{}_ls.yml".format(args.net) if args.large_scale else "cfgs/{}.yml".format(args.net)

  if args.cfg_file is not None:
    cfg_from_file(args.cfg_file)
  if args.set_cfgs is not None:
    cfg_from_list(args.set_cfgs)

  cfg.fix_bn = True
  cfg.nums = 0
  cfg.bnn = 0
  cfg.random_resize = False
  
  print('Using config:')
  pprint.pprint(cfg)

  cfg.TRAIN.USE_FLIPPED = False
  imdb, roidb, ratio_list, ratio_index = combined_roidb(args.imdbval_name, False)
  imdb.competition_mode(on=True)

  print('{:d} roidb entries'.format(len(roidb)))

  input_dir = args.load_dir + "/" + args.net + "/" + args.TEST_MODEL
  if not os.path.exists(input_dir):
    raise Exception('There is no input directory for loading network from ' + input_dir)
  load_name = os.path.join(input_dir,
    'faster_rcnn_{}_{}_{}.pth'.format(args.checksession, args.checkepoch, args.checkpoint))

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
    cfg.TRAIN.RPN_BATCHSIZE = 128 # num of rois fot training rpn
    cfg.TRAIN.BATCH_SIZE = 64  # num of rois for training rcnn
    cfg.TEST.RPN_MIN_SIZE = 16 # original is 16
    cfg.TEST.RPN_NMS_THRESH = 0.7 # original is 0.7
  # initilize the network here.
  if args.net == 'vgg16':
    fasterRCNN = vgg16(imdb.classes, pretrained=False, class_agnostic=args.class_agnostic, batch_norm =False, rpn_batchsize=cfg.TRAIN.RPN_BATCHSIZE)
  elif args.net == 'vgg16_bn':
    fasterRCNN = vgg16(imdb.classes, pretrained=False, class_agnostic=args.class_agnostic, batch_norm =True, rpn_batchsize=cfg.TRAIN.RPN_BATCHSIZE)
  elif args.net == 'res101':
    fasterRCNN = resnet(imdb.classes, 101, pretrained=False, class_agnostic=args.class_agnostic,rpn_batchsize=cfg.TRAIN.RPN_BATCHSIZE)
  elif args.net == 'res50':
    fasterRCNN = resnet(imdb.classes, 50, pretrained=False, class_agnostic=args.class_agnostic,rpn_batchsize=cfg.TRAIN.RPN_BATCHSIZE)
  elif args.net == 'se_res50':
    fasterRCNN = se_resnet(imdb.classes, 50, pretrained=False, class_agnostic=args.class_agnostic,rpn_batchsize=cfg.TRAIN.RPN_BATCHSIZE)
  elif args.net == 'res18':
    fasterRCNN = resnet(imdb.classes, 18, pretrained=False, class_agnostic=args.class_agnostic,rpn_batchsize=cfg.TRAIN.RPN_BATCHSIZE)
  elif args.net == 'res152':
    fasterRCNN = resnet(imdb.classes, 152, pretrained=False, class_agnostic=args.class_agnostic,rpn_batchsize=cfg.TRAIN.RPN_BATCHSIZE)
  else:
    print("network is not defined")
    pdb.set_trace()
  
  fasterRCNN.create_architecture()

  print("load checkpoint %s" % (load_name))
  checkpoint = torch.load(load_name)
  fasterRCNN.load_state_dict(checkpoint['model'])
  if 'pooling_mode' in checkpoint.keys():
    cfg.POOLING_MODE = checkpoint['pooling_mode']

  print('load model successfully!')
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
  im_data = Variable(im_data, volatile=True)
  im_info = Variable(im_info, volatile=True)
  num_boxes = Variable(num_boxes, volatile=True)
  gt_boxes = Variable(gt_boxes, volatile=True)

  if args.cuda:
    cfg.CUDA = True

  if args.cuda:
    fasterRCNN.cuda()

  start = time.time()
  max_per_image = 100

  vis = args.vis

  if vis:
    thresh = 0.05
  else:
    thresh = 0.0

  save_name = 'faster_rcnn_10'
  num_images = len(imdb.image_index)
  all_boxes = [[[] for _ in xrange(num_images)]
               for _ in xrange(imdb.num_classes)]

  output_dir = get_output_dir(imdb, save_name)
  dataset = roibatchLoader(roidb, ratio_list, ratio_index, args.batch_size, \
                        imdb.num_classes, training=False, normalize = False)
  dataloader = torch.utils.data.DataLoader(dataset, batch_size=args.batch_size,
                            shuffle=False, num_workers=0,
                            pin_memory=True)

  data_iter = iter(dataloader)

  _t = {'im_detect': time.time(), 'misc': time.time()}
  det_file = os.path.join(output_dir, 'detections.pkl')

  fasterRCNN.eval()
  empty_array = np.transpose(np.array([[],[],[],[],[]]), (1,0))

  for i in range(num_images):

      data = next(data_iter)
      im_data.data.resize_(data[0].size()).copy_(data[0])
      im_info.data.resize_(data[1].size()).copy_(data[1])
      gt_boxes.data.resize_(data[2].size()).copy_(data[2])
      num_boxes.data.resize_(data[3].size()).copy_(data[3])

      det_tic = time.time()
      rois, cls_prob, bbox_pred, \
      rpn_loss_cls, rpn_loss_box, \
      RCNN_loss_cls, RCNN_loss_bbox, \
      rois_label = fasterRCNN(im_data, im_info, gt_boxes, num_boxes)

      scores = cls_prob.data
      boxes = rois.data[:, :, 1:5]

      if cfg.TEST.BBOX_REG:
          # Apply bounding-box regression deltas
          box_deltas = bbox_pred.data
          if cfg.TRAIN.BBOX_NORMALIZE_TARGETS_PRECOMPUTED:
          # Optionally normalize targets by a precomputed mean and stdev
            if args.class_agnostic:
                box_deltas = box_deltas.view(-1, 4) * torch.FloatTensor(cfg.TRAIN.BBOX_NORMALIZE_STDS).cuda() \
                           + torch.FloatTensor(cfg.TRAIN.BBOX_NORMALIZE_MEANS).cuda()
                box_deltas = box_deltas.view(1, -1, 4)
            else:
                box_deltas = box_deltas.view(-1, 4) * torch.FloatTensor(cfg.TRAIN.BBOX_NORMALIZE_STDS).cuda() \
                           + torch.FloatTensor(cfg.TRAIN.BBOX_NORMALIZE_MEANS).cuda()
                box_deltas = box_deltas.view(1, -1, 4 * len(imdb.classes))

          pred_boxes = bbox_transform_inv(boxes, box_deltas, 1)
          pred_boxes = clip_boxes(pred_boxes, im_info.data, 1)
      else:
          # Simply repeat the boxes, once for each class
          pred_boxes = np.tile(boxes, (1, scores.shape[1]))

      pred_boxes /= data[1][0][2].cuda()

      scores = scores.squeeze()
      pred_boxes = pred_boxes.squeeze()
      det_toc = time.time()
      detect_time = det_toc - det_tic
      misc_tic = time.time()
      if vis:
          im = cv2.imread(imdb.image_path_at(i))
          im2show = np.copy(im)
      for j in xrange(1, imdb.num_classes):
          inds = torch.nonzero(scores[:,j]>thresh).view(-1)
          # if there is det
          if inds.numel() > 0:
            cls_scores = scores[:,j][inds]
            _, order = torch.sort(cls_scores, 0, True)
            if args.class_agnostic:
              cls_boxes = pred_boxes[inds, :]
            else:
              cls_boxes = pred_boxes[inds][:, j * 4:(j + 1) * 4]
            
            cls_dets = torch.cat((cls_boxes, cls_scores.unsqueeze(1)), 1)
            # cls_dets = torch.cat((cls_boxes, cls_scores), 1)
            cls_dets = cls_dets[order]
            keep = nms(cls_dets, cfg.TEST.NMS)
            cls_dets = cls_dets[keep.view(-1).long()]
            if vis:
              im2show = vis_detections(im2show, imdb.classes[j], cls_dets.cpu().numpy(), 0.3)
            all_boxes[j][i] = cls_dets.cpu().numpy()
          else:
            all_boxes[j][i] = empty_array

      # Limit to max_per_image detections *over all classes*
      if max_per_image > 0:
          image_scores = np.hstack([all_boxes[j][i][:, -1]
                                    for j in xrange(1, imdb.num_classes)])
          if len(image_scores) > max_per_image:
              image_thresh = np.sort(image_scores)[-max_per_image]
              for j in xrange(1, imdb.num_classes):
                  keep = np.where(all_boxes[j][i][:, -1] >= image_thresh)[0]
                  all_boxes[j][i] = all_boxes[j][i][keep, :]

      misc_toc = time.time()
      nms_time = misc_toc - misc_tic

      sys.stdout.write('im_detect: {:d}/{:d} {:.3f}s {:.3f}s   \r' \
          .format(i + 1, num_images, detect_time, nms_time))
      sys.stdout.flush()

      if vis:
          cv2.imwrite('result.png', im2show)
          pdb.set_trace()
          #cv2.imshow('test', im2show)
          #cv2.waitKey(0)

  with open(det_file, 'wb') as f:
      pickle.dump(all_boxes, f, pickle.HIGHEST_PROTOCOL)

  print('Evaluating detections')
  imdb.evaluate_detections(all_boxes, output_dir)

  end = time.time()
  print("test time: %0.4fs" % (end - start))
