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
      adjust_learning_rate, save_checkpoint, clip_gradient, update_chosen_se_layer, print_chosen_se_layer
from model.faster_rcnn.vgg16_uni import vgg16
from model.faster_rcnn.resnet_uni import resnet
from model.faster_rcnn.SEResNet_univ import seresnet
from model.faster_rcnn.vgg_16_bn_univ import VGG16_bn
from datasets.datasets_info import get_datasets_info
from model.faster_rcnn.SEResNet_Data_Attention import Datasets_Attention
from datasets.datasets_info import univ_info

import matplotlib
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm
from mpl_toolkits.mplot3d import Axes3D
from collections import defaultdict
import gc

def cycle(iterable):
    while True:
        for x in iterable:
            yield x

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
                        default=50, type=int)
    parser.add_argument('--checkpoint_interval', dest='checkpoint_interval',
                        help='number of iterations to display',
                        default=10000, type=int)
    parser.add_argument('--backward_together', dest='backward_together',
                        help='whether use original backward method, will update optimizer every datasets is finished if choose False',
                        default=10000, type=int)                    

    parser.add_argument('--save_dir', dest='save_dir',
                        help='directory to save models', default="/home/Xwang/HeadNode-1/universal_model_/models",
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
                        default=0.01, type=float)
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
    parser.add_argument('--USE_FLIPPED', dest='USE_FLIPPED',
                        help='whether use tensorflow tensorboard',
                        default=1, type=int)  
    parser.add_argument('--DATA_DIR', dest='DATA_DIR',
                        help='path to DATA_DIR',
                        default="/home/xuw080/data4/universal_model/data/", type=str)      
    parser.add_argument('--random_resize', dest='random_resize',
                        help='whether randomly resize images',
                        default="False", type=str)       
    parser.add_argument('--fix_bn', dest='fix_bn',
                        help='whether fix batch norm layer',
                        default="False", type=str)
    parser.add_argument('--use_mux', dest='use_mux',
                        help='whether use BN MUX',
                        default="False", type=str)    
    parser.add_argument('--domain_pred', dest='domain_pred',
                        help='whether add domain prediction loss for domain attention module',
                        default="False", type=str)
    args = parser.parse_args()
    return args                           

# Check whether although gradients is zero, data is still updated
def check_grad(model):
    for name, param in model.named_parameters():
        if param.requires_grad:
            if 'RCNN_cls_score_layers' in name and 'weight' in name:
                print(name, torch.sum(param.grad.data)*1000, torch.sum(param)*1000)

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

    if args.use_tfboard:
        from model.utils.logger import Logger
        # Set the logger
        logger = Logger('./logs')

    info_name = ['imdb_name','imdbval_name','dataset','imdb_name','USE_FLIPPED','RPN_BATCHSIZE','BATCH_SIZE','RPN_POSITIVE_OVERLAP','RPN_NMS_THRESH',\
        'POOLING_SIZE_H','POOLING_SIZE_W','FG_THRESH','SCALES','sample_mode','VGG_ORIGIN','USE_ALL_GT,ignore_people','filter_empty','DEBUG','set_cfgs']
    
    args.imdb_name, args.imdbval_name, args.dataset, cfg.imdb_name, cfg.TRAIN.USE_FLIPPED, cfg.TRAIN.RPN_BATCHSIZE, cfg.TRAIN.BATCH_SIZE, \
    cfg.TRAIN.RPN_POSITIVE_OVERLAP, cfg.TRAIN.RPN_NMS_THRESH, cfg.POOLING_SIZE_H, cfg.POOLING_SIZE_W, cfg.TRAIN.FG_THRESH, cfg.TRAIN.SCALES, \
    cfg.sample_mode, cfg.VGG_ORIGIN, cfg.USE_ALL_GT, cfg.ignore_people, cfg.filter_empty, cfg.DEBUG, args.set_cfgs \
    = get_datasets_info('pascal_voc_0712')

    cfg.datasets_list                 = ['KITTIVOC','widerface']#,'pascal_voc_0712','Kitchen','LISA']
    # cfg.datasets_list                 = ['LISA','pascal_voc_0712','Kitchen','coco','clipart','watercolor','comic','widerface','dota','deeplesion','KITTIVOC']
    cfg.imdb_name_list                = univ_info(cfg.datasets_list, 'imdb_name')
    cfg.imdbval_name_list             = univ_info(cfg.datasets_list, 'imdbval_name')
    cfg.train_scales_list             = univ_info(cfg.datasets_list, 'SCALES')
    cfg.train_rpn_batchsize_list      = univ_info(cfg.datasets_list, 'RPN_BATCHSIZE')
    cfg.train_batchsize_list          = univ_info(cfg.datasets_list, 'BATCH_SIZE')
    cfg.rpn_positive_overlap_list     = univ_info(cfg.datasets_list, 'RPN_POSITIVE_OVERLAP')
    cfg.rpn_nms_thresh_list           = univ_info(cfg.datasets_list, 'RPN_NMS_THRESH')
    cfg.train_fg_thresh_list          = univ_info(cfg.datasets_list, 'FG_THRESH')
    cfg.MAX_NUM_GT_BOXES_LIST         = univ_info(cfg.datasets_list, 'MAX_NUM_GT_BOXES')
    cfg.ANCHOR_SCALES_LIST            = univ_info(cfg.datasets_list, 'ANCHOR_SCALES')
    cfg.ANCHOR_RATIOS_LIST            = univ_info(cfg.datasets_list, 'ANCHOR_RATIOS')
    cfg.filp_image                    = univ_info(cfg.datasets_list, 'USE_FLIPPED')
    cfg.num_classes                   = univ_info(cfg.datasets_list, 'num_classes')
    cfg.universal = True
    cfg.DATA_DIR = args.DATA_DIR
    cfg.random_resize = args.random_resize == "True"
    
    # IMPORTANT:
    # For training widerface, we need to set anchor ratio be [1], or it is impossible to get good training results.
    args.set_cfgs = ['ANCHOR_SCALES', '[0.75, 1, 1.5, 2, 3, 4, 6, 8, 12, 16, 24, 30]', 'ANCHOR_RATIOS', '[0.5,1,2]', 'MAX_NUM_GT_BOXES', '300']
    cfg.POOLING_SIZE_H = 7
    cfg.POOLING_SIZE_W = 7
    cfg.ANCHOR_NUM = np.arange(len(cfg.num_classes))
    for i in range(len(cfg.num_classes)):
        cfg.ANCHOR_NUM[i] = len(cfg.ANCHOR_SCALES_LIST[i])*len(cfg.ANCHOR_RATIOS_LIST[i])

    cfg.epoch_diff = False
    fine_tune_single_datastes = False
    cfg.add_filter_num=0 # set 0 if do not add filters
    cfg.add_filter_ratio=0.0

    ## CONFIG FILES
    args.cfg_file = "cfgs/{}_ls.yml".format(args.net) if args.large_scale else "cfgs/{}.yml".format(args.net)
    cfg.has_people = False
    cfg.VGG_ORIGIN = True # If false, need to change pooling size of classfication layers size in vgg_16_bn_univ.py
    cfg.sample_mode = 'random'
    cfg.DEBUG = False # set as True if debug whether 'people' is ignored
    cfg.filter_empty = True
    cfg.ignore_people = False

    cfg.Only_FinetuneBN = False
    cfg.reinit_rpn = False
    cfg.nums = 0
    cfg.domain_pred = args.domain_pred == "True"

    if args.cfg_file is not None:
        cfg_from_file(args.cfg_file)
    if args.set_cfgs is not None:
        cfg_from_list(args.set_cfgs)
    cfg.plot_curve = False

    print('Using config:')
    pprint.pprint(cfg)
    np.random.seed(cfg.RNG_SEED)

    cfg.fix_bn = args.fix_bn == "True"
    if cfg.fix_bn: print("INFO: Fix batch normalization layers")
    else: print("INFO: Do not fix batch normalization layers")

    cfg.use_mux = args.use_mux == "True"
    if cfg.use_mux: print("INFO: Using BN MUX")
    else: print("INFO: Do not use BN MUX")

    #torch.backends.cudnn.benchmark = True
    if torch.cuda.is_available() and not args.cuda:
        print("WARNING: You have a CUDA device, so you should probably run with --cuda")
    print("INFO: Domain Prediction is: ", cfg.domain_pred)

    # train set
    # -- Note: Use validation set and disable the flipped to enable faster loading.
    cfg.USE_GPU_NMS = args.cuda
    dataloader_list = []
    im_data_list = []
    im_info_list = []
    num_boxes_list = []
    gt_boxes_list =[]
    iters_per_epoch_list = []
    cfg.imdb_list = {}
    
    for i in range(len(cfg.imdb_name_list)):
        cfg.dataset = cfg.datasets_list[i]
        args.imdb_name = cfg.imdb_name_list[i]
        args.imdbval_name = cfg.imdbval_name_list[i]
        cfg.TRAIN.RPN_BATCHSIZE = cfg.train_rpn_batchsize_list[i]
        cfg.TRAIN.BATCH_SIZE = cfg.train_batchsize_list[i]
        cfg.RPN_POSITIVE_OVERLAP = cfg.rpn_positive_overlap_list[i]
        cfg.TRAIN.RPN_NMS_THRESH = cfg.rpn_nms_thresh_list[i]
        cfg.TRAIN.FG_THRESH = cfg.train_fg_thresh_list[i]
        cfg.imdb_name = cfg.imdb_name_list[i]
        cfg.TRAIN.SCALES= cfg.train_scales_list[i]
        cfg.MAX_NUM_GT_BOXES = cfg.MAX_NUM_GT_BOXES_LIST[i]
        cfg.TRAIN.USE_FLIPPED = args.USE_FLIPPED == 1
        if 'deeplesion' == cfg.dataset:
            cfg.TRAIN.USE_FLIPPED = cfg.filp_image[i]
        cfg.ANCHOR_SCALES = cfg.ANCHOR_SCALES_LIST[i] 
        cfg.ANCHOR_RATIOS = cfg.ANCHOR_RATIOS_LIST[i]
        
        print('loading datasets:',args.imdb_name)

        imdb, roidb, ratio_list, ratio_index = combined_roidb(args.imdb_name)

        cfg.imdb_list[str(i)] = imdb

        train_size = len(roidb)

        print('{:d} roidb entries'.format(len(roidb)))

        output_dir = args.save_dir + "/" + args.net + "/universal"
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)
        
        sampler_batch = sampler(train_size, args.batch_size)
        dataset = roibatchLoader(roidb, ratio_list, ratio_index, args.batch_size, \
                                imdb.num_classes, datasets_name=cfg.dataset,training=True)

        dataloader = torch.utils.data.DataLoader(dataset, batch_size=args.batch_size,
                                sampler=sampler_batch, num_workers=args.num_workers, pin_memory=True)
        dataloader_list.append(dataloader)

        cfg.num_classes[i] = imdb.num_classes

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
        im_data_list.append(im_data)
        im_info_list.append(im_info)
        num_boxes_list.append(num_boxes)
        gt_boxes_list.append(gt_boxes)

        if args.cuda:
            cfg.CUDA = True
        iters_per_epoch = int(train_size / args.batch_size)
        iters_per_epoch_list.append(iters_per_epoch)

    for iters in range(len(iters_per_epoch_list)):
        print('iters_per_epoch for datasets {%s} is: {%d}'%(cfg.imdb_name_list[iters], iters_per_epoch_list[iters]))
        print('num of classes for datasets {%s} is: {%d}'%(cfg.imdb_name_list[iters], cfg.num_classes[iters]))
    # initilize the network here.
    if args.net == 'vgg16':
        fasterRCNN = vgg16(imdb.classes, pretrained=True, class_agnostic=args.class_agnostic, rpn_batchsize_list=cfg.train_batchsize_list)
    elif args.net == 'vgg16_bn':
        print(imdb.classes)
        fasterRCNN = VGG16_bn(imdb.classes, pretrained=True, class_agnostic=args.class_agnostic, batch_norm=True, rpn_batchsize_list=cfg.train_batchsize_list)
    elif args.net == 'res101':
        fasterRCNN = resnet(imdb.classes, 101, pretrained=True, class_agnostic=args.class_agnostic,rpn_batchsize_list=cfg.train_batchsize_list)
    elif args.net == 'res50':
        fasterRCNN = resnet(imdb.classes, 50, pretrained=True, class_agnostic=args.class_agnostic,rpn_batchsize_list=cfg.train_batchsize_list)
    elif args.net == 'se_res50':
        fasterRCNN = seresnet(imdb.classes, 50, pretrained=True, class_agnostic=args.class_agnostic,rpn_batchsize_list=cfg.train_batchsize_list, \
        se_loss=False, se_weight=1.0)
    elif  args.net == 'data_att_res50':
        fasterRCNN = Datasets_Attention(imdb.classes, 50, pretrained=True, class_agnostic=args.class_agnostic,rpn_batchsize_list=cfg.train_batchsize_list, \
        se_loss=False, se_weight=1.0)        
    elif  args.net == 'data_att_res18':
        fasterRCNN = Datasets_Attention(imdb.classes, 18, pretrained=True, class_agnostic=args.class_agnostic,rpn_batchsize_list=cfg.train_batchsize_list, \
        se_loss=False, se_weight=1.0)
    elif args.net == 'res18':
        cfg.RESNET.FIXED_BLOCKS = 0
        fasterRCNN = resnet(imdb.classes, 18, pretrained=True, class_agnostic=args.class_agnostic,rpn_batchsize_list=cfg.train_batchsize_list)
    elif args.net == 'res152':
        fasterRCNN = resnet(imdb.classes, 152, pretrained=True, class_agnostic=args.class_agnostic,rpn_batchsize_list=cfg.train_batchsize_list)
    else:
        print("network is not defined")
        pdb.set_trace()

    fasterRCNN.create_architecture()

    lr = args.lr

    args.backward_together = args.backward_together == 1
    if args.backward_together == 1:
        print('INFO: backward_together')
    else:
        print('INFO: backward after each datasets')

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
        load_name = os.path.join(output_dir,
            'faster_rcnn_universal_{}_{}_{}.pth'.format(args.checksession, args.checkepoch, args.checkpoint))
        print("loading checkpoint %s" % (load_name))
        checkpoint = torch.load(load_name)
        args.session = checkpoint['session']
        fasterRCNN.load_state_dict(checkpoint['model'])
        optimizer.load_state_dict(checkpoint['optimizer'])
        lr = optimizer.param_groups[0]['lr']
        if fine_tune_single_datastes:
            args.start_epoch = 0
            args.session = 11
            lr=0.002
        else:
            args.start_epoch = checkpoint['epoch']
        print('lr is: ', lr)
        if 'pooling_mode' in checkpoint.keys():
            print('loading faster-rcnn based on: %s' %(checkpoint['pooling_mode']))
            cfg.POOLING_MODE = checkpoint['pooling_mode']
        print("loaded checkpoint %s" % (load_name))

    if args.mGPUs:
        fasterRCNN = nn.DataParallel(fasterRCNN)

    if args.cuda:
        fasterRCNN.cuda()

    iters_per_epoch = max(iters_per_epoch_list)
    
    overall_n = 0
    overall_process = defaultdict(list)

    overall_name = dict()
    overall_loss = defaultdict(list)
    for i in range(len(cfg.num_classes)):
        overall_name[i] = cfg.datasets_list[i]

    plt.figure(figsize=(16, 12), dpi=80)
    plt.ion()
    
    for epoch in range(args.start_epoch, args.max_epochs):
        # setting to train mode
        fasterRCNN.train()
        start = time.time()
        epoch_start = time.time()
        if epoch % (args.lr_decay_step + 1) == 0:
            adjust_learning_rate(optimizer, args.lr_decay_gamma)
            lr *= args.lr_decay_gamma
        # build a dataloader iterations for each dataloader
        # Put them all into a new data_iter_list
        data_iter_list = []
        for dataloader in dataloader_list:
            dataloader.dataset.reset_target_size()
            #data_iter = iter(dataloader)
            data_iter = iter(cycle(dataloader))
            data_iter_list.append(data_iter)
            
        #### plot learning error
        samples = defaultdict(list)
        loss_temp = np.zeros((len(data_iter_list)))

        for step in range(iters_per_epoch):
            # build a data batch list for storing data batch
            # from different dataloader
            for cls_ind in range(len(data_iter_list)):
                samples[cls_ind].append(step)
                cfg.cls_ind = cls_ind
                cfg.TRAIN.BATCH_SIZE = cfg.train_batchsize_list[cls_ind]
                cfg.TRAIN.RPN_BATCHSIZE = cfg.train_rpn_batchsize_list[cls_ind]
                cfg.RPN_POSITIVE_OVERLAP = cfg.rpn_positive_overlap_list[cls_ind]
                cfg.TRAIN.RPN_NMS_THRESH = cfg.rpn_nms_thresh_list[cls_ind]
                cfg.TRAIN.FG_THRESH = cfg.train_fg_thresh_list[cls_ind]
                cfg.TRAIN.SCALES = cfg.train_scales_list[cls_ind]
                cfg.ANCHOR_SCALES = cfg.ANCHOR_SCALES_LIST[cls_ind]
                cfg.ANCHOR_RATIOS = cfg.ANCHOR_RATIOS_LIST[cls_ind]
                
                try:
                    data = next(data_iter_list[cls_ind])
                except:
                    #gc.collect()
                    # because train_loader first allocates the memory, and then assign it to x, so we need to delete these 
                    # variables manually.
                    print('INFO: All the data of datasets {%s} is trained, reset it now' %(cfg.imdb_name_list[cls_ind]))
                    data_iter_list[cls_ind] = iter(cycle(dataloader_list[cls_ind]))
                    #data_iter_list[cls_ind] = iter(dataloader_list[cls_ind])
                    data = next(data_iter_list[cls_ind])

                im_data_list[cls_ind].data = data[0].cuda()
                im_info_list[cls_ind].data = data[1].cuda()
                gt_boxes_list[cls_ind].data = data[2].cuda()
                num_boxes_list[cls_ind].data = data[3].cuda()

                fasterRCNN.zero_grad()

                rois, cls_prob, bbox_pred, \
                rpn_loss_cls, rpn_loss_box, \
                RCNN_loss_cls, RCNN_loss_bbox, \
                rois_label = fasterRCNN(im_data_list[cls_ind], im_info_list[cls_ind], gt_boxes_list[cls_ind], num_boxes_list[cls_ind], cls_ind)

                loss = rpn_loss_cls.mean() + rpn_loss_box.mean() \
                    + RCNN_loss_cls.mean() + RCNN_loss_bbox.mean()
                loss_temp[cls_ind] += loss.data.item()

                # backward after training all datasets
                if args.backward_together:
                    if cls_ind == 0:
                        optimizer.zero_grad()
                    loss.backward()
                    if args.net == "vgg16" :
                        clip_gradient(fasterRCNN, 10.)
                    if cls_ind == len(data_iter_list) - 1:
                        if 'data_att_res50' == args.net or 'data_att_res18' == args.net or 'data_att' in args.net:
                            update_chosen_se_layer(fasterRCNN, cls_ind)
                        optimizer.step()
                    #check_grad(fasterRCNN)
                # backward after training ONE datasets

                else:
                    optimizer.zero_grad()
                    loss.backward()
                    if args.net == "vgg16" or args.net == "vgg16_bn":
                        clip_gradient(fasterRCNN, 10.)
                    if 'data_att_res50' == args.net or 'data_att_res18' == args.net or 'data_att' in args.net:
                        update_chosen_se_layer(fasterRCNN, cls_ind)
                        # print_chosen_se_layer(fasterRCNN, cls_ind)
                    optimizer.step()
                    #check_grad(fasterRCNN)

                if step % args.disp_interval == 0:
                    end = time.time()
                    if step > 0:
                        loss_temp[cls_ind] /= args.disp_interval

                    if args.mGPUs:
                        loss_rpn_cls = rpn_loss_cls.mean().data.item()
                        loss_rpn_box = rpn_loss_box.mean().data.item()
                        loss_rcnn_cls = RCNN_loss_cls.mean().data.item()
                        loss_rcnn_box = RCNN_loss_bbox.mean().data.item()
                        fg_cnt = torch.sum(rois_label.data.ne(0))
                        bg_cnt = rois_label.data.numel() - fg_cnt
                    else:
                        loss_rpn_cls = rpn_loss_cls.data.item()
                        loss_rpn_box = rpn_loss_box.data.item()
                        loss_rcnn_cls = RCNN_loss_cls.data.item()
                        loss_rcnn_box = RCNN_loss_bbox.data.item()
                        fg_cnt = torch.sum(rois_label.data.ne(0))
                        bg_cnt = rois_label.data.numel() - fg_cnt
                    if cfg.epoch_diff:
                        print('num of coco is: %d, num of pascal is: %d' %(n_coco, n_pas))
                    print("[session %d][epoch %2d][iter %4d/%4d][datasets %d] loss: %.4f, lr: %.2e" \
                                            % (args.session, epoch, step, iters_per_epoch, cls_ind, loss_temp[cls_ind], lr))
                    print("\t\t\tfg/bg=(%d/%d), time cost: %f" % (fg_cnt, bg_cnt, end-start))
                    print("\t\t\trpn_cls: %.4f, rpn_box: %.4f, rcnn_cls: %.4f, rcnn_box %.4f" \
                                    % (loss_rpn_cls, loss_rpn_box, loss_rcnn_cls, loss_rcnn_box))
                    if args.use_tfboard:
                        info = {
                        'loss': loss_temp[cls_ind],
                        'loss_rpn_cls': loss_rpn_cls,
                        'loss_rpn_box': loss_rpn_box,
                        'loss_rcnn_cls': loss_rcnn_cls,
                        'loss_rcnn_box': loss_rcnn_box
                        }
                        for tag, value in info.items():
                            logger.scalar_summary(tag, value, step)

                    loss_temp[cls_ind] = 0
                    start = time.time()

            if (step+1) % int(iters_per_epoch/2) == 0:
                if args.mGPUs:
                    save_name = os.path.join(output_dir, 'faster_rcnn_universal_{}_{}_{}.pth'.format(args.session, epoch, step))
                    save_checkpoint({
                        'session': args.session,
                        'epoch': epoch + 1,
                        'model': fasterRCNN.module.state_dict(),
                        'optimizer': optimizer.state_dict(),
                        'pooling_mode': cfg.POOLING_MODE,
                        'class_agnostic': args.class_agnostic,
                    }, save_name)
                else:
                    save_name = os.path.join(output_dir, 'faster_rcnn_universal_{}_{}_{}.pth'.format(args.session, epoch, step))
                    save_checkpoint({
                        'session': args.session,
                        'epoch': epoch + 1,
                        'model': fasterRCNN.state_dict(),
                        'optimizer': optimizer.state_dict(),
                        'pooling_mode': cfg.POOLING_MODE,
                        'class_agnostic': args.class_agnostic,
                    }, save_name)
                print('save model: {}'.format(save_name))

        if args.mGPUs:
            save_name = os.path.join(output_dir, 'faster_rcnn_universal_{}_{}_{}.pth'.format(args.session, epoch, step))
            save_checkpoint({
                'session': args.session,
                'epoch': epoch + 1,
                'model': fasterRCNN.module.state_dict(),
                'optimizer': optimizer.state_dict(),
                'pooling_mode': cfg.POOLING_MODE,
                'class_agnostic': args.class_agnostic,
            }, save_name)
        else:
            save_name = os.path.join(output_dir, 'faster_rcnn_universal_{}_{}_{}.pth'.format(args.session, epoch, step))
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
        print(end - epoch_start)
