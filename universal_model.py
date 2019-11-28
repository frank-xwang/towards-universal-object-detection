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
from model.faster_rcnn.resnet_uni import resnet
from datasets.datasets_info import get_datasets_info
from model.faster_rcnn.DAResNet import Domain_Attention
from datasets.datasets_info import univ_info
import torch.nn.functional as F

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
                    help='res50',
                    default='res50', type=str)
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
                        default="data/", type=str)      
    parser.add_argument('--random_resize', dest='random_resize',
                        help='whether randomly resize images',
                        default="False", type=str)       
    parser.add_argument('--fix_bn', dest='fix_bn',
                        help='whether fix batch norm layer',
                        default="False", type=str)
    parser.add_argument('--use_mux', dest='use_mux',
                        help='whether use BN MUX',
                        default="False", type=str)    
    parser.add_argument('--randomly_chosen_datasets', dest='randomly_chosen_datasets',
                        help='Whether randomly choose datasets',
                        default="False", type=str)
    parser.add_argument('--update_chosen', dest='update_chosen',
                        help='Whether update chosen layers',
                        default="False", type=str)
    parser.add_argument('--warmup_steps', dest='warmup_steps',
                            help='Whether use warm up',
                            default=0, type=int)
    parser.add_argument('--less_blocks', dest='less_blocks',
                        help='Whether use less blocks',
                        default='False', type=str)
    parser.add_argument('--num_adapters', dest='num_adapters',
                        help='Number of se layers adapter',
                        default=0, type=int)    
    parser.add_argument('--rpn_univ', dest='rpn_univ',
                        help='Whether use universal rpn',
                        default='False', type=str)
    parser.add_argument('--datasets_list', nargs='+', 
                    help='datasets list for training', required=True)
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

    args.imdb_name, args.imdbval_name, args.dataset, cfg.imdb_name, cfg.TRAIN.USE_FLIPPED, cfg.TRAIN.RPN_BATCHSIZE, cfg.TRAIN.BATCH_SIZE, cfg.TEST.BATCH_SIZE,\
    cfg.TRAIN.RPN_POSITIVE_OVERLAP, cfg.TRAIN.RPN_NMS_THRESH, cfg.POOLING_SIZE_H, cfg.POOLING_SIZE_W, cfg.TRAIN.FG_THRESH, cfg.TRAIN.SCALES, \
    cfg.sample_mode, cfg.VGG_ORIGIN, cfg.USE_ALL_GT, cfg.ignore_people, cfg.filter_empty, cfg.DEBUG, args.set_cfgs \
    = get_datasets_info('pascal_voc_0712')

    cfg.datasets_list                 = args.datasets_list # ['KITTI','widerface','pascal_voc_0712','Kitchen','LISA']
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

    cfg.POOLING_SIZE_H = 7
    cfg.POOLING_SIZE_W = 7
    cfg.ANCHOR_NUM = np.arange(len(cfg.num_classes))
    cfg.random_resize = args.random_resize == "True"
    args.update_chosen = args.update_chosen == 'True'
    for i in range(len(cfg.num_classes)):
        cfg.ANCHOR_NUM[i] = len(cfg.ANCHOR_SCALES_LIST[i])*len(cfg.ANCHOR_RATIOS_LIST[i])

    cfg.rpn_univ = args.rpn_univ == 'True'
    if cfg.rpn_univ:
        cfg.ANCHOR_SCALES_LIST = [cfg.ANCHOR_SCALES]
        cfg.ANCHOR_RATIOS_LIST = [cfg.ANCHOR_RATIOS]
        cfg.ANCHOR_NUM = [len(cfg.ANCHOR_SCALES)*len(cfg.ANCHOR_RATIOS)]

    ## CONFIG FILES
    args.cfg_file = "cfgs/{}_ls.yml".format(args.net) if args.large_scale else "cfgs/{}.yml".format(args.net)
    cfg.sample_mode = 'random'
    cfg.filter_empty = True

    if args.cfg_file is not None:
        cfg_from_file(args.cfg_file)
    if args.set_cfgs is not None:
        cfg_from_list(args.set_cfgs)
    cfg.plot_curve = False
    np.random.seed(cfg.RNG_SEED)

    cfg.fix_bn = args.fix_bn == "True"
    if cfg.fix_bn: print("INFO: Fix batch normalization layers")
    else: print("INFO: Do not fix batch normalization layers")

    cfg.use_mux = args.use_mux == "True"
    if cfg.use_mux: print("INFO: Using BN MUX")
    else: print("INFO: Do not use BN MUX")
    
    cfg.less_blocks = args.less_blocks == 'True'
    if cfg.less_blocks: print('INFO: Using less blocks')
    cfg.num_adapters = args.num_adapters
    print("INFO: number of adapter is: ", cfg.num_adapters)
    cfg.mode = 'universal'
    if args.update_chosen:
        print('INFO: Update chosen layers')

    torch.backends.cudnn.benchmark = True
    if torch.cuda.is_available() and not args.cuda:
        print("WARNING: You have a CUDA device, so you should probably run with --cuda")
    # print('Using config:')
    # pprint.pprint(cfg)

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
        # Deeplesion will not flip as in https://arxiv.org/pdf/1711.10535.pdf
        if 'deeplesion' == cfg.dataset: 
            cfg.TRAIN.USE_FLIPPED = cfg.filp_image[i]
        cfg.ANCHOR_SCALES = cfg.ANCHOR_SCALES_LIST[i]
        cfg.ANCHOR_RATIOS = cfg.ANCHOR_RATIOS_LIST[i]
        
        print('loading datasets:',args.imdb_name)

        imdb, roidb, ratio_list, ratio_index = combined_roidb(args.imdb_name)

        cfg.imdb_list[str(i)] = imdb

        train_size = len(roidb)


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
    fasterRCNN = Domain_Attention(imdb.classes, 50, pretrained=True, class_agnostic=args.class_agnostic, \
                                    rpn_batchsize_list=cfg.train_batchsize_list)
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
        args.start_epoch = checkpoint['epoch']
        print("loaded checkpoint %s" % (load_name))

    if args.mGPUs:
        fasterRCNN = nn.DataParallel(fasterRCNN)

    if args.cuda:
        fasterRCNN.cuda()

    args.randomly_chosen_datasets = args.randomly_chosen_datasets == 'True'
    if args.randomly_chosen_datasets: 
        print('INFO: Randomly choose datasets during training')
    else: 
        print('INFO: Train every datasets equal times')
    if args.randomly_chosen_datasets:
        total_iters = 0
        for iters_num in iters_per_epoch_list:
            total_iters += iters_num
        datasets_ids = list(range(total_iters))
        datasets_start = 0
        for index, iters in enumerate(iters_per_epoch_list):
            datasets_ids[datasets_start:datasets_start  + iters] = [index for temp in list(range(iters))]
            datasets_start += iters
        np.random.shuffle(datasets_ids)
        iters_per_epoch = total_iters
    else:
        iters_per_epoch = max(iters_per_epoch_list)*len(iters_per_epoch_list)

    base_lr = lr
    for epoch in range(args.start_epoch, args.max_epochs):
        # setting to train mode
        fasterRCNN.train()
        start = time.time()
        epoch_start = time.time()
        if args.randomly_chosen_datasets:
            np.random.shuffle(datasets_ids)
        if epoch % (args.lr_decay_step + 1) == 0:
            adjust_learning_rate(optimizer, args.lr_decay_gamma)
            lr *= args.lr_decay_gamma

        data_iter_list = []
        for dataloader in dataloader_list:
            dataloader.dataset.reset_target_size()
            data_iter = iter(cycle(dataloader))
            data_iter_list.append(data_iter)
            
        loss_temp = np.zeros((len(data_iter_list)))
        step_list = list(np.zeros([len(data_iter_list)]))
        step_list = [0 for i in step_list]

        for step in range(iters_per_epoch):
            if epoch == 1 and step <= args.warmup_steps - 1:
                lr_decay_gamma = base_lr * (step + 1) / (args.warmup_steps)
                adjust_learning_rate(optimizer, lr_decay_gamma)
                lr = base_lr * lr_decay_gamma
            elif epoch == 1:
                lr_decay_gamma = base_lr/lr
                adjust_learning_rate(optimizer, lr_decay_gamma)
                lr = lr * lr_decay_gamma

            # build a data batch list for storing data batch
            # from different dataloader
            if args.randomly_chosen_datasets:
                cls_ind = datasets_ids[step]
            else:
                cls_ind = step % len(iters_per_epoch_list)
            step_list[cls_ind] += 1

            cfg.cls_ind = cls_ind
            cfg.TRAIN.BATCH_SIZE = cfg.train_batchsize_list[cls_ind]
            cfg.TRAIN.RPN_BATCHSIZE = cfg.train_rpn_batchsize_list[cls_ind]
            cfg.RPN_POSITIVE_OVERLAP = cfg.rpn_positive_overlap_list[cls_ind]
            cfg.TRAIN.RPN_NMS_THRESH = cfg.rpn_nms_thresh_list[cls_ind]
            cfg.TRAIN.FG_THRESH = cfg.train_fg_thresh_list[cls_ind]
            cfg.TRAIN.SCALES = cfg.train_scales_list[cls_ind]
            index_ = cls_ind
            cfg.ANCHOR_SCALES = cfg.ANCHOR_SCALES_LIST[index_]
            cfg.ANCHOR_RATIOS = cfg.ANCHOR_RATIOS_LIST[index_]
            
            try:
                data = next(data_iter_list[cls_ind])
            except:
                print('INFO: All the data of datasets {%s} is trained, reset it now' %(cfg.imdb_name_list[cls_ind]))
                data_iter_list[cls_ind] = iter(cycle(dataloader_list[cls_ind]))
                data = next(data_iter_list[cls_ind])

            im_data_list[cls_ind].data = data[0].cuda()
            im_info_list[cls_ind].data = data[1].cuda()
            gt_boxes_list[cls_ind].data = data[2].cuda()
            num_boxes_list[cls_ind].data = data[3].cuda()

            fasterRCNN.zero_grad()

            rois, cls_prob, bbox_pred, \
            rpn_loss_cls, rpn_loss_box, \
            RCNN_loss_cls, RCNN_loss_bbox, \
            rois_label = fasterRCNN(im_data_list[cls_ind], im_info_list[cls_ind],\
                         gt_boxes_list[cls_ind], num_boxes_list[cls_ind], cls_ind)

            loss = rpn_loss_cls.mean() + rpn_loss_box.mean() \
                + RCNN_loss_cls.mean() + RCNN_loss_bbox.mean()
            loss_temp[cls_ind] += loss.data.item()

            # backward after training all datasets
            if args.backward_together:
                if cls_ind == 0:
                    optimizer.zero_grad()
                loss.backward()

                if cls_ind == len(data_iter_list) - 1:
                    if args.update_chosen and 'data_att' in args.net:
                        update_chosen_se_layer(fasterRCNN, cls_ind)
                    optimizer.step()
            else:
                optimizer.zero_grad()
                loss.backward()
                if args.update_chosen and 'data_att' in args.net:
                    update_chosen_se_layer(fasterRCNN, cls_ind)
                optimizer.step()

            if step_list[cls_ind] % args.disp_interval == 0:
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

                print("[session %d][epoch %2d][iter %4d/%4d/%4d][datasets %d] loss: %.4f, lr: %.2e" \
                                        % (args.session, epoch,  step_list[cls_ind], iters_per_epoch_list[cls_ind], iters_per_epoch, cls_ind, loss_temp[cls_ind], lr))
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

        if args.mGPUs:
            save_name = os.path.join(output_dir, 'faster_rcnn_universal_{}_{}_{}.pth'.format(args.session, epoch, step))
            save_checkpoint({
                'session': args.session,
                'epoch': epoch + 1,
                'model': fasterRCNN.module.state_dict(),
                'optimizer': optimizer.state_dict(),
                'class_agnostic': args.class_agnostic,
            }, save_name)
        else:
            save_name = os.path.join(output_dir, 'faster_rcnn_universal_{}_{}_{}.pth'.format(args.session, epoch, step))
            save_checkpoint({
                'session': args.session,
                'epoch': epoch + 1,
                'model': fasterRCNN.state_dict(),
                'optimizer': optimizer.state_dict(),
                'class_agnostic': args.class_agnostic,
            }, save_name)
        print('save model: {}'.format(save_name))

        end = time.time()
        print(end - epoch_start)