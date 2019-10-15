from __future__ import absolute_import
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable

from model.utils.config import cfg
from .proposal_layer import _ProposalLayer
from .anchor_target_layer import _AnchorTargetLayer
from model.utils.net_utils import _smooth_l1_loss

import numpy as np
import math
import pdb
import time

class _RPN(nn.Module):
    """ region proposal network """
    def __init__(self, din, rpn_batchsize_list):
        super(_RPN, self).__init__()

        self.din = din # get depth of input feature map, e.g., 512
        self.anchor_scales = cfg.ANCHOR_SCALES
        self.anchor_ratios = cfg.ANCHOR_RATIOS
        self.feat_stride = cfg.FEAT_STRIDE[0]
        #self.rpn_batchsize = rpn_batchsize
        self.rpn_batchsize_list = rpn_batchsize_list

        # define the convrelu layers processing input feature map
        add_num1,add_num2 = self.add_filter_num2(self.din, 512)
        self.RPN_Conv = nn.Conv2d(self.din+add_num1, 512+add_num2, 3, 1, 1, bias=True)
        if cfg.reinit_rpn == False and cfg.Only_FinetuneBN:
            state_dict = torch.load(cfg.pretrain_model)['model']
            params_name_w = 'RCNN_rpn.RPN_Conv.weight'
            params_name_b = 'RCNN_rpn.RPN_Conv.bias'
            shape = self.RPN_Conv.weight.data.shape
            #print(shape, self.RPN_Conv.weight.data.shape, state_dict[params_name_w].shape)
            add_num1,add_num2 = self.add_filter_num2(512,self.din)
            self.RPN_Conv.weight.data[:shape[0]-add_num1,:shape[1]-add_num2] = state_dict[params_name_w]
            self.RPN_Conv.bias.data[:shape[0]-add_num1] = state_dict[params_name_b]

        # define bg/fg classifcation score layer
        #self.nc_score_out = len(self.anchor_scales) * len(self.anchor_ratios) * 2 # 2(bg/fg) * 9 (anchors)
        add_num = self.add_filter_num(512)
        self.RPN_cls_score_layers = nn.ModuleList([nn.Conv2d(512+add_num, 2*nc_score_out, 1, 1, 0) for nc_score_out in cfg.ANCHOR_NUM])
        if cfg.reinit_rpn == False and cfg.Only_FinetuneBN:
            params_name_w = 'RCNN_rpn.RPN_cls_score.weight'
            params_name_b = 'RCNN_rpn.RPN_cls_score.bias'
            shape = self.RPN_cls_score_layers[1].weight.data.shape
            #print(shape, self.RPN_cls_score_layers[1].weight.data.shape, state_dict[params_name_w].shape)
            add_num1,add_num2 = self.add_filter_num2(2*nc_score_out, 512)
            self.RPN_cls_score_layers[1].weight.data[:,:shape[1]-add_num2] = state_dict[params_name_w]
            self.RPN_cls_score_layers[1].bias.data = state_dict[params_name_b]
            
        # define anchor box offset prediction layer
        #self.nc_bbox_out = len(self.anchor_scales) * len(self.anchor_ratios) * 4 # 4(coords) * 9 (anchors)
        add_num = self.add_filter_num(512)
        self.RPN_bbox_pred_layers = nn.ModuleList([nn.Conv2d(512+add_num, 4*nc_score_out, 1, 1, 0) for nc_score_out in cfg.ANCHOR_NUM])
            
        if cfg.reinit_rpn == False and cfg.Only_FinetuneBN:
            params_name_w = 'RCNN_rpn.RPN_bbox_pred.weight'
            params_name_b = 'RCNN_rpn.RPN_bbox_pred.bias'
            shape = self.RPN_bbox_pred_layers[1].weight.data.shape
            #print(shape, self.RPN_bbox_pred_layers[1].weight.data.shape, state_dict[params_name_w].shape)  
            add_num1,add_num2 = self.add_filter_num2(4*nc_score_out, 512)          
            self.RPN_bbox_pred_layers[1].weight.data[:,:shape[1]-add_num2] = state_dict[params_name_w]
            self.RPN_bbox_pred_layers[1].bias.data = state_dict[params_name_b]        
        
        # define proposal layer
        self.RPN_proposal = nn.ModuleList([_ProposalLayer(self.feat_stride, cfg.ANCHOR_SCALES_LIST[i], cfg.ANCHOR_RATIOS_LIST[i]) for i in np.arange(len(cfg.ANCHOR_NUM))])

        # define anchor target layer
        # self.RPN_anchor_target = _AnchorTargetLayer(self.feat_stride, self.anchor_scales, self.anchor_ratios,self.rpn_batchsize)

        # define module list for diffrent _AnchorTargetLayer
        self.RPN_anchor_target_layers = nn.ModuleList([_AnchorTargetLayer(self.feat_stride, cfg.ANCHOR_SCALES_LIST[i], 
                                    cfg.ANCHOR_RATIOS_LIST[i], cfg.train_rpn_batchsize_list[i]) for i in np.arange(len(cfg.ANCHOR_NUM))])

        self.rpn_loss_cls = 0
        self.rpn_loss_box = 0

    def add_filter_num(self,base):
        if cfg.add_filter_ratio >= 0.005:
            add_num = int(base*cfg.add_filter_ratio)
        else:
            add_num = cfg.add_filter_num
        return add_num

    def add_filter_num2(self,base1, base2):
        if cfg.add_filter_ratio >= 0.005:
            add_num1 = int(base1*cfg.add_filter_ratio)
            add_num2 = int(base2*cfg.add_filter_ratio)
        else:
            add_num1 = cfg.add_filter_num
            add_num2 = cfg.add_filter_num
        return add_num1,add_num2

    @staticmethod
    def reshape(x, d):
        input_shape = x.size()
        x = x.view(
            input_shape[0],
            int(d),
            int(float(input_shape[1] * input_shape[2]) / float(d)),
            input_shape[3]
        )
        return x

    def forward(self, base_feat, im_info, gt_boxes, num_boxes, cls_ind):
        if cfg.rpn_univ:
            cls_ind = 0
        else:
            cls_ind = cfg.cls_ind
        
        batch_size = base_feat.size(0)

        # return feature map after convrelu layer
        rpn_conv1 = F.relu(self.RPN_Conv(base_feat), inplace=True)
        # get rpn classification score
        rpn_cls_score = self.RPN_cls_score_layers[cls_ind](rpn_conv1)

        rpn_cls_score_reshape = self.reshape(rpn_cls_score, 2)
        rpn_cls_prob_reshape = F.softmax(rpn_cls_score_reshape)
        self.nc_score_out = cfg.ANCHOR_NUM[cls_ind]*2
        rpn_cls_prob = self.reshape(rpn_cls_prob_reshape, self.nc_score_out)

        # get rpn offsets to the anchor boxes
        rpn_bbox_pred = self.RPN_bbox_pred_layers[cls_ind](rpn_conv1)

        # proposal layer
        cfg_key = 'TRAIN' if self.training else 'TEST'

        rois = self.RPN_proposal[cls_ind]((rpn_cls_prob.data, rpn_bbox_pred.data,
                                 im_info, cfg_key))

        self.rpn_loss_cls = 0
        self.rpn_loss_box = 0

        # generating training labels and build the rpn loss
        if self.training:
            assert gt_boxes is not None

            # decide which RPN_anchor_target layers to use
            rpn_data = self.RPN_anchor_target_layers[cls_ind]((rpn_cls_score.data, gt_boxes, im_info, num_boxes))

            # compute classification loss
            rpn_cls_score = rpn_cls_score_reshape.permute(0, 2, 3, 1).contiguous().view(batch_size, -1, 2)
            rpn_label = rpn_data[0].view(batch_size, -1)

            rpn_keep = Variable(rpn_label.view(-1).ne(-1).nonzero().view(-1))
            rpn_cls_score = torch.index_select(rpn_cls_score.view(-1,2), 0, rpn_keep)
            rpn_label = torch.index_select(rpn_label.view(-1), 0, rpn_keep.data)
            rpn_label = Variable(rpn_label.long())
            self.rpn_loss_cls = F.cross_entropy(rpn_cls_score, rpn_label)
            fg_cnt = torch.sum(rpn_label.data.ne(0))

            rpn_bbox_targets, rpn_bbox_inside_weights, rpn_bbox_outside_weights = rpn_data[1:]

            # compute bbox regression loss
            rpn_bbox_inside_weights = Variable(rpn_bbox_inside_weights)
            rpn_bbox_outside_weights = Variable(rpn_bbox_outside_weights)
            rpn_bbox_targets = Variable(rpn_bbox_targets)

            self.rpn_loss_box = _smooth_l1_loss(rpn_bbox_pred, rpn_bbox_targets, rpn_bbox_inside_weights,
                                                            rpn_bbox_outside_weights, sigma=3, dim=[1,2,3])

        
        return rois, self.rpn_loss_cls, self.rpn_loss_box