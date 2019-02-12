# --------------------------------------------------------
# Tensorflow Faster R-CNN
# Licensed under The MIT License [see LICENSE for details]
# Written by Xinlei Chen
# --------------------------------------------------------
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
import math
import torchvision.models as models
from model.faster_rcnn.faster_rcnn_uni import _fasterRCNN
import pdb
from model.utils.config import cfg
from .vgg_bn import VGG

class VGG16_bn_finetune(_fasterRCNN):
    def __init__(self, classes=None, pretrained=False, class_agnostic=False, batch_norm=False, rpn_batchsize_list=None):
        if batch_norm:
            self.model_path = 'data/pretrained_model/vgg16_bn.pth'
        else:
            self.model_path = 'data/pretrained_model/vgg16_caffe.pth'
        self.dout_base_model = 512
        self.pretrained = pretrained
        self.class_agnostic = class_agnostic
        self.rpn_batchsize_list = rpn_batchsize_list
        self.batch_norm = batch_norm

        _fasterRCNN.__init__(self, classes, class_agnostic, rpn_batchsize_list)

    def _init_modules(self):
        vgg = vgg16_bn()
        if self.pretrained:
            print("Loading pretrained weights from %s" %(cfg.pretrain_model))
            state_dict = torch.load(cfg.pretrain_model)['model']
            # for i in vgg.state_dict():
            #     print(i)
            layers_list = []
            idx = 0
            new_state_dict = {}

            for k, v in state_dict.items():
                k_list = k.split('.')
                #print(k)
                if 'RCNN_base' in k:
                    if 'running_var' in k:
                        for bottom in ['weight','bias','running_mean','running_var']:
                            for n in range(len(cfg.num_classes)):
                                current = 'features.' + k_list[1] + '.bn' + '.' + str(n) + '.' + bottom
                                if current in vgg.state_dict():
                                    pre_name = k_list[0] + '.' + k_list[1] + '.' + bottom
                                    new_state_dict[current] = state_dict[pre_name]
                    else:
                        current = 'features.' + str(k_list[1]) + '.' + k_list[2]
                        if current in vgg.state_dict():
                            new_state_dict[current] = v
                elif 'RCNN_top' in k:
                    current = 'classifier.' + str(k_list[1]) + '.' + k_list[2]
                    if current in vgg.state_dict():
                        new_state_dict[current] = v
            for current in ['classifier.6.bias', 'classifier.6.weight']:
                new_state_dict[current] = vgg.state_dict()[current].fill_(0)
            vgg.load_state_dict(new_state_dict)

        if cfg.VGG_ORIGIN:
            print('INFO: Using original VGG16 classifier')
            # Do not use the last layer
            vgg.classifier = nn.Sequential(*list(vgg.classifier._modules.values())[:-1])
        else:
            print('INFO: Using new classifier layers')
            vgg.classifier = nn.Sequential(
                            nn.Linear(512 *4 *8, 2048),
                            nn.ReLU(),
                            nn.Dropout(0.5),
                            )

        # not using the last maxpool layer
        self.RCNN_base = nn.Sequential(*list(vgg.features._modules.values())[:-1])

        # Fix the all layers, excapt batch norm layers
        for name, params in self.RCNN_base.named_parameters():
            if 'bn' in name:
                if '.1.' in name:
                    #print('False', name)
                    params.requires_grad = False
                else:
                    #print('True', name)
                    continue
            else:
                #print('false', name)
                params.requires_grad = False

        # self.RCNN_base = _RCNN_base(vgg.features, self.classes, self.dout_base_model)
        self.RCNN_top = vgg.classifier

        # change the original classifier into new classifier
        # not using the last maxpool layer
        if cfg.VGG_ORIGIN:
            self.RCNN_cls_score_layers = nn.ModuleList([nn.Linear(4096, n_classes) for n_classes in cfg.num_classes])
            #print(self.RCNN_cls_score_layers.state_dict())
            if self.pretrained:
                params_name_w = 'RCNN_cls_score.weight'
                params_name_b = 'RCNN_cls_score.bias'
                self.RCNN_cls_score_layers[1].weight.data = state_dict[params_name_w]
                self.RCNN_cls_score_layers[1].bias.data = state_dict[params_name_b]
        else:
            self.RCNN_cls_score_layers = nn.ModuleList([nn.Linear(2048, n_classes) for n_classes in cfg.num_classes])

        if cfg.VGG_ORIGIN:
            if self.class_agnostic:
                self.RCNN_bbox_pred_layers = nn.ModuleList([nn.Linear(4096, 4) for n_classes in cfg.num_classes])        
            else:
                self.RCNN_bbox_pred_layers = nn.ModuleList([nn.Linear(4096, 4 * n_classes) for n_classes in cfg.num_classes])
                if self.pretrained:
                    #print(self.RCNN_bbox_pred_layers.state_dict())
                    params_name_w = 'RCNN_bbox_pred.weight'
                    params_name_b = 'RCNN_bbox_pred.bias'
                    self.RCNN_bbox_pred_layers[1].weight.data = state_dict[params_name_w]
                    self.RCNN_bbox_pred_layers[1].bias.data = state_dict[params_name_b]
        else:
            if self.class_agnostic:
                self.RCNN_bbox_pred_layers = nn.ModuleList([nn.Linear(2048, 4) for n_classes in cfg.num_classes])
            else:
                self.RCNN_bbox_pred_layers = nn.ModuleList([nn.Linear(2048, 4 * n_classes) for n_classes in cfg.num_classes])

    def _head_to_tail(self, pool5):
        pool5_flat = pool5.view(pool5.size(0), -1)
        fc7 = self.RCNN_top(pool5_flat)

        return fc7

class BnMux(nn.Module):
    def __init__(self, planes):
        super(BnMux, self).__init__()
        self.bn = nn.ModuleList(
            [nn.BatchNorm2d(planes) for n_classes in cfg.num_classes]
        )

    def forward(self, x):
        #print('bn_mux is forwarding', cfg.cls_ind)
        out = self.bn[cfg.cls_ind](x)
        # if cfg.nums == 1:
        #     print(('layer.weight',self.bn[cfg.cls_ind].weight))
        return out

def make_layers(CFG, batch_norm=False):
    layers = []
    in_channels = 3
    for v in CFG:
        if v == 'M':
            layers += [nn.MaxPool2d(kernel_size=2, stride=2)]
        else:
            conv2d = nn.Conv2d(in_channels, v, kernel_size=3, padding=1)
            if batch_norm:
                bn = BnMux(v)
                layers += [conv2d, bn, nn.ReLU(inplace=True)]
            else:
                layers += [conv2d, nn.ReLU(inplace=True)]
            in_channels = v
    return nn.ModuleList(layers)

CFG = {
    'A': [64, 'M', 128, 'M', 256, 256, 'M', 512, 512, 'M', 512, 512, 'M'],
    'B': [64, 64, 'M', 128, 128, 'M', 256, 256, 'M', 512, 512, 'M', 512, 512, 'M'],
    'D': [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 'M', 512, 512, 512, 'M', 512, 512, 512, 'M'],
    'E': [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 256, 'M', 512, 512, 512, 512, 'M', 512, 512, 512, 512, 'M'],
}
model_urls = {
    'vgg11': 'https://download.pytorch.org/models/vgg11-bbd30ac9.pth',
    'vgg13': 'https://download.pytorch.org/models/vgg13-c768596a.pth',
    'vgg16': 'https://download.pytorch.org/models/vgg16-397923af.pth',
    'vgg19': 'https://download.pytorch.org/models/vgg19-dcbb9e9d.pth',
    'vgg11_bn': 'https://download.pytorch.org/models/vgg11_bn-6002323d.pth',
    'vgg13_bn': 'https://download.pytorch.org/models/vgg13_bn-abd245e5.pth',
    'vgg16_bn': 'https://download.pytorch.org/models/vgg16_bn-6c64b313.pth',
    'vgg19_bn': 'https://download.pytorch.org/models/vgg19_bn-c79401a0.pth',
}

def vgg16_bn(pretrained=False, **kwargs):
    """VGG 16-layer model (configuration "D") with batch normalization
    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    if pretrained:
        kwargs['init_weights'] = False
    model = VGG(make_layers(CFG['D'], batch_norm=True), **kwargs)
    if pretrained:
        model.load_state_dict(model_zoo.load_url(model_urls['vgg16_bn']))
    return model