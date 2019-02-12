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
from model.faster_rcnn.faster_rcnn import _fasterRCNN
import pdb
from model.utils.config import cfg
from .vgg_bn import VGG

class vgg16(_fasterRCNN):
  def __init__(self, classes, pretrained=False, class_agnostic=False, batch_norm=False, rpn_batchsize=None):
    if batch_norm:
      self.model_path = 'data/pretrained_model/vgg16_bn.pth'
    else:
      self.model_path = 'data/pretrained_model/vgg16_caffe.pth'
    self.dout_base_model = 512
    self.pretrained = pretrained
    self.class_agnostic = class_agnostic
    self.rpn_batchsize = rpn_batchsize
    self.batch_norm = batch_norm

    _fasterRCNN.__init__(self, classes, class_agnostic, rpn_batchsize)

  def _init_modules(self):
    if self.batch_norm == False:
      vgg = models.vgg16()
      if self.pretrained:
          print("Loading pretrained weights from %s" %(self.model_path))
          state_dict = torch.load(self.model_path)
          vgg.load_state_dict({k:v for k,v in state_dict.items() if k in vgg.state_dict()})
      #print(state_dict['model'])

      if cfg.VGG_ORIGIN:
        print('INFO: Using original VGG16 classifier')
        vgg.classifier = nn.Sequential(*list(vgg.classifier._modules.values())[:-1])
      else:
        print('INFO: Using new classifier layers')
        vgg.classifier = nn.Sequential(
                          nn.Linear(512 * 4 * 8, 2048),
                          nn.ReLU(),
                          nn.Dropout(0.5),
                        )

      # not using the last maxpool layer
      self.RCNN_base = nn.Sequential(*list(vgg.features._modules.values())[:-1])

      # Fix the layers before conv3:
      for layer in range(10):
        for p in self.RCNN_base[layer].parameters(): p.requires_grad = False

      # self.RCNN_base = _RCNN_base(vgg.features, self.classes, self.dout_base_model)

      self.RCNN_top = vgg.classifier

      # change the original classifier into new classifier
      # not using the last maxpool layer
      if cfg.VGG_ORIGIN: 
        self.RCNN_cls_score = nn.Linear(4096, self.n_classes)
      else:
        self.RCNN_cls_score = nn.Linear(2048, self.n_classes)    
      if cfg.VGG_ORIGIN:
        if self.class_agnostic:
          self.RCNN_bbox_pred = nn.Linear(4096, 4)
        else:
          self.RCNN_bbox_pred = nn.Linear(4096, 4 * self.n_classes)  
      else:
        if self.class_agnostic:
          self.RCNN_bbox_pred = nn.Linear(2048, 4)
        else:
          self.RCNN_bbox_pred = nn.Linear(2048, 4 * self.n_classes)   

    ################################################################
    # VGG_16 with batch normalization
    else:
      vgg = vgg16_bn()
      if self.pretrained:
          print("Loading pretrained weights from %s" %(self.model_path))
          state_dict = torch.load(self.model_path)
          vgg.load_state_dict({k:v for k,v in state_dict.items() if k in vgg.state_dict()})
      #print(state_dict['model'])
      for i in vgg.state_dict():
        print(i)

      if cfg.VGG_ORIGIN:
        print('INFO: Using original VGG16 classifier')
        # self.classifier = nn.Sequential(
        #     nn.Linear(512 * 7 * 7, 4096),
        #     nn.ReLU(True),
        #     nn.Dropout(),
        #     nn.Linear(4096, 4096),
        #     nn.ReLU(True),
        #     nn.Dropout(),
        #     nn.Linear(4096, num_classes),
        # )
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
      #self.RCNN_base = nn.Sequential(*list(vgg.features._modules.values())[:-1])
      self.RCNN_base = nn.ModuleList(list(vgg.features._modules.values())[:-1])

      # Fix the layers before conv3:
      for layer in range(13):
        for p in self.RCNN_base[layer].parameters(): p.requires_grad = False

      # self.RCNN_base = _RCNN_base(vgg.features, self.classes, self.dout_base_model)
      self.RCNN_top = vgg.classifier
      # change the original classifier into new classifier
      # not using the last maxpool layer
      if cfg.VGG_ORIGIN: 
        self.RCNN_cls_score = nn.Linear(4096, self.n_classes)
      else:
        self.RCNN_cls_score = nn.Linear(2048, self.n_classes)    
      if cfg.VGG_ORIGIN:
        if self.class_agnostic:
          self.RCNN_bbox_pred = nn.Linear(4096, 4)
        else:
          self.RCNN_bbox_pred = nn.Linear(4096, 4 * self.n_classes)  
      else:
        if self.class_agnostic:
          self.RCNN_bbox_pred = nn.Linear(2048, 4)
        else:
          self.RCNN_bbox_pred = nn.Linear(2048, 4 * self.n_classes)                   

  def _head_to_tail(self, pool5):
    
    pool5_flat = pool5.view(pool5.size(0), -1)
    fc7 = self.RCNN_top(pool5_flat)

    return fc7

def make_layers(CFG, batch_norm=False):
    layers = []
    in_channels = 3
    for v in CFG:
        if v == 'M':
            layers += [nn.MaxPool2d(kernel_size=2, stride=2)]
        else:
            conv2d = nn.Conv2d(in_channels, v, kernel_size=3, padding=1)
            if batch_norm:
                layers += [conv2d, nn.BatchNorm2d(v), nn.ReLU(inplace=True)]
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
