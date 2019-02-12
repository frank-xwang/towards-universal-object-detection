from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from model.utils.config import cfg
from model.faster_rcnn.faster_rcnn_uni import _fasterRCNN

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
import math
import torch.utils.model_zoo as model_zoo
import pdb
import torchvision

__all__ = ['ResNet', 'resnet18', 'resnet34', 'resnet50', 'resnet101',
       'resnet152']

model_urls = {
  'resnet18': 'https://s3.amazonaws.com/pytorch/models/resnet18-5c106cde.pth',
  'resnet34': 'https://s3.amazonaws.com/pytorch/models/resnet34-333f7ec4.pth',
  'resnet50': 'https://s3.amazonaws.com/pytorch/models/resnet50-19c8e357.pth',
  'resnet101': 'https://s3.amazonaws.com/pytorch/models/resnet101-5d3b4d8f.pth',
  'resnet152': 'https://s3.amazonaws.com/pytorch/models/resnet152-b121ed2d.pth',
}

def conv3x3(in_planes, out_planes, stride=1):
  "3x3 convolution with padding"
  return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,
           padding=1, bias=False)

class BasicBlock(nn.Module):
  expansion = 1

  def __init__(self, inplanes, planes, stride=1, downsample=None, downsample_fa = None):
    super(BasicBlock, self).__init__()
    self.conv1 = conv3x3(inplanes, planes, stride)
    if cfg.finetuneBN_linear:
        self.linear1 = nn.Linear(planes, planes, bias=False)
    self.bn1 = nn.ModuleList([nn.BatchNorm2d(planes) for datasets in cfg.num_classes])
    self.relu = nn.ReLU(inplace=True)
    self.conv2 = conv3x3(planes, planes)
    if cfg.finetuneBN_linear:
        self.linear2 = nn.Linear(planes, planes, bias=False)
    self.bn2 = nn.ModuleList([nn.BatchNorm2d(planes) for datasets in cfg.num_classes])
    if cfg.finetuneBN_DS:
        self.downsample_fa = downsample_fa
    self.downsample = downsample
    self.stride = stride

  def forward(self, x):
    residual = x
    out = self.conv1(x)

    if cfg.finetuneBN_linear:
        weights = Variable(self.conv1.weight.data, requires_grad=False)
        weight_ = self.linear1(weights.permute(3,2,1,0))
        self.conv1.weight.data = weight_.permute(3,2,1,0).data.contiguous()
        out += self.conv1(x)
        self.conv1.weight.data = weights.data

    out = self.bn1[cfg.cls_ind](out)
    out = self.relu(out)

    if cfg.finetuneBN_linear:
        residual = out
        out = self.conv2(out)
        weights = Variable(self.conv2.weight.data, requires_grad=False)
        weight_ = self.linear2(weights.permute(3,2,1,0))
        self.conv2.weight.data = weight_.permute(3,2,1,0).data.contiguous()
        out += self.conv2(residual)
        self.conv2.weight.data = weights.data
    else:
        out = self.conv2(out)
    out = self.bn2[cfg.cls_ind](out)

    if self.downsample is not None:
      residual = self.downsample(x)
      if cfg.finetuneBN_DS and self.downsample_fa != None:
        residual += self.downsample_fa(x)

    out += residual
    out = self.relu(out)

    return out

class Bottleneck(nn.Module):
  expansion = 4

  def __init__(self, inplanes, planes, stride=1, downsample=None):
    super(Bottleneck, self).__init__()
    self.conv1 = nn.Conv2d(inplanes, planes, kernel_size=1, stride=stride, bias=False) # change
    self.bn1 = nn.ModuleList([nn.BatchNorm2d(planes) for datasets in cfg.num_classes])
    self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=1, # change
                 padding=1, bias=False)
    self.bn2 = nn.ModuleList([nn.BatchNorm2d(planes) for datasets in cfg.num_classes])
    self.conv3 = nn.Conv2d(planes, planes * 4, kernel_size=1, bias=False)
    self.bn3 = nn.ModuleList([nn.BatchNorm2d(planes * 4) for datasets in cfg.num_classes])
    self.relu = nn.ReLU(inplace=True)
    self.downsample = downsample
    self.stride = stride
    self.inplanes = inplanes
    self.planes = planes

  def forward(self, x):
    residual = x

    out = self.conv1(x)
    out = self.bn1[cfg.cls_ind](out)
    out = self.relu(out)

    out = self.conv2(out)
    out = self.bn2[cfg.cls_ind](out)
    out = self.relu(out)

    out = self.conv3(out)
    out = self.bn3[cfg.cls_ind](out)

    if self.downsample is not None:
      residual = self.downsample(x)

    out += residual
    out = self.relu(out)

    return out

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
        
class ResNet(nn.Module):
  def __init__(self, block, layers, num_classes=1000):
    self.inplanes = 64
    super(ResNet, self).__init__()
    self.conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3,
                 bias=False)                                        # rcnn.base.0
    self.bn1 = nn.BatchNorm2d(64)                                   # rcnn.base.1
    self.relu = nn.ReLU(inplace=True)                               # 2
    self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=0, ceil_mode=True) # 3
    self.layer1 = self._make_layer(block, 64, layers[0])            # rcnn.base.4
    self.layer2 = self._make_layer(block, 128, layers[1], stride=2) # rcnn.base.5
    self.layer3 = self._make_layer(block, 256, layers[2], stride=2) # rcnn.base.6
    self.layer4 = self._make_layer(block, 512, layers[3], stride=2) # rcnn.top
    # it is slightly better whereas slower to set stride = 1
    # self.layer4 = self._make_layer(block, 512, layers[3], stride=1)
    self.avgpool = nn.AvgPool2d(7)
    # block.expansion is 1
    #self.fc = nn.ModuleList([nn.Linear(512 * block.expansion, num_classes) for num_class in cfg.num_classes])
    self.fc = nn.Linear(512 * block.expansion, num_classes)

    for m in self.modules():
      if isinstance(m, nn.Conv2d):
        n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
        m.weight.data.normal_(0, math.sqrt(2. / n))
      elif isinstance(m, nn.BatchNorm2d):
        m.weight.data.fill_(1)
        m.bias.data.zero_()

  def _make_layer(self, block, planes, blocks, stride=1):
    downsample = None
    downsample_fa = None
    if stride != 1 or self.inplanes != planes * block.expansion:
      downsample = nn.Sequential(
        nn.Conv2d(self.inplanes, planes * block.expansion,
              kernel_size=1, stride=stride, bias=False),
        BnMux(planes * block.expansion),
        #nn.BatchNorm2d(planes * block.expansion),
      )
      if cfg.fa_conv_num == 1:
        downsample_fa = nn.Sequential(
            nn.Conv2d(self.inplanes, planes * block.expansion,
                kernel_size=1, stride=stride, bias=False),
                #BnMux(planes * block.expansion),
                nn.BatchNorm2d(planes * block.expansion),
            )
      if cfg.fa_conv_num == 2:
        downsample_fa = nn.Sequential(
            nn.Conv2d(self.inplanes, self.inplanes,
                kernel_size=1, stride=1, bias=False),
            nn.BatchNorm2d(self.inplanes),
            nn.Conv2d(self.inplanes, planes * block.expansion,
                kernel_size=1, stride=stride, bias=False),
                #BnMux(planes * block.expansion),
            nn.BatchNorm2d(planes * block.expansion),
            )

    if cfg.finetuneBN_DS:
      layers = []
      layers.append(block(self.inplanes, planes, stride, downsample, downsample_fa))
    else:
      layers = []
      layers.append(block(self.inplanes, planes, stride, downsample))  

    self.inplanes = planes * block.expansion
    for i in range(1, blocks):
      layers.append(block(self.inplanes, planes))

    return nn.Sequential(*layers)

  def forward(self, x):
    x = self.conv1(x)
    x = self.bn1(x)
    x = self.relu(x)
    x = self.maxpool(x)

    x = self.layer1(x)
    x = self.layer2(x)
    x = self.layer3(x)
    x = self.layer4(x)

    x = self.avgpool(x)
    x = x.view(x.size(0), -1)
    #x = self.fc[cfg.cls_ind](x)
    x = self.fc(x)

    return x

def resnet18(pretrained=False):
  """Constructs a ResNet-18 model.
  Args:
    pretrained (bool): If True, returns a model pre-trained on ImageNet
  """
  model = ResNet(BasicBlock, [2, 2, 2, 2])
  if pretrained:
    model.load_state_dict(model_zoo.load_url(model_urls['resnet18']))
  return model

def resnet34(pretrained=False):
  """Constructs a ResNet-34 model.
  Args:
    pretrained (bool): If True, returns a model pre-trained on ImageNet
  """
  model = ResNet(BasicBlock, [3, 4, 6, 3])
  if pretrained:
    model.load_state_dict(model_zoo.load_url(model_urls['resnet34']))
  return model

def resnet50(pretrained=False):
  """Constructs a ResNet-50 model.
  Args:
    pretrained (bool): If True, returns a model pre-trained on ImageNet
  """
  model = ResNet(Bottleneck, [3, 4, 6, 3])
  if pretrained:
    model.load_state_dict(model_zoo.load_url(model_urls['resnet50']))
  return model

def resnet101(pretrained=False):
  """Constructs a ResNet-101 model.
  Args:
    pretrained (bool): If True, returns a model pre-trained on ImageNet
  """
  model = ResNet(Bottleneck, [3, 4, 23, 3])
  if pretrained:
    model.load_state_dict(model_zoo.load_url(model_urls['resnet101']))
  return model

def resnet152(pretrained=False):
  """Constructs a ResNet-152 model.
  Args:
    pretrained (bool): If True, returns a model pre-trained on ImageNet
  """
  model = ResNet(Bottleneck, [3, 8, 36, 3])
  if pretrained:
    model.load_state_dict(model_zoo.load_url(model_urls['resnet152']))
  return model

class resnet_bn(_fasterRCNN):
  def __init__(self, classes, num_layers=101, pretrained=False, class_agnostic=False, rpn_batchsize_list=None):
    self.model_path = 'data/pretrained_model/resnet' + str(num_layers) + '_caffe.pth'
    if num_layers == 18:
      self.dout_base_model = 256
    else:
      self.dout_base_model = 1024

    self.pretrained = pretrained
    self.class_agnostic = class_agnostic
    self.rpn_batchsize_list = rpn_batchsize_list
    self.num_layers = num_layers

    _fasterRCNN.__init__(self, classes, class_agnostic,rpn_batchsize_list)

  def _init_modules(self):
    resnet = eval('resnet' + str(self.num_layers))()

    if self.pretrained == True:
      if cfg.Only_FinetuneBN:
        self.model_path = cfg.pretrain_model
        state_dict = torch.load(self.model_path)['model']
      else:
        state_dict = torch.load(self.model_path)     
      print("Loading pretrained weights from %s" %(self.model_path))
      
      # load weight and bias of bn params for each datasets
      for k, v in state_dict.items():
        print(k)
      for k in resnet.state_dict():
        print(k)
      new_state_dict = {}
      pre_head_names_1 = ['RCNN_base', 'RCNN_top']
      pre_head_names_2 = ['4.0', '4.1', '5.0', '5.1', '6.0', '6.1', '0.0', '0.1']
      cur_head_names = ['layer1.0', 'layer1.1', 'layer2.0', 'layer2.1', 'layer3.0', 'layer3.1', 'layer4.0', 'layer4.1']

      # layer4.0.downsample.1.bn.0.weight
      for k, v in state_dict.items():
        k_list = k.split('.')
        #print(k)
        if len(k_list) == 3:
            if 'RCNN_base' in k:
                if '.0.' in k:
                    current = 'conv1.' + k_list[-1]
                if '.1.' in k:
                    current = 'bn1.' + k_list[-1]
                if current in resnet.state_dict():
                    new_state_dict[current] = v              
        elif len(k_list) > 3:
            # Get head name of current and pretrained model 
            pre_head_name_1 = k_list[0]
            pre_head_name_2 = k_list[1] + '.' + k_list[2]
            cur_head_name   = cur_head_names[pre_head_names_2.index(pre_head_name_2)]

            if pre_head_name_1 == 'RCNN_base' or pre_head_name_1 == 'RCNN_top':
                if 'downsample' in k:
                    # assign convolution layer params into new model
                    current = cur_head_name + '.' + k_list[-3] + '.' +  k_list[-2] + '.' + k_list[-1]
                    if current in resnet.state_dict():
                        new_state_dict[current] = v

                    if cfg.fa_conv_num == 1:
                        current_fa_0 = cur_head_name + '.' + 'downsample_fa' + '.' +  k_list[-2] + '.' + k_list[-1]
                        if cfg.finetuneBN_DS and current_fa_0 in resnet.state_dict():
                            new_state_dict[current_fa_0] = v            

                    if cfg.fa_conv_num == 2:
                        current_fa_0 = cur_head_name + '.' + 'downsample_fa' + '.' +  k_list[-2] + '.' + k_list[-1]
                        current_fa_2 = cur_head_name + '.' + 'downsample_fa' + '.' +  str(2) + '.' + k_list[-1]
                        current_fa_3 = cur_head_name + '.' + 'downsample_fa' + '.' +  str(3) + '.' + k_list[-1]
                        #print(torch.mean(resnet.state_dict()[current_fa_0]),torch.std(resnet.state_dict()[current_fa_0]))
                        if cfg.finetuneBN_DS and current_fa_0 in resnet.state_dict():
                            new_state_dict[current_fa_0] = resnet.state_dict()[current_fa_0]
                        if cfg.finetuneBN_DS and current_fa_2 in resnet.state_dict():
                            if 'downsample.0.' in k: # if it is .0. in k, weight belong to conv layer
                                new_state_dict[current_fa_2] = v
                        if cfg.finetuneBN_DS and current_fa_3 in resnet.state_dict():
                            if 'downsample.1.' in k: # if it is .1. in k, weight belong to bn layer
                                new_state_dict[current_fa_3] = v
                        
                    # assign batch normalization layer of downsampling part params into new model
                    for n in range(len(cfg.num_classes)):
                        current    = cur_head_name + '.' + k_list[-3] + '.' +  k_list[-2] + '.bn.' + str(n) + '.' + k_list[-1]
                        if current in resnet.state_dict():
                            new_state_dict[current] = v
                    
                elif 'bn' in k:
                    # assign batch normalization layer params into new model
                    for n in range(len(cfg.num_classes)):
                        current = cur_head_name + '.' + k_list[3] + '.' + str(n) + '.' + k_list[-1]
                        if current in resnet.state_dict():
                            new_state_dict[current] = v
                elif 'conv' in k:
                    # assign convolution layer params into new model
                    current = cur_head_name + '.' + k_list[-2] + '.' + k_list[-1] 
                    if current in resnet.state_dict():
                        new_state_dict[current] = v

      for k in resnet.state_dict():
          if 'linear' in k:
              diag = torch.ones(resnet.state_dict()[k].size(0))
              diag = torch.diag(diag)
              new_state_dict[k] = diag

      # initial fc layer, we will not use it. just for passing load_state_dict
      new_state_dict['fc.weight'] = resnet.state_dict()['fc.weight'].fill_(0)
      new_state_dict['fc.bias'] =  resnet.state_dict()['fc.bias'].fill_(0)

      resnet.load_state_dict(new_state_dict)

    # Build resnet.
    # RCNN_base[0]: resnet.conv1, RCNN_base[1]: resnet.bn1 ......
    self.RCNN_base = nn.Sequential(resnet.conv1, resnet.bn1,resnet.relu,
      resnet.maxpool,resnet.layer1,resnet.layer2,resnet.layer3)

    self.RCNN_top = nn.Sequential(resnet.layer4)
    if self.num_layers == 18:
      self.RCNN_cls_score_layers = nn.ModuleList([nn.Linear(512, n_classes) for n_classes in cfg.num_classes])
    else:
      self.RCNN_cls_score_layers = nn.ModuleList([nn.Linear(2048, n_classes) for n_classes in cfg.num_classes])

    if self.pretrained and cfg.Only_FinetuneBN:
      params_name_w = 'RCNN_cls_score.weight'
      params_name_b = 'RCNN_cls_score.bias'
      self.RCNN_cls_score_layers[1].weight.data = state_dict[params_name_w]
      self.RCNN_cls_score_layers[1].bias.data = state_dict[params_name_b]

    if self.class_agnostic:
      if self.num_layers == 18:
        self.RCNN_bbox_pred_layers = nn.ModuleList([nn.Linear(512,  4) for n_classes in cfg.num_classes])
      else:
        self.RCNN_bbox_pred_layers = nn.ModuleList([nn.Linear(2048, 4) for n_classes in cfg.num_classes])
    else:
      if self.num_layers == 18:
        self.RCNN_bbox_pred_layers = nn.ModuleList([nn.Linear(512,  4 * n_classes) for n_classes in cfg.num_classes])
      else:
        self.RCNN_bbox_pred_layers = nn.ModuleList([nn.Linear(2048, 4 * n_classes) for n_classes in cfg.num_classes])

    if self.pretrained and cfg.Only_FinetuneBN:
      #print(self.RCNN_bbox_pred_layers.state_dict())
      params_name_w = 'RCNN_bbox_pred.weight'
      params_name_b = 'RCNN_bbox_pred.bias'
      self.RCNN_bbox_pred_layers[1].weight.data = state_dict[params_name_w]
      self.RCNN_bbox_pred_layers[1].bias.data = state_dict[params_name_b]        

    # Fix blocks
    for p in self.RCNN_base[0].parameters(): p.requires_grad=False
    for p in self.RCNN_base[1].parameters(): p.requires_grad=False
    
    assert (0 <= cfg.RESNET.FIXED_BLOCKS < 4)
    if cfg.RESNET.FIXED_BLOCKS >= 3:
      for p in self.RCNN_base[6].parameters(): p.requires_grad=False
    if cfg.RESNET.FIXED_BLOCKS >= 2:
      for p in self.RCNN_base[5].parameters(): p.requires_grad=False
    if cfg.RESNET.FIXED_BLOCKS >= 1:
      for p in self.RCNN_base[4].parameters(): p.requires_grad=False

    def set_bn_fix(m):
      classname = m.__class__.__name__
      if classname.find('BatchNorm') != -1:
        for p in m.parameters(): p.requires_grad=False
          
    if cfg.fix_bn:
      self.RCNN_base.apply(set_bn_fix)
      self.RCNN_top.apply(set_bn_fix)

    if cfg.Only_FinetuneBN:
        # Fix the all layers, excapt batch norm layers
        for name, params in self.RCNN_base.named_parameters():
          if 'bn' in name:
            params.requires_grad = True
            print('Require_grad', name, params.requires_grad)
          elif 'linear' in name:
            params.requires_grad = True
            print('Require_grad', name, params.requires_grad)
          else:
            if cfg.finetuneBN_DS and 'downsample_fa' in name:
                params.requires_grad = True
                print('Require_grad', name)                
            else:
                #if '0.1.conv' in name or '6.1.conv' in name:
                #if '0.1.conv' in name or '6.1.conv' in name:
                #if 'downsample' in name:
                if '.conv2' in name or '.conv1' in name:
                  params.requires_grad = True
                  print('Require_grad', name, params.requires_grad)
                else:
                  params.requires_grad = True
                  print('Require_grad', name, params.requires_grad)
        for name, params in self.RCNN_top.named_parameters():
          if 'bn' in name:
            params.requires_grad = True
            print('Require_grad', name,params.requires_grad)
          elif 'linear' in name:
            params.requires_grad = True
            print('Require_grad', name)
          else:
            if cfg.finetuneBN_DS and 'downsample_fa' in name:
                params.requires_grad = True
                print('Require_grad', name, params.requires_grad)
            else:
                #if '0.1.conv' in name or '0.0.conv' in name or '0.0.downsample' in name:
                #if 'downsample' in name:
                if '.conv2' in name or '.conv1' in name:
                  params.requires_grad = True
                  print('Require_grad', name,params.requires_grad)
                else:
                  params.requires_grad = True  
                  print('Require_grad', name,params.requires_grad)               

  def train(self, mode=True):
    # Override train so that the training mode is set as we want
    nn.Module.train(self, mode)
    if mode:
      print('traing in process')
      # Set fixed blocks to be in eval mode
      self.RCNN_base.eval()
      self.RCNN_base[5].train()
      self.RCNN_base[6].train()

      def set_bn_eval(m):
        classname = m.__class__.__name__
        if classname.find('BatchNorm') != -1:
          m.eval()

      if cfg.fix_bn:
        self.RCNN_base.apply(set_bn_eval)
        self.RCNN_top.apply(set_bn_eval)

  def _head_to_tail(self, pool5):
    fc7 = self.RCNN_top(pool5).mean(3).mean(2)
    return fc7