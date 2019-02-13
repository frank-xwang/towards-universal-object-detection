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
from model.faster_rcnn.dataset_attention_module import DatasetsAttention

__all__ = ['DataAttentionResNet', 'data_att_resnet18', 'data_att_resnet34', 'data_att_resnet50', 'data_att_resnet101',
       'data_att_resnet152']

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

class SEBasicBlock(nn.Module):
  expansion = 1

  def __init__(self, inplanes, planes, stride=1, downsample=None, se_loss=False, fixed_block=False):
    super(SEBasicBlock, self).__init__()
    self.se_loss = se_loss
    self.fixed_block = fixed_block
    self.conv1 = conv3x3(inplanes, planes, stride)
    if cfg.use_mux: self.bn1 = nn.ModuleList([nn.BatchNorm2d(planes) for datasets in cfg.num_classes])
    else: self.bn1 = nn.BatchNorm2d(planes)
    self.relu = nn.ReLU(inplace=True)
    self.conv2 = conv3x3(planes, planes)
    if cfg.use_mux: self.bn2 = nn.ModuleList([nn.BatchNorm2d(planes) for datasets in cfg.num_classes])
    else: self.bn2 = nn.BatchNorm2d(planes)
    self.datasets_attention = DatasetsAttention(planes, reduction=16, se_loss=se_loss, nclass_list=cfg.num_classes,\
                              fixed_block=fixed_block, domain_pred=cfg.domain_pred)
    self.downsample = downsample
    self.stride = stride

  def forward(self, x):
    residual = x

    out = self.conv1(x)
    if cfg.use_mux: out = self.bn1[cfg.cls_ind](out)
    else: out = self.bn1(out)
    out = self.relu(out)

    out = self.conv2(out)
    if cfg.use_mux: out = self.bn2[cfg.cls_ind](out)
    else: out = self.bn2(out)

    if cfg.domain_pred and not self.fixed_block:
      out, domain_pred = self.datasets_attention(out)
    else:
      out = self.datasets_attention(out)

    if self.downsample is not None:
      residual = self.downsample(x)

    out += residual
    out = self.relu(out)
    if cfg.domain_pred and not self.fixed_block:
      return out, domain_pred
    return out

class SEBottleneck(nn.Module):
  expansion = 4

  def __init__(self, inplanes, planes, stride=1, downsample=None, se_loss=False, fixed_block=False):
    super(SEBottleneck, self).__init__()
    self.se_loss = se_loss
    self.fixed_block = fixed_block
    self.conv1 = nn.Conv2d(inplanes, planes, kernel_size=1, bias=False)
    if cfg.use_mux: self.bn1 = nn.ModuleList([nn.BatchNorm2d(planes) for datasets in cfg.num_classes])
    else: self.bn1 = nn.BatchNorm2d(planes)
    self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=stride, # change
                 padding=1, bias=False)
    if cfg.use_mux: self.bn2 = nn.ModuleList([nn.BatchNorm2d(planes) for datasets in cfg.num_classes])
    else: self.bn2 = nn.BatchNorm2d(planes)
    self.conv3 = nn.Conv2d(planes, planes * 4, kernel_size=1, bias=False)
    if cfg.use_mux: self.bn3 = nn.ModuleList([nn.BatchNorm2d(planes * 4) for datasets in cfg.num_classes])
    else: self.bn3 = nn.BatchNorm2d(planes * 4)
    self.datasets_attention = DatasetsAttention(planes * 4, reduction=16, se_loss=se_loss,\
                              nclass_list=cfg.num_classes, fixed_block=fixed_block, domain_pred=cfg.domain_pred)
    self.relu = nn.ReLU(inplace=True)
    self.downsample = downsample
    self.stride = stride

  def forward(self, x):
    residual = x

    out = self.conv1(x)
    if cfg.use_mux: out = self.bn1[cfg.cls_ind](out)
    else: out = self.bn1(out)
    out = self.relu(out)

    out = self.conv2(out)
    if cfg.use_mux: out = self.bn2[cfg.cls_ind](out)
    else: out = self.bn2(out)
    out = self.relu(out)

    out = self.conv3(out)
    if cfg.use_mux: out = self.bn3[cfg.cls_ind](out)
    else: out = self.bn3(out)

    if cfg.domain_pred and not self.fixed_block:
      out, domain_pred = self.datasets_attention(out)
    else:
      out = self.datasets_attention(out)

    if self.downsample is not None:
      residual = self.downsample(x)

    out += residual
    out = self.relu(out)
    if cfg.domain_pred and not self.fixed_block:
      if cfg.domain_preds is None:
        cfg.domain_preds = domain_pred.cpu()
      else:
        domain_pred = domain_pred.view(cfg.domain_preds.size(0), -1, domain_pred.size(1), domain_pred.size(2))
        cfg.domain_preds += torch.mean(domain_pred,1).cpu()
    return out

class BnMux(nn.Module):
    def __init__(self, planes):
        super(BnMux, self).__init__()
        self.bn = nn.ModuleList(
            [nn.BatchNorm2d(planes) for n_classes in cfg.num_classes]
        )

    def forward(self, x):
        out = self.bn[cfg.cls_ind](x)
        return out
        
class DataAttentionResNet(nn.Module):
  def __init__(self, block, layers, num_classes=1000, se_loss=False):
    self.inplanes = 64
    super(DataAttentionResNet, self).__init__()
    self.se_loss = se_loss
    self.conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3,
                 bias=False)                                        # rcnn.base.0
    self.bn1 = nn.BatchNorm2d(64)                                   # rcnn.base.1
    self.relu = nn.ReLU(inplace=True)                               # 2
    self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=0, ceil_mode=True)  # 3
    self.layer1 = self._make_layer(block, 64, layers[0], se_loss=False, fixed_block=True)  # rcnn.base.4
    self.layer2 = self._make_layer(block, 128, layers[1], stride=2, se_loss=se_loss)       # rcnn.base.5
    self.layer3 = self._make_layer(block, 256, layers[2], stride=2, se_loss=se_loss)       # rcnn.base.6
    self.layer4 = self._make_layer(block, 512, layers[3], stride=2, se_loss=False)         # rcnn.top
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

  def _make_layer(self, block, planes, blocks, stride=1, se_loss=False, fixed_block=False):
    downsample = None
    if stride != 1 or self.inplanes != planes * block.expansion:
      if cfg.use_mux:
        downsample = nn.Sequential(
          nn.Conv2d(self.inplanes, planes * block.expansion,
                kernel_size=1, stride=stride, bias=False),
          BnMux(planes * block.expansion),
        )
      else:
        downsample = nn.Sequential(
          nn.Conv2d(self.inplanes, planes * block.expansion,
                kernel_size=1, stride=stride, bias=False),
          nn.BatchNorm2d(planes * block.expansion),
        )

    layers = []
    layers.append(block(self.inplanes, planes, stride, downsample, fixed_block=fixed_block))
    self.inplanes = planes * block.expansion
    for i in range(1, blocks):
      if i == blocks - 1:
        layers.append(block(self.inplanes, planes, se_loss=se_loss, fixed_block=fixed_block))
      else:
        layers.append(block(self.inplanes, planes, se_loss=False, fixed_block=fixed_block))
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
    x = self.fc(x)
    return x

def data_att_resnet18(pretrained=False, se_loss=False):
  """Constructs a ResNet-18 model.
  Args:
    pretrained (bool): If True, returns a model pre-trained on ImageNet
  """
  model = DataAttentionResNet(SEBasicBlock, [2, 2, 2, 2], se_loss=se_loss)
  if pretrained:
    model.load_state_dict(model_zoo.load_url(model_urls['resnet18']))
  return model

def data_att_resnet34(pretrained=False, se_loss=False):
  """Constructs a ResNet-34 model.
  Args:
    pretrained (bool): If True, returns a model pre-trained on ImageNet
  """
  model = DataAttentionResNet(SEBasicBlock, [3, 4, 6, 3], se_loss=se_loss)
  if pretrained:
    model.load_state_dict(model_zoo.load_url(model_urls['resnet34']))
  return model

def data_att_resnet50(pretrained=False, se_loss=False):
  """Constructs a ResNet-50 model.
  Args:
    pretrained (bool): If True, returns a model pre-trained on ImageNet
  """
  model = DataAttentionResNet(SEBottleneck, [3, 4, 6, 3], se_loss=se_loss)
  return model

def data_att_resnet101(pretrained=False, se_loss=False):
  """Constructs a ResNet-101 model.
  Args:
    pretrained (bool): If True, returns a model pre-trained on ImageNet
  """
  model = DataAttentionResNet(SEBottleneck, [3, 4, 23, 3], se_loss=se_loss)
  if pretrained:
    model.load_state_dict(model_zoo.load_url(model_urls['resnet101']))
  return model

def data_att_resnet152(pretrained=False, se_loss=False):
  """Constructs a ResNet-152 model.
  Args:
    pretrained (bool): If True, returns a model pre-trained on ImageNet
  """
  model = DataAttentionResNet(SEBottleneck, [3, 8, 36, 3], se_loss=se_loss)
  if pretrained:
    model.load_state_dict(model_zoo.load_url(model_urls['resnet152']))
  return model

class Datasets_Attention(_fasterRCNN):
  def __init__(self, classes, num_layers=101, pretrained=False, class_agnostic=False, rpn_batchsize_list=None, se_loss=False, se_weight=1.0):
    self.se_loss = se_loss
    self.se_weight = se_weight
    self.model_path = '/home/Xwang/HeadNode-1/universal_model_/data/pretrained_model/se_resnet' + str(num_layers) + '.pth.tar'
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
    resnet = eval('data_att_resnet' + str(self.num_layers))(se_loss=self.se_loss)

    if self.pretrained == True:
      state_dict = torch.load(self.model_path)['state_dict']     
      print("Loading pretrained weights from %s" %(self.model_path))
      
      # load weight and bias of bn params for each datasets
    #   for k, v in state_dict.items():
    #     print(k)
    #   for k in resnet.state_dict():
    #     print(k)
      new_state_dict = {}
      
      for k, v in state_dict.items():
        k_list = k.split('.')
        name = ''
        for n in range(len(k_list)-1):
          if n == 0: continue
          if 'se' == k_list[n]: 
            se_name = name + 'datasets_attention.SE_Layers.'
          name += k_list[n] + '.'
        k_new = name + k_list[-1]
        if k_new in resnet.state_dict():
          new_state_dict[k_new] = v
        if 'bn' in k and 'layer' in k:
          for n in range(len(cfg.num_classes)):
            current = name + str(n) + '.' + k_list[-1]
            if current in resnet.state_dict():
              new_state_dict[current] = v
        elif 'downsample.1.' in  k:
          for n in range(len(cfg.num_classes)):
            current = name + 'bn.' + str(n) + '.' + k_list[-1]
            if current in resnet.state_dict():
              new_state_dict[current] = v
        elif '.se.' in k:
          for n in range(len(cfg.num_classes)):
            current = se_name + str(n) + '.' + k_list[-3] + '.' + k_list[-2] + '.' + k_list[-1]
            if current in resnet.state_dict():
              new_state_dict[current] = v          
        else:
          if k_new in resnet.state_dict():
            new_state_dict[k_new] = v
        for id in range(2):
          se_weight_name = k_list[1] + '.' + k_list[2] + '.datasets_attention.weight_' + str(id+1)
          if se_weight_name in resnet.state_dict():
            new_state_dict[se_weight_name] = resnet.state_dict()[se_weight_name]
          se_bias_name = k_list[1] + '.' + k_list[2] + '.datasets_attention.bias_' + str(id+1)
          if se_bias_name in resnet.state_dict():
            new_state_dict[se_bias_name] = resnet.state_dict()[se_bias_name]
          se_weight_name = k_list[1] + '.' + k_list[2] + '.datasets_attention.fc_'
          for appendix in ['1.weight','1.bias','2.weight','2.bias']:
            final_name = se_weight_name + appendix
            if final_name in resnet.state_dict():
              new_state_dict[final_name] = resnet.state_dict()[final_name]

        if 'fc.0.' in k:
          for n in range(len(cfg.num_classes)):
            current = se_name + str(n) + '.' + 'linear1.' + k_list[-1]
            if current in resnet.state_dict():
              new_state_dict[current] = v
        elif 'fc.2.' in k:
          for n in range(len(cfg.num_classes)):
            current = se_name + str(n) + '.' + 'linear2.' + k_list[-1]
            if current in resnet.state_dict():
              new_state_dict[current] = v
      for n in range(len(cfg.num_classes)):
        for tail in ['weight', 'bias']:
          for mid in ['3.5', '2.3']:
            current = 'layer' + mid + '.se.' + str(n) + '.seloss.' + tail
            if current in resnet.state_dict():
              new_state_dict[current] = resnet.state_dict()[current]
      resnet.load_state_dict(new_state_dict)
    # Build resnet.
    self.RCNN_base = nn.Sequential(resnet.conv1, resnet.bn1,resnet.relu,
      resnet.maxpool,resnet.layer1,resnet.layer2,resnet.layer3)

    self.RCNN_top = nn.Sequential(resnet.layer4)
    if self.num_layers == 18:
      self.RCNN_cls_score_layers = nn.ModuleList([nn.Linear(512, n_classes) for n_classes in cfg.num_classes])
    else:
      self.RCNN_cls_score_layers = nn.ModuleList([nn.Linear(2048, n_classes) for n_classes in cfg.num_classes])
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
            print('Require_grad', name)
          else:
            print('Not Require_grad', name)
            params.requires_grad = False
        for name, params in self.RCNN_top.named_parameters():
          if 'bn' in name:
            print('Require_grad', name)
          else:
            print('Not Require_grad', name)
            params.requires_grad = False

  def train(self, mode=True):
    # Override train so that the training mode is set as we want
    nn.Module.train(self, mode)
    if mode:
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
    # if isinstance(self.RCNN_top, nn.Sequential) and cfg.domain_pred:
    #   for neck in self.RCNN_top:
    #     print("before neck: ", len(pool5))
    #     pool5, domain_pred = neck(pool5)
    #     print("after neck: ", len(pool5))
    #   return pool5.mean(3).mean(2), domain_pred
    # else:
    #   fc7 = self.RCNN_top(pool5).mean(3).mean(2)
    fc7 = self.RCNN_top(pool5).mean(3).mean(2)
    return fc7