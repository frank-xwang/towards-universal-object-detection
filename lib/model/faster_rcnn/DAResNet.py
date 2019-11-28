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
from model.faster_rcnn.domain_attention_module import DomainAttention

__all__ = ['DAResNet', 'da_resnet18', 'da_resnet34', 'da_resnet50', 'da_resnet101',
       'da_resnet152']

def conv3x3(in_planes, out_planes, stride=1):
  "3x3 convolution with padding"
  return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,
           padding=1, bias=False)

class DABasicBlock(nn.Module):
  expansion = 1

  def __init__(self, inplanes, planes, stride=1, downsample=None, fixed_block=False):
    super(DABasicBlock, self).__init__()
    self.conv1 = conv3x3(inplanes, planes, stride)
    if cfg.use_mux: self.bn1 = nn.ModuleList([nn.BatchNorm2d(planes) for datasets in cfg.num_classes])
    else: self.bn1 = nn.BatchNorm2d(planes)
    self.relu = nn.ReLU(inplace=True)
    self.conv2 = conv3x3(planes, planes)
    if cfg.use_mux: self.bn2 = nn.ModuleList([nn.BatchNorm2d(planes) for datasets in cfg.num_classes])
    else: self.bn2 = nn.BatchNorm2d(planes)
    self.domain_attention = DomainAttention(planes, reduction=16, nclass_list=cfg.num_classes, fixed_block=fixed_block)
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

    out = self.domain_attention(out)

    if self.downsample is not None:
      residual = self.downsample(x)

    out += residual
    out = self.relu(out)
    return out

class DABottleneck(nn.Module):
  expansion = 4

  def __init__(self, inplanes, planes, stride=1, downsample=None, fixed_block=False):
    super(DABottleneck, self).__init__()
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
    self.domain_attention = DomainAttention(planes * 4, reduction=16, nclass_list=cfg.num_classes, fixed_block=fixed_block)
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

    out = self.domain_attention(out)

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
        out = self.bn[cfg.cls_ind](x)
        return out
        
class DAResNet(nn.Module):
  def __init__(self, block, layers, num_classes=1000):
    self.inplanes = 64
    fixed_block = False
    super(DAResNet, self).__init__()
    self.conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3,
                 bias=False)                                        # rcnn.base.0
    self.bn1 = nn.BatchNorm2d(64)                                   # rcnn.base.1
    self.relu = nn.ReLU(inplace=True)                               # 2
    self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=0, ceil_mode=True) # 3
    cfg.start_fix = False
    if cfg.RESNET.FIXED_BLOCKS >= 1: fixed_block = True
    else:
      cfg.start_fix = True
      cfg.layer_index = 0
      cfg.block_id = 1
    self.layer1 = self._make_layer(block, 64, layers[0], fixed_block=fixed_block)              # rcnn.base.4
    fixed_block = False
    if cfg.RESNET.FIXED_BLOCKS >= 2: fixed_block = True
    else: 
      cfg.start_fix = True
      cfg.layer_index = 0
      cfg.block_id = 2
    self.layer2 = self._make_layer(block, 128, layers[1], stride=2, fixed_block=fixed_block) # rcnn.base.5
    fixed_block = False
    if cfg.RESNET.FIXED_BLOCKS >= 3: fixed_block = True
    else:
      cfg.start_fix = True
      cfg.layer_index = 0
      cfg.block_id = 3
    self.layer3 = self._make_layer(block, 256, layers[2], stride=2, fixed_block=fixed_block) # rcnn.base.6
    fixed_block = False
    if cfg.RESNET.FIXED_BLOCKS >= 4: fixed_block = True
    else: 
      cfg.start_fix = True
      cfg.layer_index = 0
      cfg.block_id = 4
    self.layer4 = self._make_layer(block, 512, layers[3], stride=2, fixed_block=fixed_block)   # rcnn.top
    fixed_block = False
    # it is slightly better whereas slower to set stride = 1
    # self.layer4 = self._make_layer(block, 512, layers[3], stride=1)
    self.avgpool = nn.AvgPool2d(7)
    # block.expansion is 1
    self.fc = nn.Linear(512 * block.expansion, num_classes)

    for m in self.modules():
      if isinstance(m, nn.Conv2d):
        n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
        m.weight.data.normal_(0, math.sqrt(2. / n))
      elif isinstance(m, nn.BatchNorm2d):
        m.weight.data.fill_(1)
        m.bias.data.zero_()

  def _make_layer(self, block, planes, blocks, stride=1, fixed_block=False):
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
    layers.append(block(self.inplanes, planes, stride, downsample,fixed_block=fixed_block))
    if cfg.start_fix:
      cfg.layer_index += 1
    self.inplanes = planes * block.expansion
    for i in range(1, blocks):
      if i == blocks - 1:
        layers.append(block(self.inplanes, planes, fixed_block=fixed_block))
      else:
        layers.append(block(self.inplanes, planes, fixed_block=fixed_block))
      if cfg.start_fix:
        cfg.layer_index += 1
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

def da_resnet18(pretrained=False):
  """Constructs a ResNet-18 model.
  Args:
    pretrained (bool): If True, returns a model pre-trained on ImageNet
  """
  model = DAResNet(DABasicBlock, [2, 2, 2, 2])
  return model

def da_resnet34(pretrained=False):
  """Constructs a ResNet-34 model.
  Args:
    pretrained (bool): If True, returns a model pre-trained on ImageNet
  """
  model = DAResNet(DABasicBlock, [3, 4, 6, 3])
  return model

def da_resnet50(pretrained=False):
  """Constructs a ResNet-50 model.
  Args:
    pretrained (bool): If True, returns a model pre-trained on ImageNet
  """
  model = DAResNet(DABottleneck, [3, 4, 6, 3])
  return model

def da_resnet101(pretrained=False):
  """Constructs a ResNet-101 model.
  Args:
    pretrained (bool): If True, returns a model pre-trained on ImageNet
  """
  model = DAResNet(DABottleneck, [3, 4, 23, 3])
  return model

def da_resnet152(pretrained=False):
  """Constructs a ResNet-152 model.
  Args:
    pretrained (bool): If True, returns a model pre-trained on ImageNet
  """
  model = DAResNet(DABottleneck, [3, 8, 36, 3])
  return model

class Domain_Attention(_fasterRCNN):
  def __init__(self, classes, num_layers=101, pretrained=False, class_agnostic=False, rpn_batchsize_list=None):
    extension = '_less.pth.tar' if cfg.less_blocks else '_full.pth.tar'
    self.model_path = 'data/pretrained_model/da_resnet' + str(num_layers) + '_' + str(cfg.num_adapters) + extension
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
    resnet = eval('da_resnet' + str(self.num_layers))()

    if self.pretrained == True:
      state_dict = torch.load(self.model_path)['state_dict']     
      print("Loading pretrained weights from %s" %(self.model_path))
      resnet.load_state_dict(state_dict)
    # Build resnet.
    # RCNN_base[0]: resnet.conv1, RCNN_base[1]: resnet.bn1 ......
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