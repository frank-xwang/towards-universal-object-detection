# --------------------------------------------------------
# Fast R-CNN
# Copyright (c) 2015 Microsoft
# Licensed under The MIT License [see LICENSE for details]
# Written by Ross Girshick
# --------------------------------------------------------

"""Factory method for easily getting imdbs by name."""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

__sets = {}
from datasets.pascal_voc import pascal_voc
from datasets.kitti_voc import kitti_voc
from datasets.widerface_voc import widerface_voc
from datasets.coco import coco
from datasets.imagenet import imagenet
from datasets.cross_domain_voc import cross_domain
from datasets.dota_voc import dota_voc
from datasets.kitchen_voc import kitchen_voc
from datasets.vg import vg
from datasets.deeplesion_voc import deeplesion_voc
from datasets.LISA_voc import LISA_voc
from model.utils.config import cfg
import os

import numpy as np

# Set up voc_<year>_<split>
for year in ['2007', '2012']:
  for split in ['train', 'val', 'trainval', 'test']:
    name = 'voc_{}_{}'.format(year, split)
    __sets[name] = (lambda split=split, year=year: pascal_voc(split, year))

for split in ['train', 'val', 'trainval', 'test']:
  name = 'LISA_{}'.format(split)
  year='2007'
  __sets[name] = (lambda split=split, year=year: LISA_voc(split, year))

for split in ['train', 'val', 'trainval', 'test']:
  name = 'deeplesion_{}'.format(split)
  year='2007'
  __sets[name] = (lambda split=split, year=year: deeplesion_voc(split, year))

for split in ['train', 'val', 'trainval', 'test']:
  name = 'kitchen_{}'.format(split)
  year='2007'
  __sets[name] = (lambda split=split, year=year: kitchen_voc(split, year))

for split in ['train', 'val', 'trainval', 'test']:
  name = 'dota_{}'.format(split)
  year='2007'
  __sets[name] = (lambda split=split, year=year: dota_voc(split, year))

for split in ['train', 'val', 'trainval', 'test']:
    year = '2007'
    name = 'watercolor_{}'.format(split)
    __sets[name] = (lambda split=split, year=year, name=name: cross_domain(split, year, datasets='watercolor'))

for split in ['train', 'val', 'trainval', 'test']:
    year = '2007'
    name = 'comic_{}'.format(split)
    __sets[name] = (lambda split=split, year=year, name=name: cross_domain(split, year, datasets='comic'))

for split in ['train', 'val', 'trainval', 'test']:
    year = '2007'
    name = 'clipart_{}'.format(split)
    __sets[name] = (lambda split=split, year=year, name=name: cross_domain(split, year, datasets='clipart'))

# for year in ['2007', '2012']:
for split in ['train', 'val', 'trainval', 'test']:
  name = 'kittivoc_{}'.format(split)
  year='2007'
  __sets[name] = (lambda split=split, year=year: kitti_voc(split, year))

for split in ['train', 'val', 'trainval', 'test']:
    name = 'widerface_{}'.format(split)
    year = '2007'
    __sets[name] = (lambda split=split, year=year: widerface_voc(split, year)) 

# Set up coco_2014_<split>
for year in ['2014']:
  for split in ['train', 'val', 'minival', 'valminusminival', 'trainval']:
    name = 'coco_{}_{}'.format(year, split)
    __sets[name] = (lambda split=split, year=year: coco(split, year))

# Set up coco_2014_cap_<split>
for year in ['2014']:
  for split in ['train', 'val', 'capval', 'valminuscapval', 'trainval']:
    name = 'coco_{}_{}'.format(year, split)
    __sets[name] = (lambda split=split, year=year: coco(split, year))

# Set up coco_2015_<split>
for year in ['2015']:
  for split in ['test', 'test-dev']:
    name = 'coco_{}_{}'.format(year, split)
    __sets[name] = (lambda split=split, year=year: coco(split, year))
        
# set up image net.
for split in ['train', 'val', 'val1', 'val2', 'test']:
    name = 'imagenet_{}'.format(split)
    devkit_path = 'data/imagenet/ILSVRC/devkit'
    data_path = 'data/imagenet/ILSVRC'
    __sets[name] = (lambda split=split, devkit_path=devkit_path, data_path=data_path: imagenet(split,devkit_path,data_path))

def get_imdb(name):
  """Get an imdb (image database) by name."""
  if name not in __sets:
    raise KeyError('Unknown dataset: {}'.format(name))
  if cfg.dataset == 'cross_domain':
    return __sets[name](name=name)
  return __sets[name]()

def list_imdbs():
  """List all registered imdbs."""
  return list(__sets.keys())