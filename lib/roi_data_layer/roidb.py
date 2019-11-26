"""Transform a roidb into a trainable roidb by adding a bunch of metadata."""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import datasets
import numpy as np
from model.utils.config import cfg
from datasets.factory import get_imdb
import PIL
import pdb
import torch
def prepare_roidb(imdb):
  """Enrich the imdb's roidb by adding some derived quantities that
  are useful for training. This function precomputes the maximum
  overlap, taken over ground-truth boxes, between each ROI and
  each ground-truth box. The class with maximum overlap is also
  recorded.
  """
  roidb = imdb.roidb
  if not (imdb.name.startswith('coco')):
    sizes = [PIL.Image.open(imdb.image_path_at(i)).size
         for i in range(imdb.num_images)]
  fg_num = 0
  for i in range(len(imdb.image_index)):
    roidb[i]['img_id'] = imdb.image_id_at(i)
    roidb[i]['image'] = imdb.image_path_at(i)
    if not (imdb.name.startswith('coco')):
      roidb[i]['width'] = sizes[i][0]
      roidb[i]['height'] = sizes[i][1]
    # need gt_overlaps as a dense array for argmax
    gt_overlaps = roidb[i]['gt_overlaps'].toarray()
    # max overlap with gt over classes (columns)
    max_overlaps = gt_overlaps.max(axis=1)
    # gt class that had the max overlap
    max_classes = gt_overlaps.argmax(axis=1)
    roidb[i]['max_classes'] = max_classes
    roidb[i]['max_overlaps'] = max_overlaps
    # sanity checks
    zero_inds = np.where(max_overlaps == 0)[0]
    assert all(max_classes[zero_inds] == 0)
    nonzero_inds = np.where(max_overlaps > 0)[0]
    fg_num += len(np.where(max_overlaps == 1)[0])

def rank_roidb_ratio(roidb,imdb_names):
    # rank roidb based on the ratio between width and height.
    if 'kitti' in imdb_names:
      print('current imdb name is kitti, ratio_large is set as 4(original is 2)')
      ratio_large = 4.0
      ratio_small = 0.5
    else:
      ratio_large = 2.0 # largest ratio to preserve.
      ratio_small = 0.5 # smallest ratio to preserve.    

    ratio_list = []
    for i in range(len(roidb)):
      width = roidb[i]['width']
      height = roidb[i]['height']
      ratio = width / float(height)
      #print('ratio is: ',ratio, width, height)

      if ratio > ratio_large:
        roidb[i]['need_crop'] = 1
        ratio = ratio_large
      elif ratio < ratio_small:
        roidb[i]['need_crop'] = 1
        ratio = ratio_small        
      else:
        roidb[i]['need_crop'] = 0

      ratio_list.append(ratio)

    ratio_list = np.array(ratio_list)
    ratio_index = np.argsort(ratio_list)
    return ratio_list[ratio_index], ratio_index

def filter_roidb(roidb):
    # filter the image without bounding box.
    print('before filtering, there are %d images...' % (len(roidb)))
    i = 0
    while i < len(roidb):
      if cfg.imdb_name == "KITTIVOC":
        roidb[i]['boxes'] +=1
      if (len(roidb[i]['boxes']) == 0):
        if cfg.filter_empty:
          del roidb[i]
          i -= 1
        else:
          i+=1
        continue
      max_num = int(np.max(roidb[i]['boxes']))
      min_num = int(np.min(roidb[i]['boxes']))
      min_y1 = int(np.min(roidb[i]['boxes'][:,1]))
      max_y2 = int(np.max(roidb[i]['boxes'][:,3]))
      min_x1 = int(np.min(roidb[i]['boxes'][:,0]))
      max_x2 = int(np.max(roidb[i]['boxes'][:,2]))
      if max_num > 60000:
        print('error, please check rois coordinate of: \n %s is correct'%(roidb[i]['image']))
        print(roidb[i]['boxes'])
        index = np.where(roidb[i]['boxes'] > 60000)
        roidb[i]['boxes'][index] = 0
        print(roidb[i]['boxes'])
      if min_x1 == max_x2 or min_y1 == max_y2:
        del roidb[i]
        i -= 1
        continue
      i+=1

    print('after filtering, there are %d images...' % (len(roidb)))
    return roidb

def combined_roidb(imdb_names, training=True):
  """
  Combine multiple roidbs
  """

  def get_training_roidb(imdb):
    """Returns a roidb (Region of Interest database) for use in training."""
    if cfg.TRAIN.USE_FLIPPED:
      print('Appending horizontally-flipped training examples...')
      imdb.append_flipped_images()
      print('done')

    print('Preparing roidb data...')

    prepare_roidb(imdb)
    print('done')

    return imdb.roidb
  
  def get_roidb(imdb_name):
    imdb = get_imdb(imdb_name)
    print('dataset `{:s}` is loaded'.format(imdb.name))
    imdb.set_proposal_method(cfg.TRAIN.PROPOSAL_METHOD)
    roidb = get_training_roidb(imdb)
    return roidb
  roidbs = [get_roidb(s) for s in imdb_names.split('+')]
  roidb = roidbs[0]

  if len(roidbs) > 1:
    for r in roidbs[1:]:
      roidb.extend(r)
    tmp = get_imdb(imdb_names.split('+')[1])
    imdb = datasets.imdb.imdb(imdb_names, tmp.classes)
  else:
    imdb = get_imdb(imdb_names)

  if training:
    roidb = filter_roidb(roidb)
  ratio_list, ratio_index = rank_roidb_ratio(roidb, imdb_names)
  return imdb, roidb, ratio_list, ratio_index