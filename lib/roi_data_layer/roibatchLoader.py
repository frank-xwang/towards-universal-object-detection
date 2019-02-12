
"""The data layer used during training to train a Fast R-CNN network.
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import torch.utils.data as data
from PIL import Image
import torch

from roi_data_layer.minibatch import get_minibatch
from model.rpn.bbox_transform import bbox_transform_inv, clip_boxes

import numpy as np
import random
import time
import pdb
from model.utils.config import cfg

class roibatchLoader(data.Dataset):
  def __init__(self, roidb, ratio_list, ratio_index, batch_size, num_classes, datasets_name=None, training=True, normalize=None):
    self._roidb = roidb
    self._num_classes = num_classes
    self.datasets_name = datasets_name
    # we make the height of image consistent to trim_height, trim_width
    self.trim_height = cfg.TRAIN.TRIM_HEIGHT
    self.trim_width = cfg.TRAIN.TRIM_WIDTH
    self.max_num_box = cfg.MAX_NUM_GT_BOXES
    self.training = training
    self.normalize = normalize
    self.ratio_list = ratio_list
    self.ratio_index = ratio_index
    self.batch_size = batch_size
    self.data_size = len(self.ratio_list)

    # given the ratio_list, we want to make the ratio same for each batch.
    self.ratio_list_batch = torch.Tensor(self.data_size).zero_()
    self.target_size_list_600 = torch.Tensor(self.data_size).zero_()
    self.target_size_list_800 = torch.Tensor(self.data_size).zero_()
    self.target_size_list_576 = torch.Tensor(self.data_size).zero_()
    #self.ratio_list800 = [0.80, 0.84, 0.86, 0.88, 0.90, 0.92, 0.94, 0.96, 0.98, 1.00] # 640-816
    self.ratio_list800 = [0.76, 0.80, 0.84, 0.86, 0.88, 0.90, 0.92, 0.94, 0.96] # 640-816
    #self.ratio_list600 = [1.05, 1.04, 1.03, 1.02, 1.01, 1.00, 0.99] # 630-588
    self.ratio_list600 = [1.015, 1.01, 1.005, 1.00, 0.99] # 630-588
    self.ratio_list576 = [576, 600]
    self.left_idx_list = torch.Tensor(self.data_size).zero_()

    num_batch = int(np.ceil(len(ratio_index) / batch_size))
    for i in range(num_batch):
        left_idx = i*batch_size
        right_idx = min((i+1)*batch_size-1, self.data_size-1)

        if ratio_list[right_idx] < 1:
            # for ratio < 1, we preserve the leftmost in each batch.
            target_ratio = ratio_list[left_idx]
        elif ratio_list[left_idx] > 1:
            # for ratio > 1, we preserve the rightmost in each batch.
            target_ratio = ratio_list[right_idx]
        else:
            # for ratio cross 1, we make it to be 1.
            target_ratio = 1

        self.ratio_list_batch[left_idx:(right_idx+1)] = target_ratio
        if cfg.random_resize:
            self.target_size_list_600[left_idx:(right_idx+1)] = random.choice(self.ratio_list600)
            self.target_size_list_800[left_idx:(right_idx+1)] = random.choice(self.ratio_list800)
            self.target_size_list_576[left_idx:(right_idx+1)] = random.choice(self.ratio_list576)
            self.left_idx_list[left_idx:(right_idx+1)] = left_idx
 
  def reset_target_size(self):
    num_batch = int(np.ceil(len(self.ratio_index) / self.batch_size))
    for i in range(num_batch):
        left_idx = i*self.batch_size
        right_idx = min((i+1)*self.batch_size-1, self.data_size-1)
        
        self.target_size_list_600[left_idx:(right_idx+1)] = random.choice(self.ratio_list600)
        self.target_size_list_800[left_idx:(right_idx+1)] = random.choice(self.ratio_list800)
        self.target_size_list_576[left_idx:(right_idx+1)] = random.choice(self.ratio_list576)

        self.left_idx_list[left_idx:(right_idx+1)] = left_idx

  def test_reset_target_size(self):
    print('Target size are: ',self.target_size_list_800[0], self.target_size_list_600[0])
    self.reset_target_size()
    print('After reset, Target size are: ',self.target_size_list_800[0], self.target_size_list_600[0])

  # Used for getting certain index item. Usage: loader[index]
  def __getitem__(self, index):
    if self.training:
        index_ratio = int(self.ratio_index[index])
    else:
        index_ratio = index
    # get the anchor index for current sample index
    # here we set the anchor index to the last one
    # sample in this group
    minibatch_db = [self._roidb[index_ratio]]

    if cfg.random_resize:
        target_size576 = self.target_size_list_576[index]
        target_size600 = self.target_size_list_600[index]
        target_size800 = self.target_size_list_800[index]                       

        blobs = get_minibatch(minibatch_db, self._num_classes, target_size576, target_size600, target_size800)
    else:
        blobs = get_minibatch(minibatch_db, self._num_classes)

    data = torch.from_numpy(blobs['data'])
    im_info = torch.from_numpy(blobs['im_info'])
    # print('data size is: ', data.size())
    # we need to random shuffle the bounding box.
    data_height, data_width = data.size(1), data.size(2)
    # print('data_height, data_width',data_height, data_width, data.size())
    if self.training:
        np.random.shuffle(blobs['gt_boxes'])
        #print(blobs['gt_boxes'],blobs['gt_boxes'].shape)
        #print(blobs['gt_boxes'].shape)
        if blobs['gt_boxes'].shape[0]==0:
            blobs['gt_boxes'] = np.array([1.,1.,1.,1.,1.]).reshape((1,5))
        gt_boxes = torch.from_numpy(blobs['gt_boxes'])

        ########################################################
        # padding the input image to fixed size for each group #
        ########################################################

        # NOTE1: need to cope with the case where a group cover both conditions. (done)
        # NOTE2: need to consider the situation for the tail samples. (no worry)
        # NOTE3: need to implement a parallel data loader. (no worry)
        # get the index range

        # if the image need to crop, crop to the target size.
        ratio = self.ratio_list_batch[index]
        #print('========index_ratio is: ',ratio )

        if self._roidb[index_ratio]['need_crop']:
            # print('need_crop', data_height, data_width)
            if ratio < 1:
                # this means that data_width << data_height, we need to crop the
                # data_height
                min_y = int(torch.min(gt_boxes[:,1]))
                max_y = int(torch.max(gt_boxes[:,3]))
                trim_size = int(np.floor(data_width / ratio))
                if trim_size > data_height:
                    trim_size = data_height
                box_region = max_y - min_y + 1
                if min_y == 0:
                    y_s = 0
                else:
                    if (box_region-trim_size) < 0:
                        y_s_min = max(max_y-trim_size, 0)
                        y_s_max = min(min_y, data_height-trim_size)
                        if y_s_min == y_s_max:
                            y_s = y_s_min
                        else:
                            y_s = np.random.choice(range(y_s_min, y_s_max))
                    else:
                        y_s_add = int((box_region-trim_size)/2)
                        if y_s_add == 0:
                            y_s = min_y
                        else:
                            y_s = np.random.choice(range(min_y, min_y+y_s_add))
                # crop the image
                data = data[:, y_s:(y_s + trim_size), :, :]

                # shift y coordiante of gt_boxes
                #print('gt_boxes',gt_boxes[:, 1], float(x_s))
                #print(gt_boxes)
                gt_boxes[:, 1] = gt_boxes[:, 1] - float(y_s)
                gt_boxes[:, 3] = gt_boxes[:, 3] - float(y_s)

                # update gt bounding box according the trip
                gt_boxes[:, 1].clamp_(0, trim_size - 1)
                gt_boxes[:, 3].clamp_(0, trim_size - 1)

            else:
                # this means that data_width >> data_height, we need to crop the
                # data_width
                #print('ratio is: ', ratio)
                min_x = int(torch.min(gt_boxes[:,0]))
                max_x = int(torch.max(gt_boxes[:,2]))
                trim_size = int(np.ceil(data_height * ratio))
                if trim_size > data_width:
                    trim_size = data_width
                box_region = max_x - min_x + 1
                if min_x == 0:
                    x_s = 0
                else:
                    if (box_region-trim_size) < 0:
                        x_s_min = max(max_x-trim_size, 0)
                        x_s_max = min(min_x, data_width-trim_size)
                        if x_s_min == x_s_max:
                            x_s = x_s_min
                        else:
                            x_s = np.random.choice(range(x_s_min, x_s_max))
                    else:
                        x_s_add = int((box_region-trim_size)/2)
                        if x_s_add == 0:
                            x_s = min_x
                        else:
                            x_s = np.random.choice(range(min_x, min_x+x_s_add))
                # crop the image
                data = data[:, :, x_s:(x_s + trim_size), :]

                # shift x coordiante of gt_boxes
                gt_boxes[:, 0] = gt_boxes[:, 0] - float(x_s)
                gt_boxes[:, 2] = gt_boxes[:, 2] - float(x_s)
                # update gt bounding box according the trip
                gt_boxes[:, 0].clamp_(0, trim_size - 1)
                gt_boxes[:, 2].clamp_(0, trim_size - 1)

        # based on the ratio, padding the image.
        if ratio < 1:
            # this means that data_width < data_height
            trim_size = int(np.floor(data_width / ratio))

            padding_data = torch.FloatTensor(int(np.ceil(data_width / ratio)), \
                                             data_width, 3).zero_()

            padding_data[:data_height, :, :] = data[0]
            # update im_info
            im_info[0, 0] = padding_data.size(0)
            # print("height %d %d \n" %(index, anchor_idx))
        elif ratio > 1:
            # this means that data_width > data_height
            # if the image need to crop.
            if 'caltech' in self.datasets_name or 'KITTIVOC' in self.datasets_name:
                #print(data[0], data[0].shape)
                padding_data = torch.FloatTensor(data_height, \
                                                int(np.ceil(data_height * ratio)), 3).zero_()
                padding_data[:, :data_width, :] = data[0]
                im_info[0, 1] = padding_data.size(1)
            else:
                padding_data = torch.FloatTensor(data_height, \
                                                int(np.ceil(data_height * ratio)), 3).zero_()
                padding_data[:, :data_width, :] = data[0]
                im_info[0, 1] = padding_data.size(1)
        else:
            trim_size = min(data_height, data_width)
            padding_data = torch.FloatTensor(trim_size, trim_size, 3).zero_()
            padding_data = data[0][:trim_size, :trim_size, :]
            gt_boxes.clamp_(0, trim_size)
            im_info[0, 0] = trim_size
            im_info[0, 1] = trim_size

        # check the bounding box:
        not_keep = (gt_boxes[:,0] == gt_boxes[:,2]) | (gt_boxes[:,1] == gt_boxes[:,3])
        keep = torch.nonzero(not_keep == 0).view(-1)

        gt_boxes_padding = torch.FloatTensor(self.max_num_box, gt_boxes.size(1)).zero_()
        if keep.numel() != 0:
            gt_boxes = gt_boxes[keep]
            num_boxes = min(gt_boxes.size(0), self.max_num_box)
            gt_boxes_padding[:num_boxes,:] = gt_boxes[:num_boxes]
        else:
            num_boxes = 0

            # permute trim_data to adapt to downstream processing
        padding_data = padding_data.permute(2, 0, 1).contiguous()
        im_info = im_info.view(3)

        return padding_data, im_info, gt_boxes_padding, num_boxes
    else:
        data = data.permute(0, 3, 1, 2).contiguous().view(3, data_height, data_width)
        im_info = im_info.view(3)

        gt_boxes = torch.FloatTensor([1,1,1,1,1])
        num_boxes = 0

        return data, im_info, gt_boxes, num_boxes

  def __len__(self):
    return len(self._roidb)
