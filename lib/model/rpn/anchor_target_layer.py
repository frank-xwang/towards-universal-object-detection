from __future__ import absolute_import
# --------------------------------------------------------
# Faster R-CNN
# Copyright (c) 2015 Microsoft
# Licensed under The MIT License [see LICENSE for details]
# Written by Ross Girshick and Sean Bell
# --------------------------------------------------------
# --------------------------------------------------------
# Reorganized and modified by Jianwei Yang and Jiasen Lu
# --------------------------------------------------------

import torch
import torch.nn as nn
import numpy as np
import numpy.random as npr
import torch.nn.functional as F
from model.utils.config import cfg
from .generate_anchors import generate_anchors
from .bbox_transform import clip_boxes, bbox_overlaps_batch, bbox_transform_batch

import pdb

DEBUG = False

try:
    long        # Python 2
except NameError:
    long = int  # Python 3

class _AnchorTargetLayer(nn.Module):
    """
        Assign anchors to ground-truth targets. Produces anchor classification
        labels and bounding-box regression targets.
    """
    def __init__(self, feat_stride, scales, ratios, rpn_batchsize):
        super(_AnchorTargetLayer, self).__init__()

        self._feat_stride = feat_stride
        self._scales = scales
        anchor_scales = scales
        self._anchors = torch.from_numpy(generate_anchors(scales=np.array(anchor_scales), ratios=np.array(ratios))).float()
        self._num_anchors = self._anchors.size(0)
        self.rpn_batchsize = rpn_batchsize

        # allow boxes to sit over the edge by a small amount
        self._allowed_border = 0  # default is 0

    def forward(self, input):
        # Algorithm:
        #
        # for each (H, W) location i
        #   generate 9 anchor boxes centered on cell i
        #   apply predicted bbox deltas at cell i to each of the 9 anchors
        # filter out-of-image anchors
        def reshape(x, d):
            input_shape = x.size()
            x = x.view(
                input_shape[0],
                int(d),
                int(float(input_shape[1] * input_shape[2]) / float(d)),
                input_shape[3]
            )
            #print('x shape is: ', x.shape)
            return x

        if cfg.sample_mode == 'bootstrap':
            rpn_cls_score = input[0]
            gt_boxes = input[1]
            im_info = input[2]
            num_boxes = input[3]
            fg_prob = input[4][:, self._num_anchors:, :, :].contiguous()
            has_people = False
            # map of shape (batch_size, num_anchors, H, W)
            # rpn_cls_score is the score of each each anchors in
            # each positions(there are num_anchors in each position)
            height, width = rpn_cls_score.size(2), rpn_cls_score.size(3)
            #print('=== rpn_cls_score size: ', rpn_cls_score.size())

            batch_size = gt_boxes.size(0)

            feat_height, feat_width = rpn_cls_score.size(2), rpn_cls_score.size(3)
            shift_x = np.arange(0, feat_width) * self._feat_stride
            shift_y = np.arange(0, feat_height) * self._feat_stride
            shift_x, shift_y = np.meshgrid(shift_x, shift_y)
            shifts = torch.from_numpy(np.vstack((shift_x.ravel(), shift_y.ravel(),
                                    shift_x.ravel(), shift_y.ravel())).transpose())
            shifts = shifts.contiguous().type_as(rpn_cls_score).float()

            A = self._num_anchors # A: _num_anchors per image
            K = shifts.size(0) # K: number of positions in image
            #print('A, K are: ', A, K, shifts.size())

            self._anchors = self._anchors.type_as(gt_boxes) # move to specific gpu.
            #print('_anchors size is: ',self._anchors.size())
            all_anchors = self._anchors.view(1, A, 4) + shifts.view(K, 1, 4)
            #print('all_anchors size is: ',all_anchors.size())
            all_anchors = all_anchors.view(K * A, 4)
            #print(all_anchors.size(), all_anchors[0])
            #print('all_anchors size is: ', all_anchors.size())

            fg_prob = fg_prob.view(batch_size, -1, fg_prob.size(1)).contiguous() # 5*2700*14 
            #print('fg_prob shape after reshape 1 is: ',fg_prob.size())
            fg_prob = fg_prob.view(batch_size, -1).contiguous() # 5*37800
            #print('fg_prob shape after reshape 2 is: ',fg_prob.size())
            
            scores_keep = fg_prob
            proposals_keep = all_anchors

            total_anchors = int(K * A)

            keep = ((all_anchors[:, 0] >= -self._allowed_border) &
                    (all_anchors[:, 1] >= -self._allowed_border) &
                    (all_anchors[:, 2] < long(im_info[0][1]) + self._allowed_border) &
                    (all_anchors[:, 3] < long(im_info[0][0]) + self._allowed_border))

            inds_inside = torch.nonzero(keep).view(-1)

            # keep only inside anchors
            #anchors = proposals_keep[inds_inside, :]
            proposals_keep = proposals_keep[inds_inside, :]
            scores_keep = scores_keep[:,inds_inside]
            _, order = torch.sort(scores_keep, 1, True)  

            # label: 1 is positive, 0 is negative, -1 is dont care
            labels = gt_boxes.new(batch_size, inds_inside.size(0)).fill_(-1)
            bbox_inside_weights = gt_boxes.new(batch_size, inds_inside.size(0)).zero_()
            bbox_outside_weights = gt_boxes.new(batch_size, inds_inside.size(0)).zero_()

            # gt_labelNbox[:,:,4] is the label of corresponding boxes
            if cfg.imdb_name == "caltech_train":
                overlaps, gt_labelNbox = bbox_overlaps_batch(proposals_keep, gt_boxes)
            else:
                overlaps = bbox_overlaps_batch(proposals_keep, gt_boxes)
            #print('overlaps and people index shape are: ', overlaps.size(), gt_labelNbox.size())
            #print('people_inds sum is: ', torch.sum(gt_labelNbox[:,:,4] == 2))

            max_overlaps, argmax_overlaps = torch.max(overlaps, 2)
            #print('max_overlaps size is: ', max_overlaps.size(), argmax_overlaps.size())
            # index of ground truth
            gt_max_overlaps, _ = torch.max(overlaps, 1)
            #print('gt_max_overlaps size is: ', gt_max_overlaps.size())

            if not cfg.TRAIN.RPN_CLOBBER_POSITIVES:
                labels[max_overlaps < cfg.TRAIN.RPN_NEGATIVE_OVERLAP] = 0

            gt_max_overlaps[gt_max_overlaps==0] = 1e-5
            keep = torch.sum(overlaps.eq(gt_max_overlaps.view(batch_size,1,-1).expand_as(overlaps)), 2)

            if torch.sum(keep) > 0:
                labels[keep>0] = 1

            # fg label: above threshold IOU
            labels[max_overlaps >= cfg.TRAIN.RPN_POSITIVE_OVERLAP] = 1

            if cfg.TRAIN.RPN_CLOBBER_POSITIVES:
                labels[max_overlaps < cfg.TRAIN.RPN_NEGATIVE_OVERLAP] = 0
            
            ### wrote by Xudong Wang
            # get index of people, change the lable of corresonding index to be -1, so, all the 
            # anchors corrsponding to people will be ignored
            if cfg.imdb_name == "caltech_train":
                people_inds = (gt_labelNbox[:,:,4] == 2)
                if torch.sum(people_inds) != 0:
                    has_people = True
                    for i in range(batch_size):
                        single_p_inds = people_inds[i].cpu().numpy()
                        p_list = np.where(single_p_inds == 1)[0]
                        if cfg.DEBUG:  
                            print('p_list is: ', p_list)
                        for j in p_list:
                            labels[i][(argmax_overlaps[i] == j) & (max_overlaps[i] >= cfg.TRAIN.RPN_NEGATIVE_OVERLAP)] = -1
            ### End

            num_fg = int(cfg.TRAIN.RPN_FG_FRACTION * cfg.TRAIN.RPN_BATCHSIZE)

            sum_fg = torch.sum((labels == 1).int(), 1)
            sum_bg = torch.sum((labels == 0).int(), 1)

            for i in range(batch_size):
                # subsample positive labels if we have too many
                proposals_single = proposals_keep[i]
                scores_single = scores_keep[i]
                
                if sum_fg[i] > num_fg:
                    fg_inds = torch.nonzero(labels[i] == 1).view(-1)
                    # torch.randperm seems has a bug on multi-gpu setting that cause the segfault.
                    # See https://github.com/pytorch/pytorch/issues/1868 for more details.
                    # use numpy instead.
                    #rand_num = torch.randperm(fg_inds.size(0)).type_as(gt_boxes).long()
                    rand_num = torch.from_numpy(np.random.permutation(fg_inds.size(0))).type_as(gt_boxes).long()
                    disable_inds = fg_inds[rand_num[:fg_inds.size(0)-num_fg]]
                    labels[i][disable_inds] = -1
                
                num_bg = cfg.TRAIN.RPN_BATCHSIZE - sum_fg[i]
                #print('cfg.TRAIN.RPN_BATCHSIZE',cfg.TRAIN.RPN_BATCHSIZE)

                # subsample negative labels if we have too many
                if sum_bg[i] > num_bg:
                    ### Wrote by Xudong Wang
                    bg_inds = torch.nonzero(labels[i] == 0).view(-1)
                    scores_single_bg = scores_single[bg_inds]
                    _, order_single = torch.sort(scores_single_bg, 0, True)
                    disable_inds = bg_inds[order_single[:bg_inds.size(0)-num_bg]]
                    labels[i][disable_inds] = -1
                    ### End

            offset = torch.arange(0, batch_size)*gt_boxes.size(1)

            argmax_overlaps = argmax_overlaps + offset.view(batch_size, 1).type_as(argmax_overlaps)
            bbox_targets = _compute_targets_batch(proposals_keep, gt_boxes.view(-1,5)[argmax_overlaps.view(-1), :].view(batch_size, -1, 5))

            # use a single value instead of 4 values for easy index.
            bbox_inside_weights[labels==1] = cfg.TRAIN.RPN_BBOX_INSIDE_WEIGHTS[0]
            if cfg.TRAIN.RPN_POSITIVE_WEIGHT < 0:
                num_examples = torch.sum(labels[i] >= 0).item()
                positive_weights = 1.0 / num_examples
                negative_weights = 1.0 / num_examples
            else:
                assert ((cfg.TRAIN.RPN_POSITIVE_WEIGHT > 0) &
                        (cfg.TRAIN.RPN_POSITIVE_WEIGHT < 1))

            bbox_outside_weights[labels == 1] = positive_weights
            bbox_outside_weights[labels == 0] = negative_weights

            labels = _unmap(labels, total_anchors, inds_inside, batch_size, fill=-1)
            bbox_targets = _unmap(bbox_targets, total_anchors, inds_inside, batch_size, fill=0)
            bbox_inside_weights = _unmap(bbox_inside_weights, total_anchors, inds_inside, batch_size, fill=0)
            bbox_outside_weights = _unmap(bbox_outside_weights, total_anchors, inds_inside, batch_size, fill=0)

            outputs = []

            labels = labels.view(batch_size, height, width, A).permute(0,3,1,2).contiguous()
            labels = labels.view(batch_size, 1, A * height, width)
            outputs.append(labels)

            bbox_targets = bbox_targets.view(batch_size, height, width, A*4).permute(0,3,1,2).contiguous()
            outputs.append(bbox_targets)

            anchors_count = bbox_inside_weights.size(1)
            bbox_inside_weights = bbox_inside_weights.view(batch_size,anchors_count,1).expand(batch_size, anchors_count, 4)

            bbox_inside_weights = bbox_inside_weights.contiguous().view(batch_size, height, width, 4*A)\
                                .permute(0,3,1,2).contiguous()

            outputs.append(bbox_inside_weights)

            bbox_outside_weights = bbox_outside_weights.view(batch_size,anchors_count,1).expand(batch_size, anchors_count, 4)
            bbox_outside_weights = bbox_outside_weights.contiguous().view(batch_size, height, width, 4*A)\
                                .permute(0,3,1,2).contiguous()
            outputs.append(bbox_outside_weights)

            return outputs, has_people

        else:
            rpn_cls_score = input[0]
            gt_boxes = input[1]
            im_info = input[2]
            num_boxes = input[3]

            # map of shape (batch_size, num_anchors, H, W)
            # rpn_cls_score is the score of each each anchors in 
            # each positions(there are num_anchors in each position)
            height, width = rpn_cls_score.size(2), rpn_cls_score.size(3)
            #print('=== rpn_cls_score size: ', rpn_cls_score.size())

            batch_size = gt_boxes.size(0)

            feat_height, feat_width = rpn_cls_score.size(2), rpn_cls_score.size(3)
            shift_x = np.arange(0, feat_width) * self._feat_stride
            shift_y = np.arange(0, feat_height) * self._feat_stride
            shift_x, shift_y = np.meshgrid(shift_x, shift_y, copy=False)
            shifts = torch.from_numpy(np.vstack((shift_x.ravel(), shift_y.ravel(),
                                    shift_x.ravel(), shift_y.ravel())).transpose())
            shifts = shifts.contiguous().type_as(rpn_cls_score).float()

            A = self._num_anchors # A: _num_anchors per image
            K = shifts.size(0) # K: number of positions in image

            self._anchors = self._anchors.type_as(gt_boxes) # move to specific gpu.
            all_anchors = self._anchors.view(1, A, 4) + shifts.view(K, 1, 4)
            all_anchors = all_anchors.view(K * A, 4)

            total_anchors = int(K * A)

            keep = ((all_anchors[:, 0] >= -self._allowed_border) &
                    (all_anchors[:, 1] >= -self._allowed_border) &
                    (all_anchors[:, 2] < long(im_info[0][1]) + self._allowed_border) &
                    (all_anchors[:, 3] < long(im_info[0][0]) + self._allowed_border))

            inds_inside = torch.nonzero(keep).view(-1)

            # keep only inside anchors
            anchors = all_anchors[inds_inside, :]

            # label: 1 is positive, 0 is negative, -1 is dont care
            labels = gt_boxes.new(batch_size, inds_inside.size(0)).fill_(-1)
            bbox_inside_weights = gt_boxes.new(batch_size, inds_inside.size(0)).zero_()
            bbox_outside_weights = gt_boxes.new(batch_size, inds_inside.size(0)).zero_()

            overlaps = bbox_overlaps_batch(anchors, gt_boxes)
            max_overlaps, argmax_overlaps = torch.max(overlaps, 2)
            gt_max_overlaps, _ = torch.max(overlaps, 1)

            if not cfg.TRAIN.RPN_CLOBBER_POSITIVES:
                labels[max_overlaps < cfg.TRAIN.RPN_NEGATIVE_OVERLAP] = 0

            gt_max_overlaps[gt_max_overlaps==0] = 1e-5
            keep = torch.sum(overlaps.eq(gt_max_overlaps.view(batch_size,1,-1).expand_as(overlaps)), 2)

            if torch.sum(keep) > 0:
                labels[keep>0] = 1

            # fg label: above threshold IOU
            #print('RPN_POSITIVE_OVERLAP',cfg.RPN_POSITIVE_OVERLAP)
            labels[max_overlaps >= cfg.TRAIN.RPN_POSITIVE_OVERLAP] = 1

            if cfg.TRAIN.RPN_CLOBBER_POSITIVES:
                labels[max_overlaps < cfg.TRAIN.RPN_NEGATIVE_OVERLAP] = 0

            num_fg = int(cfg.TRAIN.RPN_FG_FRACTION * cfg.TRAIN.RPN_BATCHSIZE)
            #print('rpn batch fg size is: ', num_fg)

            sum_fg = torch.sum((labels == 1).int(), 1)
            sum_bg = torch.sum((labels == 0).int(), 1)

            for i in range(batch_size):
                # subsample positive labels if we have too many
                if sum_fg[i] > num_fg:
                    fg_inds = torch.nonzero(labels[i] == 1).view(-1)
                    # torch.randperm seems has a bug on multi-gpu setting that cause the segfault.
                    # See https://github.com/pytorch/pytorch/issues/1868 for more details.
                    # use numpy instead.
                    #rand_num = torch.randperm(fg_inds.size(0)).type_as(gt_boxes).long()
                    rand_num = torch.from_numpy(np.random.permutation(fg_inds.size(0))).type_as(gt_boxes).long()
                    disable_inds = fg_inds[rand_num[:fg_inds.size(0)-num_fg]]
                    labels[i][disable_inds] = -1
                
                num_bg = cfg.TRAIN.RPN_BATCHSIZE - sum_fg[i]
                #print('cfg.TRAIN.RPN_BATCHSIZE',cfg.TRAIN.RPN_BATCHSIZE)

                # subsample negative labels if we have too many
                if sum_bg[i] > num_bg:
                    bg_inds = torch.nonzero(labels[i] == 0).view(-1)
                    #rand_num = torch.randperm(bg_inds.size(0)).type_as(gt_boxes).long()

                    rand_num = torch.from_numpy(np.random.permutation(bg_inds.size(0))).type_as(gt_boxes).long()
                    disable_inds = bg_inds[rand_num[:bg_inds.size(0)-num_bg]]
                    labels[i][disable_inds] = -1

            offset = torch.arange(0, batch_size)*gt_boxes.size(1)

            argmax_overlaps = argmax_overlaps + offset.view(batch_size, 1).type_as(argmax_overlaps)
            bbox_targets = _compute_targets_batch(anchors, gt_boxes.view(-1,5)[argmax_overlaps.view(-1), :].view(batch_size, -1, 5))

            # use a single value instead of 4 values for easy index.
            bbox_inside_weights[labels==1] = cfg.TRAIN.RPN_BBOX_INSIDE_WEIGHTS[0]

            if cfg.TRAIN.RPN_POSITIVE_WEIGHT < 0:
                num_examples = torch.sum(labels[i] >= 0).item()
                positive_weights = 1.0 / num_examples
                negative_weights = 1.0 / num_examples
            else:
                assert ((cfg.TRAIN.RPN_POSITIVE_WEIGHT > 0) &
                        (cfg.TRAIN.RPN_POSITIVE_WEIGHT < 1))

            bbox_outside_weights[labels == 1] = positive_weights
            bbox_outside_weights[labels == 0] = negative_weights

            labels = _unmap(labels, total_anchors, inds_inside, batch_size, fill=-1)
            bbox_targets = _unmap(bbox_targets, total_anchors, inds_inside, batch_size, fill=0)
            bbox_inside_weights = _unmap(bbox_inside_weights, total_anchors, inds_inside, batch_size, fill=0)
            bbox_outside_weights = _unmap(bbox_outside_weights, total_anchors, inds_inside, batch_size, fill=0)

            outputs = []

            labels = labels.view(batch_size, height, width, A).permute(0,3,1,2).contiguous()
            labels = labels.view(batch_size, 1, A * height, width)
            outputs.append(labels)

            bbox_targets = bbox_targets.view(batch_size, height, width, A*4).permute(0,3,1,2).contiguous()
            outputs.append(bbox_targets)

            anchors_count = bbox_inside_weights.size(1)
            bbox_inside_weights = bbox_inside_weights.view(batch_size,anchors_count,1).expand(batch_size, anchors_count, 4)

            bbox_inside_weights = bbox_inside_weights.contiguous().view(batch_size, height, width, 4*A)\
                                .permute(0,3,1,2).contiguous()

            outputs.append(bbox_inside_weights)

            bbox_outside_weights = bbox_outside_weights.view(batch_size,anchors_count,1).expand(batch_size, anchors_count, 4)
            bbox_outside_weights = bbox_outside_weights.contiguous().view(batch_size, height, width, 4*A)\
                                .permute(0,3,1,2).contiguous()
            outputs.append(bbox_outside_weights)

            return outputs

    def backward(self, top, propagate_down, bottom):
        """This layer does not propagate gradients."""
        pass

    def reshape(self, bottom, top):
        """Reshaping happens during the call to forward."""
        pass

def _unmap(data, count, inds, batch_size, fill=0):
    """ Unmap a subset of item (data) back to the original set of items (of
    size count) """

    if data.dim() == 2:
        ret = torch.Tensor(batch_size, count).fill_(fill).type_as(data)
        ret[:, inds] = data
    else:
        ret = torch.Tensor(batch_size, count, data.size(2)).fill_(fill).type_as(data)
        ret[:, inds,:] = data
    return ret


def _compute_targets_batch(ex_rois, gt_rois):
    """Compute bounding-box regression targets for an image."""

    return bbox_transform_batch(ex_rois, gt_rois[:, :, :4])
