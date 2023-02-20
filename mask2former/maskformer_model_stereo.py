# Copyright (c) Facebook, Inc. and its affiliates.
from collections import OrderedDict
from turtle import forward
from typing import Tuple
from numpy import float16
import numpy as np
import math

import torch
from torch import nn
from torch.nn import functional as F

from scipy.optimize import linear_sum_assignment
from torch.cuda.amp import autocast

import fvcore.nn.weight_init as weight_init
from detectron2.config import configurable
from detectron2.data import MetadataCatalog
from detectron2.layers import Conv2d
from detectron2.layers.shape_spec import ShapeSpec
from detectron2.modeling import META_ARCH_REGISTRY, build_backbone, build_sem_seg_head
from detectron2.modeling.backbone import Backbone
from detectron2.modeling.postprocessing import sem_seg_postprocess
from detectron2.structures import Boxes, ImageList, Instances, BitMasks
from detectron2.utils.memory import retry_if_cuda_oom

from .modeling.criterion import SetCriterion, calculate_uncertainty, dice_loss_jit, sigmoid_ce_loss_jit
from .modeling.matcher import HungarianMatcher
from .modeling.pixel_decoder.fpn import BasePixelDecoder

from detectron2.utils.comm import get_world_size
from detectron2.projects.point_rend.point_features import (
    get_uncertain_point_coords_with_randomness,
    point_sample
)

from .utils.misc import nested_tensor_from_tensor_list

def smooth_l1_loss(
        inputs: torch.Tensor,
        targets: torch.Tensor,
        num_masks: float,
    ):
    """
    Args:
        inputs: A float tensor of arbitrary shape.
                The predictions for each example.
        targets: A float tensor with the same shape as inputs. Stores the binary
                 classification label for each element in inputs
                (0 for the negative class and 1 for the positive class).
    Returns:
        Loss tensor
    """

    loss = F.smooth_l1_loss(inputs, targets, beta=1.0, reduction="mean")

    return loss #.mean(1).sum() / num_masks

def batch_smooth_l1_loss(
        queries: torch.Tensor,
        targets: torch.Tensor,
        num_masks: float,
    ):
    """
    Args:
        inputs: A float tensor of arbitrary shape.
                The predictions for each example.
        targets: A float tensor with the same shape as inputs. Stores the binary
                 classification label for each element in inputs
                (0 for the negative class and 1 for the positive class).
    Returns:
        Loss tensor
    """
    num_queries = queries.shape[0]
    num_targets = targets.shape[0]

    loss = torch.full([num_queries, num_targets], 0).to(queries)
    for q in range(num_queries):
        for t in range(num_targets):
            loss[q, t] = F.smooth_l1_loss(queries[q], targets[t], beta=1.0, reduction="mean")

    return loss #.mean(1).sum() / num_masks

class SetCriterionStereo(SetCriterion):

    def loss_labels(self, outputs, targets, indices, num_masks):
        """Classification loss (NLL)
        targets dicts must contain the key "labels" containing a tensor of dim [nb_target_boxes]
        """
        assert "pred_logits" in outputs
        src_logits = outputs["pred_logits"].float()

        idx = self._get_src_permutation_idx(indices)
        target_classes_o = torch.cat([t["labels"][J] for t, (_, J) in zip(targets, indices)])
        target_classes = torch.full(
            src_logits.shape[:2], 0.0, dtype=torch.int64, device=src_logits.device
        )
        target_classes[idx] = target_classes_o

        loss_ce = F.cross_entropy(src_logits.transpose(1, 2), target_classes, self.empty_weight)
        losses = {"loss_ce": loss_ce}
        return losses
    
    def loss_masks(self, outputs, targets, indices, num_masks):
        """Compute the losses related to the masks: the focal loss and the dice loss.
        targets dicts must contain the key "masks" containing a tensor of dim [nb_target_boxes, h, w]
        """
        assert "pred_masks" in outputs

        # TODO make this configurable
        kitti = False
        src_idx = self._get_src_permutation_idx(indices)
        tgt_idx = self._get_tgt_permutation_idx(indices)
        src_masks = outputs["pred_masks"]
        src_masks = src_masks[src_idx]
        masks = [t["masks"] for t in targets]
        # TODO use valid to mask invalid areas due to padding in loss
        target_masks, valid = nested_tensor_from_tensor_list(masks).decompose()
        target_masks = target_masks.to(src_masks)
        target_masks = target_masks[tgt_idx]

        if kitti:
            loss_mask = 0
            loss_dice = 0
            src_masks =  src_masks = F.interpolate(
                src_masks[None],
                size=(target_masks.shape[-2], target_masks.shape[-1]),
                mode="bilinear",
                align_corners=False,
            ).squeeze()
            for t_ind in range(len(targets)):
                valid_mask = targets[t_ind]['sem_seg'] > 0
                t_src_masks = src_masks[src_idx[0] == t_ind]
                t_target_masks = target_masks[tgt_idx[0] == t_ind]
                loss_mask += sigmoid_ce_loss_jit(t_src_masks[:,valid_mask], t_target_masks[:,valid_mask], num_masks=1)
                loss_dice += dice_loss_jit(t_src_masks[:,valid_mask], t_target_masks[:,valid_mask], num_masks=1)

            loss_mask = loss_mask/num_masks
            loss_dice = loss_dice/num_masks


        else:
            # No need to upsample predictions as we are using normalized coordinates :)
            # N x 1 x H x W
            src_masks = src_masks[:, None]
            target_masks = target_masks[:, None]

            with torch.no_grad():
                # sample point_coords
                point_coords = get_uncertain_point_coords_with_randomness(
                    src_masks,
                    lambda logits: calculate_uncertainty(logits),
                    self.num_points,
                    self.oversample_ratio,
                    self.importance_sample_ratio,
                )
                # get gt labels
                point_labels = point_sample(
                    target_masks,
                    point_coords,
                    align_corners=False,
                ).squeeze(1)

            point_logits = point_sample(
                src_masks,
                point_coords,
                align_corners=False,
            ).squeeze(1)

            loss_mask = sigmoid_ce_loss_jit(point_logits, point_labels, num_masks)
            loss_dice = dice_loss_jit(point_logits, point_labels, num_masks)

        losses = {
            "loss_mask": loss_mask,
            "loss_dice": loss_dice
        }

        del src_masks
        del target_masks
        return losses

    def loss_segs(self, outputs, targets, indices, num_masks):
        gt_seg = [t["sem_seg"].unsqueeze(0) for t in targets]
        gt_seg = torch.cat(gt_seg)
        # mask_pred = outputs["pred_masks"]
        semseg = outputs["pred_seg"].squeeze(1)
        # mask_pred = F.interpolate(
        #         mask_pred[None],
        #         size=(192, gt_seg.shape[-2], gt_seg.shape[-1]),
        #         mode="trilinear",
        #         align_corners=False,
        # ).squeeze(0)
        # semseg = F.interpolate(
        #         outputs['pred_seg'],
        #         size=(gt_seg.shape[-2], gt_seg.shape[-1]),
        #         mode="bilinear",
        #         align_corners=False,
        # ).squeeze() * 4
        # mask_pred = mask_pred.softmax(1)
        # classes = torch.tensor([range(1,mask_pred.shape[1]+1)]).to(mask_pred)[:,:,None,None]
        # semseg = (mask_pred * classes).sum(1)
        valid_mask = gt_seg <= 192
        loss_seg = smooth_l1_loss(semseg[valid_mask], gt_seg[valid_mask].float(), num_masks)
        losses = {"loss_seg": loss_seg}
        return losses


    def get_loss(self, loss, outputs, targets, indices, num_masks):
        loss_map = {
            'labels': self.loss_labels,
            'masks': self.loss_masks,
            'segs': self.loss_segs,
        }
        assert loss in loss_map, f"do you really want to compute {loss} loss?"
        return loss_map[loss](outputs, targets, indices, num_masks)

#     def loss_labels(self, outputs, targets, indices, num_masks):
#         """Classification loss (NLL)
#         targets dicts must contain the key "labels" containing a tensor of dim [nb_target_boxes]
#         """
#         assert "pred_logits" in outputs
#         src_logits = outputs["pred_logits"].float()
#         src_logits = (F.softmax(src_logits, dim=-1)* torch.tensor([range(src_logits.shape[-1])]).to(src_logits)).sum(dim=-1)

#         idx = self._get_src_permutation_idx(indices)
#         target_classes_o = torch.cat([t["labels"][J] for t, (_, J) in zip(targets, indices)])
#         target_classes = torch.full(
#             src_logits.shape[:2], 0, dtype=torch.int64, device=src_logits.device
#         )
#         target_classes[idx] = target_classes_o

#         loss_ce = smooth_l1_loss(src_logits, target_classes.float(), num_masks)
#         # loss_ce = F.cross_entropy(src_logits.transpose(1, 2), target_classes, self.empty_weight)
#         losses = {"loss_ce": loss_ce}
#         return losses

#     def loss_masks(self, outputs, targets, indices, num_masks):
#         """Compute the losses related to the masks: the focal loss and the dice loss.
#         targets dicts must contain the key "masks" containing a tensor of dim [nb_target_boxes, h, w]
#         """
#         assert "pred_masks" in outputs

#         # src_idx = self._get_src_permutation_idx(indices)
#         # tgt_idx = self._get_tgt_permutation_idx(indices)
#         # target_classes = torch.cat([t["labels"][J] for t, (_, J) in zip(targets, indices)])[..., None, None]
#         # src_masks = outputs["pred_masks"]
        
#         src_logits = F.softmax(outputs["pred_logits"], dim=-1)[..., 1:]
#         classes = torch.tensor([range(1,src_logits.shape[-1]+1)]).to(src_logits)[...,None,None]
#         # classes = torch.tensor([range(src_logits.shape[-1])]).to(src_logits)[None, ...]
#         # src_classes = (src_logits*classes).sum(-1)[..., None, None]
#         # src_classes = src_classes[src_idx]
#         # src_masks = src_masks.sigmoid() * src_classes
#         # src_masks = src_masks[src_idx]

#         src_masks = outputs["pred_masks"]



#         bs = len(targets)
#         # argsoftmax of object queries
#         # mask_cls = F.softmax(outputs["pred_logits"], dim-1)[..., :-1]
#         # mask_cls = mask_cls[src_idx]
#         src_masks = src_masks.sigmoid()
#         # src_masks = src_masks[src_idx]*src_classes
#         # src_masks = (src_masks * src_classes).sum(1)
#         # src_masks = torch.cat([(src_masks[src_idx[0]==b, :]).sum(dim=0, keepdim=True) for b in range(bs)])
#         src_masks = (torch.einsum("bqc,bqhw->bchw", src_logits, src_masks) * classes).sum(1)
#         # disp_masks = (F.softmax(disp_masks,dim=1) * torch.tensor([range(disp_masks.shape[1])])[...,None,None].to(disp_masks)).sum(1)

#         #scratch
#         # save_dir = '/home/nstarli/Mask2Former/work_dirs/argsoftmax_sigmoid_smoothl1/debug_vis'
#         # import matplotlib.pyplot as plt
#         # import os
#         # for idx, _ in enumerate(src_idx[0][src_idx[0]==0]):
#         # for idx, predmask in enumerate(pred_masks_unmatched[0]):
#         #     plt.imshow(predmask.detach().cpu())
#         #     plt.savefig(os.path.join(save_dir, f'allObjectQueries{idx}.png'))

#         masks = [t["masks"] for t in targets]
#         # TODO use valid to mask invalid areas due to padding in loss
#         target_disps = torch.cat([(t['masks'] * t['labels'][..., None, None]).sum(0,keepdim=True) for t in targets])
#         # target_disps = torch.cat([(t['masks'][J] * t['labels'][J, None, None]).sum(0, keepdim=True) for t, (_, J) in zip(targets, indices)])
#         target_disps = target_disps.to(src_masks)

#         # target_masks, valid = nested_tensor_from_tensor_list(masks).decompose()
#         # target_disps = target_disps.to(disp_masks)
#         # target_masks = target_masks.to(src_masks)
#         # target_masks = target_masks[tgt_idx]

#         # bs = len(targets)
# # 
#         # src_masks = torch.cat([(F.softmax(src_masks[src_idx[0]==b, :],0)*target_classes[src_idx[0]==b]).sum(dim=0, keepdim=True) for b in range(bs)])
#         # src_masks_disp = src_masks_disp.to(src_masks)

#         # for b in range(bs):
#         #     target_masks[tgt_idx[0]==b, :] *= target_classes[tgt_idx[0]==b]
#         # target_masks_disp = [(target_masks[tgt_idx[0]==b, :]*target_classes[tgt_idx[0]==b]).sum(dim=0, keepdim=True) for b in range(bs)]
#         # target_masks_disp, valid = nested_tensor_from_tensor_list(target_masks_disp).decompose()
#         # target_masks_disp = target_masks_disp.to(src_masks).squeeze()

#         # Upsample disparity predictions
#         # N x 1 x H x W
#         # src_masks = F.interpolate(
#         #     src_masks[:, None], size=target_disps.shape[-2:], mode="bilinear", align_corners=True
#         # )
#         src_masks = src_masks.flatten(1)
#         target_disps = target_disps.flatten(1)
#         # target_masks_disp = target_masks.view(src_masks.shape)
#         valid_mask = target_disps != 0

#         # # do not need to upsample for dice loss since we are using normalized point coords for point rend 
#         # src_masks = src_masks[:, None]
#         # target_masks = target_masks[:, None]

#         # with torch.no_grad():
#         #     # sample point_coords
#         #     point_coords = get_uncertain_point_coords_with_randomness(
#         #         src_masks,
#         #         lambda logits: calculate_uncertainty(logits),
#         #         self.num_points,
#         #         self.oversample_ratio,
#         #         self.importance_sample_ratio,
#         #     )
#         #     # get gt labels
#         #     point_labels = point_sample(
#         #         target_masks,
#         #         point_coords,
#         #         align_corners=False,
#         #     ).squeeze(1)

#         # point_logits = point_sample(
#         #     src_masks,
#         #     point_coords,
#         #     align_corners=False,
#         # ).squeeze(1)

        # losses = {
        #     # "loss_mask": sigmoid_ce_loss_jit(point_logits, point_labels, num_masks),
        #     "loss_mask": smooth_l1_loss(src_masks[valid_mask], target_disps[valid_mask], num_masks)
        #     # "loss_dice": dice_loss_jit(point_logits, point_labels, num_masks),
        # }

        # del src_masks
        # del src_logits
        # # del src_classes
        # del classes
        # # del disp_masks
        # # del target_masks
        # del target_disps
        # return losses


def batch_sigmoid_ce_loss(inputs: torch.Tensor, targets: torch.Tensor):
    """
    Args:
        inputs: A float tensor of arbitrary shape.
                The predictions for each example.
        targets: A float tensor with the same shape as inputs. Stores the binary
                 classification label for each element in inputs
                (0 for the negative class and 1 for the positive class).
    Returns:
        Loss tensor
    """
    hw = inputs.shape[1]

    pos = F.binary_cross_entropy_with_logits(
        inputs, torch.ones_like(inputs), reduction="none"
    )
    neg = F.binary_cross_entropy_with_logits(
        inputs, torch.zeros_like(inputs), reduction="none"
    )

    loss = torch.einsum("nc,mc->nm", pos, targets) + torch.einsum(
        "nc,mc->nm", neg, (1 - targets)
    )

    return loss / hw


batch_sigmoid_ce_loss_jit = torch.jit.script(
    batch_sigmoid_ce_loss
)  # type: torch.jit.ScriptModule



class FixedMatcher(nn.Module):
    """This class assigns object query masks to ground truth according to class index
    """

    def __init__(self):
        """Creates the matcher"""
        super().__init__()

    @torch.no_grad()
    def memory_efficient_forward(self, outputs, targets):
        """More memory-friendly matching"""
        bs, num_queries = outputs["pred_logits"].shape[:2]

        indices = []

        # Iterate through batch size
        for b in range(bs):

            tgt_ids = targets[b]["labels"]
            # tgt_id_list = [i-1 for i in tgt_ids]
            b_ind = (tgt_ids, torch.arange(0, len(tgt_ids)))
            indices.append(b_ind)

        return [
            (torch.as_tensor(i, dtype=torch.int64), torch.as_tensor(j, dtype=torch.int64))
            for i, j in indices
        ]

    @torch.no_grad()
    def forward(self, outputs, targets):
        """Performs the matching

        Params:
            outputs: This is a dict that contains at least these entries:
                 "pred_logits": Tensor of dim [batch_size, num_queries, num_classes] with the classification logits
                 "pred_masks": Tensor of dim [batch_size, num_queries, H_pred, W_pred] with the predicted masks

            targets: This is a list of targets (len(targets) = batch_size), where each target is a dict containing:
                 "labels": Tensor of dim [num_target_boxes] (where num_target_boxes is the number of ground-truth
                           objects in the target) containing the class labels
                 "masks": Tensor of dim [num_target_boxes, H_gt, W_gt] containing the target masks

        Returns:
            A list of size batch_size, containing tuples of (index_i, index_j) where:
                - index_i is the indices of the selected predictions (in order)
                - index_j is the indices of the corresponding selected targets (in order)
            For each batch element, it holds:
                len(index_i) = len(index_j) = min(num_queries, num_target_boxes)
        """
        return self.memory_efficient_forward(outputs, targets)

    def __repr__(self, _repr_indent=4):
        head = ["Matcher " + self.__class__.__name__]

        return "\n".join(head)

class UpsampleMasks(nn.Module):

    def __init__(self, conv_dim=192):
        super().__init__()
        self.conv_dim = conv_dim
        self.conv2d = Conv2d(
            conv_dim,
            conv_dim,
            kernel_size=3,
            stride=1,
            padding=1,
        )
        weight_init.c2_xavier_fill(self.conv2d)

    def forward(self, pred_masks, height, width):
        pred_masks = F.interpolate(
                pred_masks[None],
                size=(self.conv_dim, height, width),
                mode="trilinear",
                align_corners=False,
            ).squeeze(0)
        pred_masks = self.conv2d(pred_masks)
        return pred_masks

def calc_disp(masks):
    masks = masks.softmax(1)
    disp_levels = torch.tensor([range(0,masks.shape[1])]).to(masks)[:,:,None,None]
    return (masks * disp_levels).sum(1, keepdim=True)


class BasicBlock(nn.Module):
    def __init__(self, in_channel, out_channel, stride, dilation):
        super().__init__()
        norm = nn.GroupNorm(32, out_channel)
        self.conv1 = Conv2d(
            in_channel, 
            out_channel,
            kernel_size=3, 
            stride=stride, 
            padding='same', 
            dilation=dilation, 
            norm=norm, 
            activation=F.relu
        )

        weight_init.c2_xavier_fill(self.conv1)

    def forward(self, x):
        out = self.conv1(x)
        out = x + out
        return out

class DispRefineLayer(nn.Module):
    def __init__(self, in_channel):
        super().__init__()

        norm = nn.GroupNorm(32, 256)
        self.conv2d_feature = Conv2d(
            in_channel, 
            256, 
            kernel_size=3, 
            stride=1, 
            padding=1, 
            norm=norm,
            activation=F.relu
        )
        self.residual_atrous_blocks = nn.ModuleList()
        atrous_list = [1, 2, 4, 8 , 1 , 1]
        for di in atrous_list:
            self.residual_atrous_blocks.append(BasicBlock(256, 256, stride=1, dilation=di))

                
        self.conv2d_out = Conv2d(256, 1, kernel_size=3, stride=1, padding=1)

        weight_init.c2_xavier_fill(self.conv2d_feature)
        weight_init.c2_xavier_fill(self.conv2d_out)

    def forward(self, features, disp):
        output = self.conv2d_feature(
            torch.cat([features, disp], dim=1))
        for atrous_block in self.residual_atrous_blocks:
            output = atrous_block(output)
        
        return F.relu(disp + self.conv2d_out(output))

def convbn_3d(in_planes, out_planes, kernel_size, stride, pad):

    return nn.Sequential(nn.Conv3d(in_planes, out_planes, kernel_size=kernel_size, padding=pad, stride=stride,bias=False),
                         nn.BatchNorm3d(out_planes))


class hourglass(nn.Module):
    def __init__(self, inplanes):
        super(hourglass, self).__init__()

        self.conv1 = nn.Sequential(convbn_3d(inplanes, inplanes*2, kernel_size=3, stride=(1,2,2), pad=1),
                                   nn.ReLU(inplace=True))

        self.conv2 = convbn_3d(inplanes*2, inplanes*2, kernel_size=3, stride=1, pad=1)

        self.conv3 = nn.Sequential(convbn_3d(inplanes*2, inplanes*2, kernel_size=3, stride=(1,2,2), pad=1),
                                   nn.ReLU(inplace=True))

        self.conv4 = nn.Sequential(convbn_3d(inplanes*2, inplanes*2, kernel_size=3, stride=1, pad=1),
                                   nn.ReLU(inplace=True))

        self.conv5 = nn.Sequential(nn.ConvTranspose3d(inplanes*2, inplanes*2, kernel_size=3, padding=1, output_padding=(0,1,1), stride=(1,2,2),bias=False),
                                   nn.BatchNorm3d(inplanes*2)) #+conv2

        self.conv6 = nn.Sequential(nn.ConvTranspose3d(inplanes*2, inplanes, kernel_size=3, padding=1, output_padding=(0,1,1), stride=(1,2,2),bias=False),
                                   nn.BatchNorm3d(inplanes)) #+x

    def forward(self, x ,presqu, postsqu):
        
        out  = self.conv1(x) #in:1/4 out:1/8
        pre  = self.conv2(out) #in:1/8 out:1/8
        if postsqu is not None:
           pre = F.relu(pre + postsqu, inplace=True)
        else:
           pre = F.relu(pre, inplace=True)

        out  = self.conv3(pre) #in:1/8 out:1/16
        out  = self.conv4(out) #in:1/16 out:1/16

        if presqu is not None:
           post = F.relu(self.conv5(out)+presqu, inplace=True) #in:1/16 out:1/8
        else:
           post = F.relu(self.conv5(out)+pre, inplace=True) 

        out  = self.conv6(post)  #in:1/8 out:1/4

        return out, pre, post

class PSMNet(nn.Module):
    def __init__(self, maxdisp, fine_disp_levels, coarse_disp_levels):
        super(PSMNet, self).__init__()
        self.maxdisp = maxdisp
        self.fine_disp_levels = fine_disp_levels
        self.coarse_disp_levels = coarse_disp_levels

        self.dres0 = nn.Sequential(convbn_3d(128, 64, 3, 1, 1),
                                     nn.ReLU(inplace=True),
                                     convbn_3d(64, 64, 3, 1, 1),
                                     nn.ReLU(inplace=True))

        self.dres1 = nn.Sequential(convbn_3d(64, 64, 3, 1, 1),
                                   nn.ReLU(inplace=True),
                                   convbn_3d(64, 64, 3, 1, 1)) 

        self.dres2 = hourglass(64)

        # self.dres3 = hourglass(32)

        # self.dres4 = hourglass(32)

        self.classif1 = nn.Sequential(convbn_3d(64, 32, 3, 1, 1),
                                      nn.ReLU(inplace=True),
                                      nn.Conv3d(32, 1, kernel_size=3, padding=1, stride=1,bias=False))

        # self.classif2 = nn.Sequential(convbn_3d(32, 32, 3, 1, 1),
        #                               nn.ReLU(inplace=True),
        #                               nn.Conv3d(32, 1, kernel_size=3, padding=1, stride=1,bias=False))

        # self.classif3 = nn.Sequential(convbn_3d(32, 32, 3, 1, 1),
        #                               nn.ReLU(inplace=True),
        #                               nn.Conv3d(32, 1, kernel_size=3, padding=1, stride=1,bias=False))

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
            elif isinstance(m, nn.Conv3d):
                n = m.kernel_size[0] * m.kernel_size[1]*m.kernel_size[2] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()
            elif isinstance(m, nn.BatchNorm3d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()
            elif isinstance(m, nn.Linear):
                m.bias.data.zero_()


    def forward_old(self, left_feat, right_feat, coarse_disp):

        #matching
        cost = torch.FloatTensor(left_feat.size()[0], left_feat.size()[1]*2, self.fine_disp_levels*2+1,  left_feat.size()[2],  left_feat.size()[3]).zero_().to(left_feat)
        y_idx, x_idx = torch.meshgrid(torch.arange(0,left_feat.shape[2]), torch.arange(0,left_feat.shape[3]))
        # coarse_disp = coarse_disp.expand(-1, refimg_fea.size()[1], -1, -1)

        # image is 1/4 resolution
        coarse_disp = coarse_disp/4.0

        disp_grid = torch.cat([(x_idx.expand(coarse_disp.shape[0],-1,-1).to(coarse_disp)-coarse_disp).unsqueeze(-1)/(left_feat.shape[3]-1)*2-1, (y_idx.expand(coarse_disp.shape[0],-1,-1).unsqueeze(-1).to(coarse_disp)/(left_feat.shape[2]-1))*2-1], dim=-1)

        for cost_idx, i in enumerate(range(-self.fine_disp_levels,self.fine_disp_levels+1)):
            i_norm = i / (4.0*left_feat.size()[3]) * 2
            if i != 0 :
                cost[:, :left_feat.size()[1], cost_idx, :, :] = F.grid_sample(right_feat, disp_grid - torch.tensor([i_norm,0]).to(disp_grid))
                cost[:, left_feat.size()[1]:, cost_idx, :, :] = left_feat
            else:
                cost[:, :left_feat.size()[1], cost_idx, :,:]   = F.grid_sample(right_feat, disp_grid)
                cost[:, left_feat.size()[1]:, cost_idx, :,:]   = left_feat

        cost = cost.contiguous()

        cost0 = self.dres0(cost)
        cost0 = self.dres1(cost0) + cost0

        out1, pre1, post1 = self.dres2(cost0, None, None)
        # out1, _, _ = self.dres2(cost0, None, None) 
        out1 = out1+cost0

        # out2, pre2, post2 = self.dres3(out1, pre1, post1) 
        # out2 = out2+cost0

        # out3, pre3, post3 = self.dres4(out2, pre1, post2) 
        # out3 = out3+cost0

        out1 = self.classif1(out1)
        # out1 = self.classif1(out3)
        # cost2 = self.classif2(out2) + cost1
        # cost3 = self.classif3(out3) + cost2

        out1 = out1.squeeze(1).softmax(1)
        disp_levels = torch.tensor([range(-self.fine_disp_levels,self.fine_disp_levels+1)]).to(out1)[:,:,None,None]
        out1 = (out1 * disp_levels).sum(1, keepdim=True)

        return out1 + coarse_disp.unsqueeze(1) * 4.0

    def forward(self, left_feat, right_feat, coarse_disp):

        coarse_disp = coarse_disp.softmax(1).argmax(1, keepdim=True).expand(-1, left_feat.shape[1], -1, -1)      
        
        coarse_shifted_feats = torch.Tensor(right_feat.shape).zero_().to(right_feat)

        for i in range(self.coarse_disp_levels):
            masked_feats = torch.zeros_like(right_feat)
            masked_feats[coarse_disp==i] = left_feat[coarse_disp==i]
            if i > 0 :
                coarse_shifted_feats[:, :, :, :-i] += masked_feats[:, :, :, i:]
            else:
                coarse_shifted_feats += masked_feats

        coarse_shifted_feats = coarse_shifted_feats.contiguous()

        coarse_shifted_feats = F.interpolate(
            coarse_shifted_feats,
            size=(coarse_shifted_feats.shape[-2]*4, coarse_shifted_feats.shape[-1]*4),
            mode="bilinear"
        )

        right_feat = F.interpolate(
            right_feat,
            size=(right_feat.shape[-2]*4, right_feat.shape[-1]*4),
            mode="bilinear"
        )

        cost = torch.FloatTensor(right_feat.size()[0], right_feat.size()[1]*2, self.fine_disp_levels*2+1,  right_feat.size()[2],  right_feat.size()[3]).zero_().to(right_feat)

        # y_idx, x_idx = torch.meshgrid(torch.arange(0,right_feat.shape[2]), torch.arange(0,right_feat.shape[3]))

        # disp_grid = torch.cat([(x_idx.expand(coarse_disp.shape[0],-1,-1)).unsqueeze(-1)/(left_feat.shape[3]-1)*2-1, (y_idx.expand(coarse_disp.shape[0],-1,-1).unsqueeze(-1)/(left_feat.shape[2]-1))*2-1], dim=-1)

        # for cost_idx, i in enumerate(range(-self.fine_disp_levels,self.fine_disp_levels+1)):
        #     i_norm = i / (4.0*left_feat.size()[3]) * 2
        #     if i != 0 :
        #         cost[:, :left_feat.size()[1], cost_idx, :, :] = F.grid_sample(coarse_shifted_feats, (disp_grid + torch.tensor([i_norm,0])).to(coarse_shifted_feats))
        #         cost[:, left_feat.size()[1]:, cost_idx, :, :] = right_feat
        #     else:
        #         cost[:, :left_feat.size()[1], cost_idx, :,:]   = coarse_shifted_feats
        #         cost[:, left_feat.size()[1]:, cost_idx, :,:]   = right_feat

        for cost_idx, i in enumerate(range(-self.fine_disp_levels,self.fine_disp_levels+1)):
            if i > 0 :
                cost[:, :left_feat.size()[1], cost_idx, :, i:] = coarse_shifted_feats[:,:,:,i:]
                cost[:, left_feat.size()[1]:, cost_idx, :, i:] = right_feat[:,:,:,:-i]
            elif i == 0:
                cost[:, :left_feat.size()[1], cost_idx, :,:]   = coarse_shifted_feats
                cost[:, left_feat.size()[1]:, cost_idx, :,:]   = right_feat
            elif i < 0 :
                cost[:, :left_feat.size()[1], cost_idx, :, :i] = coarse_shifted_feats[:,:,:,:i]
                cost[:, left_feat.size()[1]:, cost_idx, :, :i] = right_feat[:,:,:,-i:]

        cost = cost.contiguous()

        cost0 = self.dres0(cost)
        cost0 = self.dres1(cost0) + cost0

        # out1, pre1, post1 = self.dres2(cost0, None, None)
        out1, _, _ = self.dres2(cost0, None, None) 
        out1 = out1+cost0

        # out2, pre2, post2 = self.dres3(out1, pre1, post1) 
        # out2 = out2+cost0

        # out3, pre3, post3 = self.dres4(out2, pre1, post2) 
        # out3 = out3+cost0

        out1 = self.classif1(out1)
        # cost2 = self.classif2(out2) + cost1
        # cost3 = self.classif3(out3) + cost2

        out1 = out1.squeeze(1).softmax(1)
        disp_levels = torch.tensor([range(-self.fine_disp_levels,self.fine_disp_levels+1)]).to(out1)[:,:,None,None]
        out1 = (out1 * disp_levels).sum(1, keepdim=True)

        coarse_disp = F.interpolate(
            coarse_disp.float(),
            size=(coarse_disp.shape[-2]*4, coarse_disp.shape[-1]*4),
            mode="nearest"
        )

        return out1 + coarse_disp[:,0:1] * 4.0

class CosineSimMatching(nn.Module):
    def __init__(self, fine_disp_levels=4, resolution=1.0):
        super().__init__()
        self.fine_disp_levels = fine_disp_levels
        self.res = resolution
        self.similarity = torch.nn.CosineSimilarity(dim=2)

    def forward(self, left_feat, right_feat, coarse_disp):

        #debug 
        # left_feat = left_feat[0]
        # right_feat = right_feat[0]
        # features at 1/4 resolution
        coarse_disp = coarse_disp / 4.0
        y_idx, x_idx = torch.meshgrid(torch.arange(0,left_feat.shape[2]), torch.arange(0,left_feat.shape[3]))
        y_idx = y_idx.expand(coarse_disp.shape[0],-1,-1).unsqueeze(-1).to(coarse_disp)
        # shift x by coarse disparity estimate
        x_idx = (x_idx.expand(coarse_disp.shape[0],-1,-1).to(coarse_disp)-coarse_disp).unsqueeze(-1)

        # normalize between -1 to 1 for grid_sample
        x_idx = x_idx/(left_feat.shape[3]-1)*2-1
        y_idx = y_idx/(left_feat.shape[2]-1)*2-1
        disp_grid = torch.cat([x_idx, y_idx], dim=-1)

        cost = torch.FloatTensor(left_feat.size()[0], int((self.fine_disp_levels*2)/self.res+1), left_feat.size()[1], left_feat.size()[2],  left_feat.size()[3]).zero_().to(left_feat)

        # 1/4 resolution
        fine_levels = torch.arange(-self.fine_disp_levels,self.fine_disp_levels+self.res, self.res)/4.0
        

        for cost_idx, i in enumerate(fine_levels):
            # normalized -1 to 1
            i_norm = i / (left_feat.size()[3]) * 2
            if i != 0 :
                cost[:, cost_idx, :, :, :] = F.grid_sample(right_feat, disp_grid + torch.tensor([i_norm,0]).to(disp_grid))
            else:
                cost[:, cost_idx, :, :, :]   = F.grid_sample(right_feat, disp_grid)

        
        cos_sim = self.similarity(cost, left_feat.unsqueeze(1))
        max_cos_sim = cos_sim.argmax(dim=1)
        fine_levels = torch.ones_like(cos_sim) * fine_levels[None,:,None,None].to(cos_sim)
        fine_disp = fine_levels.gather(1, max_cos_sim.unsqueeze(1))


        return (coarse_disp.unsqueeze(1) + fine_disp) * 4.0

def merge_feature_levels(features):

    merged_size = (features['res2'].shape[-2], features['res2'].shape[-1])
    features_merged = torch.cat([
        features['res2'],
        F.interpolate(features['res3'], size=merged_size, mode="bilinear"),
        F.interpolate(features['res4'], size=merged_size, mode="bilinear")],
        # F.interpolate(features['res5'], size=merged_size, mode="bilinear"),
        dim=1
    )
    return features_merged

@META_ARCH_REGISTRY.register()
class MaskFormerStereo(nn.Module):
    """
    Main class for mask classification semantic segmentation architectures.
    """

    @configurable
    def __init__(
        self,
        *,
        backbone: Backbone,
        sem_seg_head: nn.Module,
        upsampler = nn.Module,
        fpn = nn.Module,
        refinement_layer = nn.Module,
        criterion: nn.Module,
        num_queries: int,
        object_mask_threshold: float,
        overlap_threshold: float,
        metadata,
        size_divisibility: int,
        sem_seg_postprocess_before_inference: bool,
        pixel_mean: Tuple[float],
        pixel_std: Tuple[float],
        # inference
        semantic_on: bool,
        panoptic_on: bool,
        instance_on: bool,
        test_topk_per_image: int,
    ):
        """
        Args:
            backbone: a backbone module, must follow detectron2's backbone interface
            sem_seg_head: a module that predicts semantic segmentation from backbone features
            criterion: a module that defines the loss
            num_queries: int, number of queries
            object_mask_threshold: float, threshold to filter query based on classification score
                for panoptic segmentation inference
            overlap_threshold: overlap threshold used in general inference for panoptic segmentation
            metadata: dataset meta, get `thing` and `stuff` category names for panoptic
                segmentation inference
            size_divisibility: Some backbones require the input height and width to be divisible by a
                specific integer. We can use this to override such requirement.
            sem_seg_postprocess_before_inference: whether to resize the prediction back
                to original input size before semantic segmentation inference or after.
                For high-resolution dataset like Mapillary, resizing predictions before
                inference will cause OOM error.
            pixel_mean, pixel_std: list or tuple with #channels element, representing
                the per-channel mean and std to be used to normalize the input image
            semantic_on: bool, whether to output semantic segmentation prediction
            instance_on: bool, whether to output instance segmentation prediction
            panoptic_on: bool, whether to output panoptic segmentation prediction
            test_topk_per_image: int, instance segmentation parameter, keep topk instances per image
        """
        super().__init__()
        self.backbone = backbone
        self.sem_seg_head = sem_seg_head
        self.refinement_layer = refinement_layer
        self.criterion = criterion
        self.upsampler = upsampler
        self.fpn = fpn
        self.num_queries = num_queries
        self.overlap_threshold = overlap_threshold
        self.object_mask_threshold = object_mask_threshold
        self.metadata = metadata
        if size_divisibility < 0:
            # use backbone size_divisibility if not set
            size_divisibility = self.backbone.size_divisibility
        self.size_divisibility = size_divisibility
        self.sem_seg_postprocess_before_inference = sem_seg_postprocess_before_inference
        self.register_buffer("pixel_mean", torch.Tensor(pixel_mean).view(-1, 1, 1), False)
        self.register_buffer("pixel_std", torch.Tensor(pixel_std).view(-1, 1, 1), False)

        # additional args
        self.semantic_on = semantic_on
        self.instance_on = instance_on
        self.panoptic_on = panoptic_on
        self.test_topk_per_image = test_topk_per_image

        if not self.semantic_on:
            assert self.sem_seg_postprocess_before_inference

    @classmethod
    def from_config(cls, cfg):
        backbone = build_backbone(cfg)

        # new seg head input with concatenate left and right backbone outputs
        seg_head_input_shape = {}
        for out_key in backbone.output_shape().keys():
            shape_spec = backbone.output_shape()[out_key]
            seg_head_input_shape[out_key] = ShapeSpec(
                channels=shape_spec.channels*2,
                height=shape_spec.height,
                width=shape_spec.width,
                stride=shape_spec.stride
            )

        sem_seg_head = build_sem_seg_head(cfg, seg_head_input_shape)

        # refinement_layer = CosineSimMatching(fine_disp_levels=4, resolution=0.25)
        refinement_layer = PSMNet(192, 4, 48)
        # refinement_layer = DispRefineLayer(in_channel=cfg.MODEL.SEM_SEG_HEAD.MASK_DIM+1)
        # refinement_layer = DispRefineLayer(in_channel=513)

        # Loss parameters:
        deep_supervision = cfg.MODEL.MASK_FORMER.DEEP_SUPERVISION
        no_object_weight = cfg.MODEL.MASK_FORMER.NO_OBJECT_WEIGHT

        # loss weights
        class_weight = cfg.MODEL.MASK_FORMER.CLASS_WEIGHT
        dice_weight = cfg.MODEL.MASK_FORMER.DICE_WEIGHT
        mask_weight = cfg.MODEL.MASK_FORMER.MASK_WEIGHT
        seg_weight = cfg.MODEL.MASK_FORMER.SEG_WEIGHT

        # building criterion
        matcher = FixedMatcher()
        # matcher = HungarianMatcher(
        #     cost_class=class_weight,
        #     cost_mask=mask_weight,
        #     cost_dice=dice_weight,
        #     num_points=cfg.MODEL.MASK_FORMER.TRAIN_NUM_POINTS,
        # )

        weight_dict = {"loss_mask": mask_weight, "loss_ce": class_weight, "loss_dice": dice_weight, "loss_seg": seg_weight}
        # weight_dict = {"loss_ce": 0, "loss_mask": 1.0}

        if deep_supervision:
            dec_layers = cfg.MODEL.MASK_FORMER.DEC_LAYERS
            aux_weight_dict = {}
            for i in range(dec_layers - 1):
                aux_weight_dict.update({k + f"_{i}": v for k, v in weight_dict.items() if k != 'loss_seg'})
            weight_dict.update(aux_weight_dict)

        losses = ["labels", "masks", "segs"]

        criterion = SetCriterionStereo(
            sem_seg_head.num_classes,
            matcher=matcher,
            weight_dict=weight_dict,
            eos_coef=no_object_weight,
            losses=losses,
            num_points=cfg.MODEL.MASK_FORMER.TRAIN_NUM_POINTS,
            oversample_ratio=cfg.MODEL.MASK_FORMER.OVERSAMPLE_RATIO,
            importance_sample_ratio=cfg.MODEL.MASK_FORMER.IMPORTANCE_SAMPLE_RATIO,
        )

        # upsampler = UpsampleMasks(
        #     num_queries=cfg.MODEL.MASK_FORMER.NUM_OBJECT_QUERIES,
        # )

        upsampler = None


        fpn_input_shape = {}
        for out_key in backbone.output_shape().keys():
            shape_spec = backbone.output_shape()[out_key]
            fpn_input_shape[out_key] = ShapeSpec(
                channels=shape_spec.channels,
                height=shape_spec.height,
                width=shape_spec.width,
                stride=shape_spec.stride
            )
        fpn = BasePixelDecoder(
            input_shape=fpn_input_shape,
            conv_dim=256,
            mask_dim=64,
            norm='GN'
        )

        # upsampler = UpsampleMasks(
        #     conv_dim=192,
        # )

        return {
            "backbone": backbone,
            "sem_seg_head": sem_seg_head,
            "refinement_layer": refinement_layer,
            "criterion": criterion,
            "upsampler": upsampler,
            "fpn": fpn,
            "num_queries": cfg.MODEL.MASK_FORMER.NUM_OBJECT_QUERIES,
            "object_mask_threshold": cfg.MODEL.MASK_FORMER.TEST.OBJECT_MASK_THRESHOLD,
            "overlap_threshold": cfg.MODEL.MASK_FORMER.TEST.OVERLAP_THRESHOLD,
            "metadata": MetadataCatalog.get(cfg.DATASETS.TRAIN[0]),
            "size_divisibility": cfg.MODEL.MASK_FORMER.SIZE_DIVISIBILITY,
            "sem_seg_postprocess_before_inference": (
                cfg.MODEL.MASK_FORMER.TEST.SEM_SEG_POSTPROCESSING_BEFORE_INFERENCE
                or cfg.MODEL.MASK_FORMER.TEST.PANOPTIC_ON
                or cfg.MODEL.MASK_FORMER.TEST.INSTANCE_ON
            ),
            "pixel_mean": cfg.MODEL.PIXEL_MEAN,
            "pixel_std": cfg.MODEL.PIXEL_STD,
            # inference
            "semantic_on": cfg.MODEL.MASK_FORMER.TEST.SEMANTIC_ON,
            "instance_on": cfg.MODEL.MASK_FORMER.TEST.INSTANCE_ON,
            "panoptic_on": cfg.MODEL.MASK_FORMER.TEST.PANOPTIC_ON,
            "test_topk_per_image": cfg.TEST.DETECTIONS_PER_IMAGE,
        }

    @property
    def device(self):
        return self.pixel_mean.device

    def forward(self, batched_inputs):
        """
        Args:
            batched_inputs: a list, batched outputs of :class:`DatasetMapper`.
                Each item in the list contains the inputs for one image.
                For now, each item in the list is a dict that contains:
                   * "image": Tensor, image in (C, H, W) format.
                   * "instances": per-region ground truth
                   * Other information that's included in the original dicts, such as:
                     "height", "width" (int): the output resolution of the model (may be different
                     from input resolution), used in inference.
        Returns:
            list[dict]:
                each dict has the results for one image. The dict contains the following keys:

                * "sem_seg":
                    A Tensor that represents the
                    per-pixel segmentation prediced by the head.
                    The prediction has shape KxHxW that represents the logits of
                    each class for each pixel.
                * "panoptic_seg":
                    A tuple that represent panoptic output
                    panoptic_seg (Tensor): of shape (height, width) where the values are ids for each segment.
                    segments_info (list[dict]): Describe each segment in `panoptic_seg`.
                        Each dict contains keys "id", "category_id", "isthing".
        """
        images_left = [x["image_left"].to(self.device) for x in batched_inputs]
        # images_left = [(x - x.float().mean(dim=[-2,-1],keepdim=True)) / x.float().std(dim=[-2,-1],keepdim=True) for x in images_left]
        images_left = [(x - self.pixel_mean) / self.pixel_std for x in images_left]
        images_left = ImageList.from_tensors(images_left, self.size_divisibility)

        images_right = [x["image_right"].to(self.device) for x in batched_inputs]
        # images_right = [(x - x.float().mean(dim=[-2,-1],keepdim=True)) / x.float().std(dim=[-2,-1],keepdim=True) for x in images_right]
        images_right = [(x - self.pixel_mean) / self.pixel_std for x in images_right]
        images_right = ImageList.from_tensors(images_right, self.size_divisibility)

        features_left = self.backbone(images_left.tensor)
        features_right = self.backbone(images_right.tensor)

        # concatenate left and right image features along channel dimension
        features = OrderedDict()
        for feat_key in features_left.keys():
            features[feat_key] = torch.cat([features_left[feat_key], features_right[feat_key]], dim=1)

        outputs = self.sem_seg_head(features)

        # coarse_disp = calc_disp(F.interpolate(outputs['pred_masks'].permute((0,2,1,3)), [192, outputs['pred_masks'].shape[-1]], mode='bilinear').permute((0,2,1,3)))

        # coarse_disp = calc_disp(outputs['pred_masks'])

        # # DEBUG
        # coarse_disp = torch.stack([t['sem_seg'] for t in batched_inputs]).unsqueeze(1)
        # coarse_disp = F.interpolate(
        #     coarse_disp,
        #     size=(features_left['res2'].shape[-2], features_left['res2'].shape[-1]),
        #     mode="bilinear",
        #     align_corners=False,
        # ) / 4.0

        # gt_seg = torch.stack([i['sem_seg'] for i in batched_inputs]).to(self.device)
        # with torch.no_grad():
        # outputs['pred_seg'] = self.refinement_layer(features_left['res2'], features_right['res2'], coarse_disp.squeeze(1))
        features_left, _, _ = self.fpn.forward_features(features_left)
        features_right, _, _ = self.fpn.forward_features(features_right)
        
        # outputs['pred_seg'] = self.refinement_layer(features_left, features_right, coarse_disp.squeeze(1))
        outputs['pred_seg'] = self.refinement_layer(features_left, features_right, outputs['pred_masks'])
            # DEBUG
            # outputs['pred_seg'] = self.refinement_layer(images_left, images_right, gt_seg)
        # outputs['pred_seg'] = coarse_disp + fine_disp * 4.0

        outputs['pred_seg'] = F.interpolate(
            outputs['pred_seg'],
            size=(images_left.tensor.shape[-2], images_left.tensor.shape[-1]),
            mode="bilinear",
            align_corners=False,
        )

        # #interpolate pred mask array
        # masks = F.interpolate(
        #     outputs['pred_masks'][None],
        #     size=(192, images_left.tensor.shape[-2], images_left.tensor.shape[-1]),
        #     mode="trilinear",
        #     align_corners=False,
        # ).squeeze(0)

        # # calculate seg masks
        # outputs['pred_seg'] = calc_disp(outputs['pred_masks'])  + fine_disp
        # outputs['pred_seg'] = self.refinement_layer(outputs['mask_features'], outputs['pred_seg'])
        # outputs['pred_seg'] = self.refinement_layer(features['res2'], outputs['pred_seg'])


        # if self.training:
        #     if 'aux_outputs' in outputs.keys():
        #         for i in range(len(outputs['aux_outputs'])):
        #             outputs['aux_outputs'][i]['pred_seg'] = calc_disp(outputs['aux_outputs'][i]['pred_masks'])
                    # outputs['aux_outputs'][i]['pred_seg'] = self.refinement_layer(
                    #     outputs['aux_outputs'][i]['mask_features'], 
                    #     outputs['aux_outputs'][i]['pred_seg']
                    # )

        

        # outputs['seg_masks'] = self.upsampler(
        #     outputs['pred_masks'],
        #     images_left.tensor.shape[-2],
        #     images_left.tensor.shape[-1]
        # )

        # if self.training:
        #     if 'aux_outputs' in outputs.keys():
        #         for i in range(len(outputs['aux_outputs'])):
        #             outputs['aux_outputs'][i]['seg_masks'] = self.upsampler(
        #                 outputs['aux_outputs'][i]['pred_masks'],
        #                 images_left.tensor.shape[-2],
        #                 images_left.tensor.shape[-1]
        #             )

        if self.training:
            # mask classification target
            if "instances" in batched_inputs[0]:
                gt_instances = [x["instances"].to(self.device) for x in batched_inputs]
                segs = [x["sem_seg"].to(self.device) for x in batched_inputs]
                targets = self.prepare_targets(gt_instances, segs, images_left)
            else:
                targets = None

            # bipartite matching-based loss
            losses = self.criterion(outputs, targets)

            for k in list(losses.keys()):
                if k in self.criterion.weight_dict:
                    losses[k] *= self.criterion.weight_dict[k]
                else:
                    # remove this loss if not specified in `weight_dict`
                    losses.pop(k)
            return losses
        else:
            mask_cls_results = outputs["pred_logits"]
            # mask_pred_results = outputs["pred_masks"]
            mask_pred_results = outputs["pred_seg"]
            # upsample masks
            # mask_pred_results = F.interpolate(
            #     mask_pred_results[None],
            #     size=(192,images_left.tensor.shape[-2], images_left.tensor.shape[-1]),
            #     mode="trilinear",
            #     align_corners=False,
            # ).squeeze(0)
            # mask_pred_results = F.interpolate(
            #     # mask_pred_results[:,None, ...],
            #     mask_pred_results,
            #     size=(images_left.tensor.shape[-2], images_left.tensor.shape[-1]),
            #     mode="bilinear",
            #     align_corners=False,
            # )

            del outputs

            processed_results = []
            for mask_cls_result, mask_pred_result, input_per_image, image_size in zip(
                mask_cls_results, mask_pred_results, batched_inputs, images_left.image_sizes
            ):
                height = input_per_image.get("height", image_size[0])
                width = input_per_image.get("width", image_size[1])
                processed_results.append({})

                if self.sem_seg_postprocess_before_inference:
                    mask_pred_result = retry_if_cuda_oom(sem_seg_postprocess)(
                        mask_pred_result, image_size, height, width
                    )
                    mask_cls_result = mask_cls_result.to(mask_pred_result)

                # semantic segmentation inference
                if self.semantic_on:
                    r = retry_if_cuda_oom(self.semantic_inference)(mask_cls_result, mask_pred_result)
                    if not self.sem_seg_postprocess_before_inference:
                        r = retry_if_cuda_oom(sem_seg_postprocess)(r, image_size, height, width)
                    processed_results[-1]["sem_seg"] = r

                # panoptic segmentation inference
                if self.panoptic_on:
                    panoptic_r = retry_if_cuda_oom(self.panoptic_inference)(mask_cls_result, mask_pred_result)
                    processed_results[-1]["panoptic_seg"] = panoptic_r
                
                # instance segmentation inference
                if self.instance_on:
                    instance_r = retry_if_cuda_oom(self.instance_inference)(mask_cls_result, mask_pred_result)
                    processed_results[-1]["instances"] = instance_r

            return processed_results

    def prepare_targets(self, targets, segs, images):
        h_pad, w_pad = images.tensor.shape[-2:]
        new_targets = []
        for targets_per_image, seg in zip(targets, segs):
            # pad gt
            gt_masks = targets_per_image.gt_masks
            padded_masks = torch.zeros((gt_masks.shape[0], h_pad, w_pad), dtype=gt_masks.dtype, device=gt_masks.device)
            padded_masks[:, : gt_masks.shape[1], : gt_masks.shape[2]] = gt_masks
            new_targets.append(
                {
                    "labels": targets_per_image.gt_classes,
                    "masks": padded_masks,
                    "sem_seg": seg
                }
            )
        return new_targets

    def semantic_inference(self, mask_cls, mask_pred):
        # mask_cls = F.softmax(mask_cls, dim=-1)[..., 1:]
        # mask_pred = mask_pred.sigmoid()
        # mask_pred = F.interpolate(mask_pred.permute((1,0,2))[None], [192, mask_pred.shape[-1]], mode='bilinear')
        # mask_pred = mask_pred.squeeze().permute((1,0,2)).softmax(0)
        # # mask_pred = mask_pred.softmax(0)
        # # mask_pred = F.interpolate(mask_pred.unsqueeze(0).permute((0,2,1,3)), [192, mask_pred.shape[-1]], mode='bilinear').squeeze().softmax(0)
        # # argsoftmax sum class likelihoods multiplied by disparities (weighted average of disparity)
        # # semseg = torch.einsum("qc,qhw->chw", mask_cls, mask_pred) #* torch.tensor([range(1,mask_cls.shape[-1]+1)])[0,:,None,None].to(mask_cls)).sum(0)
        # classes = torch.tensor([range(1,mask_pred.shape[0]+1)]).to(mask_pred).squeeze()[:,None,None]
        # # mask_cls = (mask_cls*classes).sum(-1)[..., None, None]
        # # semseg = (mask_pred * mask_cls).sum(0)
        # # semseg = (semseg * classes).sum(0)
        # semseg = (mask_pred * classes).sum(0)
        return F.relu(mask_pred)

    def panoptic_inference(self, mask_cls, mask_pred):
        scores, labels = F.softmax(mask_cls, dim=-1).max(-1)
        mask_pred = mask_pred.sigmoid()

        keep = labels.ne(self.sem_seg_head.num_classes) & (scores > self.object_mask_threshold)
        cur_scores = scores[keep]
        cur_classes = labels[keep]
        cur_masks = mask_pred[keep]
        cur_mask_cls = mask_cls[keep]
        cur_mask_cls = cur_mask_cls[:, :-1]

        cur_prob_masks = cur_scores.view(-1, 1, 1) * cur_masks

        h, w = cur_masks.shape[-2:]
        panoptic_seg = torch.zeros((h, w), dtype=torch.int32, device=cur_masks.device)
        segments_info = []

        current_segment_id = 0

        if cur_masks.shape[0] == 0:
            # We didn't detect any mask :(
            return panoptic_seg, segments_info
        else:
            # take argmax
            cur_mask_ids = cur_prob_masks.argmax(0)
            stuff_memory_list = {}
            for k in range(cur_classes.shape[0]):
                pred_class = cur_classes[k].item()
                isthing = pred_class in self.metadata.thing_dataset_id_to_contiguous_id.values()
                mask_area = (cur_mask_ids == k).sum().item()
                original_area = (cur_masks[k] >= 0.5).sum().item()
                mask = (cur_mask_ids == k) & (cur_masks[k] >= 0.5)

                if mask_area > 0 and original_area > 0 and mask.sum().item() > 0:
                    if mask_area / original_area < self.overlap_threshold:
                        continue

                    # merge stuff regions
                    if not isthing:
                        if int(pred_class) in stuff_memory_list.keys():
                            panoptic_seg[mask] = stuff_memory_list[int(pred_class)]
                            continue
                        else:
                            stuff_memory_list[int(pred_class)] = current_segment_id + 1

                    current_segment_id += 1
                    panoptic_seg[mask] = current_segment_id

                    segments_info.append(
                        {
                            "id": current_segment_id,
                            "isthing": bool(isthing),
                            "category_id": int(pred_class),
                        }
                    )

            return panoptic_seg, segments_info

    def instance_inference(self, mask_cls, mask_pred):
        # mask_pred is already processed to have the same shape as original input
        image_size = mask_pred.shape[-2:]

        # [Q, K]
        scores = F.softmax(mask_cls, dim=-1)[:, :-1]
        labels = torch.arange(self.sem_seg_head.num_classes, device=self.device).unsqueeze(0).repeat(self.num_queries, 1).flatten(0, 1)
        # scores_per_image, topk_indices = scores.flatten(0, 1).topk(self.num_queries, sorted=False)
        scores_per_image, topk_indices = scores.flatten(0, 1).topk(self.test_topk_per_image, sorted=False)
        labels_per_image = labels[topk_indices]

        topk_indices = topk_indices // self.sem_seg_head.num_classes
        # mask_pred = mask_pred.unsqueeze(1).repeat(1, self.sem_seg_head.num_classes, 1).flatten(0, 1)
        mask_pred = mask_pred[topk_indices]

        # if this is panoptic segmentation, we only keep the "thing" classes
        if self.panoptic_on:
            keep = torch.zeros_like(scores_per_image).bool()
            for i, lab in enumerate(labels_per_image):
                keep[i] = lab in self.metadata.thing_dataset_id_to_contiguous_id.values()

            scores_per_image = scores_per_image[keep]
            labels_per_image = labels_per_image[keep]
            mask_pred = mask_pred[keep]

        result = Instances(image_size)
        # mask (before sigmoid)
        result.pred_masks = (mask_pred > 0).float()
        result.pred_boxes = Boxes(torch.zeros(mask_pred.size(0), 4))
        # Uncomment the following to get boxes from masks (this is slow)
        # result.pred_boxes = BitMasks(mask_pred > 0).get_bounding_boxes()

        # calculate average mask prob
        mask_scores_per_image = (mask_pred.sigmoid().flatten(1) * result.pred_masks.flatten(1)).sum(1) / (result.pred_masks.flatten(1).sum(1) + 1e-6)
        result.scores = scores_per_image * mask_scores_per_image
        result.pred_classes = labels_per_image
        return result
