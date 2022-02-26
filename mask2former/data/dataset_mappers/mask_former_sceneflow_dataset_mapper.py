# Copyright (c) Facebook, Inc. and its affiliates.
import copy
import logging

import numpy as np
from PIL import Image
import re
import torch
from torch.nn import functional as F

from typing import Any, List, Optional, Tuple, Union
from fvcore.transforms.transform import Transform, TransformList

from detectron2.config import configurable
from detectron2.data import MetadataCatalog
from detectron2.data import detection_utils as utils
from detectron2.data import transforms as T
from detectron2.data.transforms.augmentation import Augmentation, AugmentationList
from detectron2.structures import BitMasks, Instances

__all__ = ["MaskFormerSceneFlowDatasetMapper"]


class MaskFormerSceneFlowDatasetMapper:
    """
    A callable which takes a dataset dict in Detectron2 Dataset format,
    and map it into a format used by MaskFormer for semantic segmentation.

    The callable currently does the following:

    1. Read the image from "file_name"
    2. Applies geometric transforms to the image and annotation
    3. Find and applies suitable cropping to the image and annotation
    4. Prepare image and annotation to Tensors
    """

    @configurable
    def __init__(
        self,
        is_train=True,
        *,
        augmentations,
        image_format,
        ignore_label,
        size_divisibility,
        num_classes
    ):
        """
        NOTE: this interface is experimental.
        Args:
            is_train: for training or inference
            load_disp_func: function to load disparity map
            augmentations: a list of augmentations or deterministic transforms to apply
            image_format: an image format supported by :func:`detection_utils.read_image`.
            ignore_label: the label that is ignored to evaluation
            size_divisibility: pad image size to be divisible by this value
        """
        self.is_train = is_train
        self.tfm_gens = augmentations
        self.img_format = image_format
        self.ignore_label = ignore_label
        self.size_divisibility = size_divisibility
        self.num_classes = num_classes

        logger = logging.getLogger(__name__)
        mode = "training" if is_train else "inference"
        logger.info(f"[{self.__class__.__name__}] Augmentations used in {mode}: {augmentations}")

    @classmethod
    def from_config(cls, cfg, is_train=True):
        # Build augmentation
        augs = []
        if is_train:
            # augs.append(
            #     T.ResizeShortestEdge(
            #         cfg.INPUT.MIN_SIZE_TRAIN,
            #         cfg.INPUT.MAX_SIZE_TRAIN,
            #         cfg.INPUT.MIN_SIZE_TRAIN_SAMPLING,
            #     )
            # )

            if cfg.INPUT.CROP.ENABLED:
                augs.append(
                    T.RandomCrop_CategoryAreaConstraint(
                        cfg.INPUT.CROP.TYPE,
                        cfg.INPUT.CROP.SIZE,
                        cfg.INPUT.CROP.SINGLE_CATEGORY_MAX_AREA,
                        cfg.MODEL.SEM_SEG_HEAD.IGNORE_VALUE,
                    )
                )
        
        #augs.append(T.RandomFlip())

        dataset_names = cfg.DATASETS.TRAIN
        meta = MetadataCatalog.get(dataset_names[0])
        ignore_label = meta.ignore_label
        num_classes = len(meta.stuff_classes)

        ret = {
            "is_train": is_train,
            "augmentations": augs,
            "image_format": cfg.INPUT.FORMAT,
            "ignore_label": ignore_label,
            "size_divisibility": cfg.INPUT.SIZE_DIVISIBILITY,
            "num_classes": num_classes
        }
        return ret

    def __call__(self, dataset_dict):
        """
        Args:
            dataset_dict (dict): Metadata of one image, in Detectron2 Dataset format.

        Returns:
            dict: a format that builtin models in detectron2 accept
        """
        # assert self.is_train, "MaskFormerSemanticDatasetMapper should only be used for training!"

        dataset_dict = copy.deepcopy(dataset_dict)  # it will be modified by code below
        image_left = utils.read_image(dataset_dict["file_name"], format=self.img_format)
        image_right = utils.read_image(dataset_dict["file_name_right"], format=self.img_format)
        #utils.check_image_size(dataset_dict, image)

        if "sem_seg_file_name" in dataset_dict:
            # PyTorch transformation not implemented for uint16, so converting it to double first
            sem_seg_gt = read_disparity(dataset_dict.pop("sem_seg_file_name"))
            if len(sem_seg_gt.shape)==3:
                sem_seg_gt = sem_seg_gt[:,:,0:3]
            sem_seg_gt = np.array(sem_seg_gt, dtype=np.float) #np.array(sem_seg_gt.round(), dtype=np.uint8) #
            sem_seg_gt[sem_seg_gt > 192] = 0
        else:
            sem_seg_gt = None

        if sem_seg_gt is None:
            raise ValueError(
                "Cannot find 'sem_seg_file_name' for semantic segmentation dataset {}.".format(
                    dataset_dict["file_name"]
                )
            )

        if self.is_train:
            aug_input = AugStereoInput(image_left, image_right, sem_seg=sem_seg_gt)
            aug_input, transforms = T.apply_transform_gens(self.tfm_gens, aug_input)
            image_left = aug_input.image
            image_right = aug_input.image_right
            sem_seg_gt = aug_input.sem_seg

        # Pad image and segmentation label here!
        image_left = torch.as_tensor(np.ascontiguousarray(image_left.transpose(2, 0, 1)))
        image_right = torch.as_tensor(np.ascontiguousarray(image_right.transpose(2, 0, 1)))
        if sem_seg_gt is not None:
            sem_seg_gt = torch.as_tensor(sem_seg_gt) #.astype("long"))

        if self.size_divisibility > 0:
            image_size = (image_left.shape[-2], image_left.shape[-1])
            padding_size = [
                0,
                self.size_divisibility - image_size[1],
                0,
                self.size_divisibility - image_size[0],
            ]
            image_left = F.pad(image_left, padding_size, value=128).contiguous()
            image_right = F.pad(image_right, padding_size, value=128).contiguous()
            if sem_seg_gt is not None:
                sem_seg_gt = F.pad(sem_seg_gt, padding_size, value=self.ignore_label).contiguous()

        image_shape = (image_left.shape[-2], image_left.shape[-1])  # h, w

        # Pytorch's dataloader is efficient on torch.Tensor due to shared-memory,
        # but not efficient on large generic data structures due to the use of pickle & mp.Queue.
        # Therefore it's important to use torch.Tensor.
        dataset_dict["image_left"] = image_left
        dataset_dict["image_right"] = image_right

        if sem_seg_gt is not None:
            dataset_dict["sem_seg"] = sem_seg_gt.float()

        if "annotations" in dataset_dict:
            raise ValueError("Semantic segmentation dataset should not have 'annotations'.")

        # Prepare per-category binary masks
        if sem_seg_gt is not None:
            sem_seg_gt = sem_seg_gt.numpy()
            sem_seg_gt = np.array(np.ceil(sem_seg_gt / 4.0), dtype=np.uint8)
            sem_seg_gt[sem_seg_gt > self.num_classes] = 0
            instances = Instances(image_shape)
            classes = np.unique(sem_seg_gt)
            # remove ignored region
            classes = classes[classes != self.ignore_label]
            instances.gt_classes = torch.tensor(classes, dtype=torch.int64)

            masks = []
            for class_id in classes:
                masks.append(sem_seg_gt == class_id)

            if len(masks) == 0:
                # Some image does not have annotation (all ignored)
                instances.gt_masks = torch.zeros((0, sem_seg_gt.shape[-2], sem_seg_gt.shape[-1]))
            else:
                masks = BitMasks(
                    torch.stack([torch.from_numpy(np.ascontiguousarray(x.copy())) for x in masks])
                )
                instances.gt_masks = masks.tensor

            dataset_dict["instances"] = instances

        return dataset_dict


class AugStereoInput:
    """
    Input that can be used with :meth:`Augmentation.__call__`.
    This is a standard implementation for the majority of use cases.
    This class provides the standard attributes **"image", "boxes", "sem_seg"**
    defined in :meth:`__init__` and they may be needed by different augmentations.
    Most augmentation policies do not need attributes beyond these three.

    After applying augmentations to these attributes (using :meth:`AugInput.transform`),
    the returned transforms can then be used to transform other data structures that users have.

    Examples:
    ::
        input = AugInput(image, boxes=boxes)
        tfms = augmentation(input)
        transformed_image = input.image
        transformed_boxes = input.boxes
        transformed_other_data = tfms.apply_other(other_data)

    An extended project that works with new data types may implement augmentation policies
    that need other inputs. An algorithm may need to transform inputs in a way different
    from the standard approach defined in this class. In those rare situations, users can
    implement a class similar to this class, that satify the following condition:

    * The input must provide access to these data in the form of attribute access
      (``getattr``).  For example, if an :class:`Augmentation` to be applied needs "image"
      and "sem_seg" arguments, its input must have the attribute "image" and "sem_seg".
    * The input must have a ``transform(tfm: Transform) -> None`` method which
      in-place transforms all its attributes.
    """

    # TODO maybe should support more builtin data types here
    def __init__(
        self,
        image: np.ndarray,
        image_right: np.ndarray,
        *,
        boxes: Optional[np.ndarray] = None,
        sem_seg: Optional[np.ndarray] = None,
    ):
        """
        Args:
            image (ndarray): (H,W) or (H,W,C) ndarray of type uint8 in range [0, 255], or
                floating point in range [0, 1] or [0, 255]. The meaning of C is up
                to users.
            image_right (ndarray): (H,W) or (H,W,C) ndarray of type uint8 in range [0, 255], or
                floating point in range [0, 1] or [0, 255]. The meaning of C is up
                to users.
            boxes (ndarray or None): Nx4 float32 boxes in XYXY_ABS mode
            sem_seg (ndarray or None): HxW uint8 semantic segmentation mask. Each element
                is an integer label of pixel.
        """
        _check_img_dtype(image)
        _check_img_dtype(image_right)
        self.image = image
        self.image_right = image_right
        self.boxes = boxes
        self.sem_seg = sem_seg

    def transform(self, tfm: Transform) -> None:
        """
        In-place transform all attributes of this class.

        By "in-place", it means after calling this method, accessing an attribute such
        as ``self.image`` will return transformed data.
        """
        self.image = tfm.apply_image(self.image)
        self.image_right = tfm.apply_image(self.image_right)
        if self.boxes is not None:
            self.boxes = tfm.apply_box(self.boxes)
        if self.sem_seg is not None:
            self.sem_seg = tfm.apply_segmentation(self.sem_seg)

    def apply_augmentations(
        self, augmentations: List[Union[Augmentation, Transform]]
    ) -> TransformList:
        """
        Equivalent of ``AugmentationList(augmentations)(self)``
        """
        return AugmentationList(augmentations)(self)

def _check_img_dtype(img):
    assert isinstance(img, np.ndarray), "[Augmentation] Needs an numpy array, but got a {}!".format(
        type(img)
    )
    assert not isinstance(img.dtype, np.integer) or (
        img.dtype == np.uint8
    ), "[Augmentation] Got image of type {}, use uint8 or floating points instead!".format(
        img.dtype
    )
    assert img.ndim in [2, 3], img.ndim

def readPFM(file):
    file = open(file, 'rb')

    color = None
    width = None
    height = None
    scale = None
    endian = None

    header = file.readline().rstrip()
    if header.decode("ascii") == 'PF':
        color = True
    elif header.decode("ascii") == 'Pf':
        color = False
    else:
        raise Exception('Not a PFM file.')

    dim_match = re.match(r'^(\d+)\s(\d+)\s$', file.readline().decode("ascii"))
    if dim_match:
        width, height = list(map(int, dim_match.groups()))
    else:
        raise Exception('Malformed PFM header.')

    scale = float(file.readline().decode("ascii").rstrip())
    if scale < 0: # little-endian
        endian = '<'
        scale = -scale
    else:
        endian = '>' # big-endian

    data = np.fromfile(file, endian + 'f')
    shape = (height, width, 3) if color else (height, width)

    data = np.reshape(data, shape)
    data = np.flipud(data)
    return data, scale

def read_disparity(file):
    if file.endswith('.png'): 
        img = Image.open(file)
        img = np.asarray(img)/256.0
        return img
    elif file.endswith('.pfm'): return readPFM(file)[0]
    else: raise Exception('don\'t know how to read %s' % file)