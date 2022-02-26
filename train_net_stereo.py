# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
"""
MaskFormer Training Script.

This script is a simplified version of the training script in detectron2/tools.
"""
import copy
import datetime
import itertools
import json
import logging
import numpy as np
import os
import re
import time

import cv2

from typing import List, Union
from contextlib import ExitStack, contextmanager

from collections import OrderedDict, abc
from PIL import Image
from typing import Any, Dict, List, Set

import torch
from torch import nn

import detectron2.utils.comm as comm
from detectron2.checkpoint import DetectionCheckpointer
from detectron2.config import get_cfg
from detectron2.data import DatasetCatalog, MetadataCatalog, build_detection_train_loader, build_detection_test_loader
from detectron2.data.datasets.cityscapes import load_cityscapes_stereo
from detectron2.engine import (
    DefaultTrainer,
    default_argument_parser,
    default_setup,
    launch,
)

from detectron2.evaluation import (
    CityscapesInstanceEvaluator,
    CityscapesSemSegEvaluator,
    COCOEvaluator,
    COCOPanopticEvaluator,
    DatasetEvaluators,
    LVISEvaluator,
    SemSegEvaluator,
    verify_results,
)

from detectron2.evaluation import (
    DatasetEvaluator,
    print_csv_format,
    verify_results,
)
from detectron2.projects.deeplab import add_deeplab_config, build_lr_scheduler
from detectron2.solver.build import maybe_add_gradient_clipping
from detectron2.utils.logger import setup_logger, log_every_n_seconds
from detectron2.utils.file_io import PathManager
from detectron2.utils.comm import all_gather, is_main_process, synchronize, get_world_size

# MaskFormer
from mask2former import (
    COCOInstanceNewBaselineDatasetMapper,
    COCOPanopticNewBaselineDatasetMapper,
    InstanceSegEvaluator,
    MaskFormerInstanceDatasetMapper,
    MaskFormerPanopticDatasetMapper,
    MaskFormerSemanticDatasetMapper,
    SemanticSegmentorWithTTA,
    MaskFormerStereoDatasetMapper,
    MaskFormerSceneFlowDatasetMapper,
    add_maskformer2_config,
)


class Trainer(DefaultTrainer):
    """
    Extension of the Trainer class adapted to MaskFormer.
    """

    @classmethod
    def build_evaluator(cls, cfg, dataset_name, output_folder=None):
        """
        Create evaluator(s) for a given dataset.
        This uses the special metadata "evaluator_type" associated with each
        builtin dataset. For your own dataset, you can simply create an
        evaluator manually in your script and do not have to worry about the
        hacky if-else logic here.
        """
        if output_folder is None:
            output_folder = os.path.join(cfg.OUTPUT_DIR, "inference")
        evaluator = SceneFlowStereoEvaluator(
                    dataset_name,
                    distributed=True,
                    output_dir=output_folder)
        return evaluator

    @classmethod
    def build_train_loader(cls, cfg):
        # Stereo segmentation dataset mapper
        if cfg.INPUT.DATASET_MAPPER_NAME == "mask_former_stereo":
            mapper = MaskFormerStereoDatasetMapper(cfg, True)
            return build_detection_train_loader(cfg, mapper=mapper)
        if cfg.INPUT.DATASET_MAPPER_NAME == "mask_former_sceneflow":
            mapper = MaskFormerSceneFlowDatasetMapper(cfg, True)
            return build_detection_train_loader(cfg, mapper=mapper)
        # Semantic segmentation dataset mapper
        if cfg.INPUT.DATASET_MAPPER_NAME == "mask_former_semantic":
            mapper = MaskFormerSemanticDatasetMapper(cfg, True)
            return build_detection_train_loader(cfg, mapper=mapper)
        # Panoptic segmentation dataset mapper
        elif cfg.INPUT.DATASET_MAPPER_NAME == "mask_former_panoptic":
            mapper = MaskFormerPanopticDatasetMapper(cfg, True)
            return build_detection_train_loader(cfg, mapper=mapper)
        # Instance segmentation dataset mapper
        elif cfg.INPUT.DATASET_MAPPER_NAME == "mask_former_instance":
            mapper = MaskFormerInstanceDatasetMapper(cfg, True)
            return build_detection_train_loader(cfg, mapper=mapper)
        # coco instance segmentation lsj new baseline
        elif cfg.INPUT.DATASET_MAPPER_NAME == "coco_instance_lsj":
            mapper = COCOInstanceNewBaselineDatasetMapper(cfg, True)
            return build_detection_train_loader(cfg, mapper=mapper)
        # coco panoptic segmentation lsj new baseline
        elif cfg.INPUT.DATASET_MAPPER_NAME == "coco_panoptic_lsj":
            mapper = COCOPanopticNewBaselineDatasetMapper(cfg, True)
            return build_detection_train_loader(cfg, mapper=mapper)
        else:
            mapper = None
            return build_detection_train_loader(cfg, mapper=mapper)

    @classmethod
    def build_test_loader(cls, cfg, dataset_name):
        """
        Returns:
            iterable

        It now calls :func:`detectron2.data.build_detection_test_loader`.
        Overwrite it if you'd like a different data loader.
        """
        if cfg.INPUT.DATASET_MAPPER_NAME == "mask_former_stereo":
            mapper = MaskFormerStereoDatasetMapper(cfg, is_train=False)
            return build_detection_test_loader(cfg, dataset_name, mapper=mapper)
        elif cfg.INPUT.DATASET_MAPPER_NAME == "mask_former_sceneflow":
            mapper = MaskFormerSceneFlowDatasetMapper(cfg, is_train=False)
            return build_detection_test_loader(cfg, dataset_name, mapper=mapper)
        else:
            mapper = None
            return build_detection_test_loader(cfg, dataset_name, mapper=mapper)

    @classmethod
    def build_lr_scheduler(cls, cfg, optimizer):
        """
        It now calls :func:`detectron2.solver.build_lr_scheduler`.
        Overwrite it if you'd like a different scheduler.
        """
        return build_lr_scheduler(cfg, optimizer)

    @classmethod
    def build_optimizer(cls, cfg, model):
        weight_decay_norm = cfg.SOLVER.WEIGHT_DECAY_NORM
        weight_decay_embed = cfg.SOLVER.WEIGHT_DECAY_EMBED

        defaults = {}
        defaults["lr"] = cfg.SOLVER.BASE_LR
        defaults["weight_decay"] = cfg.SOLVER.WEIGHT_DECAY

        norm_module_types = (
            torch.nn.BatchNorm1d,
            torch.nn.BatchNorm2d,
            torch.nn.BatchNorm3d,
            torch.nn.SyncBatchNorm,
            # NaiveSyncBatchNorm inherits from BatchNorm2d
            torch.nn.GroupNorm,
            torch.nn.InstanceNorm1d,
            torch.nn.InstanceNorm2d,
            torch.nn.InstanceNorm3d,
            torch.nn.LayerNorm,
            torch.nn.LocalResponseNorm,
        )

        params: List[Dict[str, Any]] = []
        memo: Set[torch.nn.parameter.Parameter] = set()
        for module_name, module in model.named_modules():
            for module_param_name, value in module.named_parameters(recurse=False):
                if not value.requires_grad:
                    continue
                # Avoid duplicating parameters
                if value in memo:
                    continue
                memo.add(value)

                hyperparams = copy.copy(defaults)
                if "backbone" in module_name:
                    hyperparams["lr"] = hyperparams["lr"] * cfg.SOLVER.BACKBONE_MULTIPLIER
                if (
                    "relative_position_bias_table" in module_param_name
                    or "absolute_pos_embed" in module_param_name
                ):
                    print(module_param_name)
                    hyperparams["weight_decay"] = 0.0
                if isinstance(module, norm_module_types):
                    hyperparams["weight_decay"] = weight_decay_norm
                if isinstance(module, torch.nn.Embedding):
                    hyperparams["weight_decay"] = weight_decay_embed
                params.append({"params": [value], **hyperparams})

        def maybe_add_full_model_gradient_clipping(optim):
            # detectron2 doesn't have full model gradient clipping now
            clip_norm_val = cfg.SOLVER.CLIP_GRADIENTS.CLIP_VALUE
            enable = (
                cfg.SOLVER.CLIP_GRADIENTS.ENABLED
                and cfg.SOLVER.CLIP_GRADIENTS.CLIP_TYPE == "full_model"
                and clip_norm_val > 0.0
            )

            class FullModelGradientClippingOptimizer(optim):
                def step(self, closure=None):
                    all_params = itertools.chain(*[x["params"] for x in self.param_groups])
                    torch.nn.utils.clip_grad_norm_(all_params, clip_norm_val)
                    super().step(closure=closure)

            return FullModelGradientClippingOptimizer if enable else optim

        optimizer_type = cfg.SOLVER.OPTIMIZER
        if optimizer_type == "SGD":
            optimizer = maybe_add_full_model_gradient_clipping(torch.optim.SGD)(
                params, cfg.SOLVER.BASE_LR, momentum=cfg.SOLVER.MOMENTUM
            )
        elif optimizer_type == "ADAMW":
            optimizer = maybe_add_full_model_gradient_clipping(torch.optim.AdamW)(
                params, cfg.SOLVER.BASE_LR
            )
        else:
            raise NotImplementedError(f"no optimizer type {optimizer_type}")
        if not cfg.SOLVER.CLIP_GRADIENTS.CLIP_TYPE == "full_model":
            optimizer = maybe_add_gradient_clipping(cfg, optimizer)
        return optimizer

    @classmethod
    def test(cls, cfg, model, evaluators=None):
        """
        Evaluate the given model. The given model is expected to already contain
        weights to evaluate.

        Args:
            cfg (CfgNode):
            model (nn.Module):
            evaluators (list[DatasetEvaluator] or None): if None, will call
                :meth:`build_evaluator`. Otherwise, must have the same length as
                ``cfg.DATASETS.TEST``.

        Returns:
            dict: a dict of result metrics
        """
        logger = logging.getLogger(__name__)
        if isinstance(evaluators, DatasetEvaluator):
            evaluators = [evaluators]
        if evaluators is not None:
            assert len(cfg.DATASETS.TEST) == len(evaluators), "{} != {}".format(
                len(cfg.DATASETS.TEST), len(evaluators)
            )

        results = OrderedDict()
        for idx, dataset_name in enumerate(cfg.DATASETS.TEST):
            data_loader = cls.build_test_loader(cfg, dataset_name)
            # When evaluators are passed in as arguments,
            # implicitly assume that evaluators can be created before data_loader.
            if evaluators is not None:
                evaluator = evaluators[idx]
            else:
                try:
                    evaluator = cls.build_evaluator(cfg, dataset_name)
                except NotImplementedError:
                    logger.warn(
                        "No evaluator found. Use `DefaultTrainer.test(evaluators=)`, "
                        "or implement its `build_evaluator` method."
                    )
                    results[dataset_name] = {}
                    continue
            results_i = inference_on_dataset(model, data_loader, evaluator)
            results[dataset_name] = results_i
            if comm.is_main_process():
                assert isinstance(
                    results_i, dict
                ), "Evaluator must return a dict on the main process. Got {} instead.".format(
                    results_i
                )
                logger.info("Evaluation results for {} in csv format:".format(dataset_name))
                print_csv_format(results_i)

        if len(results) == 1:
            results = list(results.values())[0]
        return results

    @classmethod
    def test_with_TTA(cls, cfg, model):
        logger = logging.getLogger("detectron2.trainer")
        # In the end of training, run an evaluation with TTA.
        logger.info("Running inference with test-time augmentation ...")
        model = SemanticSegmentorWithTTA(cfg, model)
        evaluators = [
            cls.build_evaluator(
                cfg, name, output_folder=os.path.join(cfg.OUTPUT_DIR, "inference_TTA")
            )
            for name in cfg.DATASETS.TEST
        ]
        res = cls.test(cfg, model, evaluators)
        res = OrderedDict({k + "_TTA": v for k, v in res.items()})
        return res

def inference_on_dataset(
    model, data_loader, evaluator: Union[DatasetEvaluator, List[DatasetEvaluator], None]
):
    """
    Run model on the data_loader and evaluate the metrics with evaluator.
    Also benchmark the inference speed of `model.__call__` accurately.
    The model will be used in eval mode.

    Args:
        model (callable): a callable which takes an object from
            `data_loader` and returns some outputs.

            If it's an nn.Module, it will be temporarily set to `eval` mode.
            If you wish to evaluate a model in `training` mode instead, you can
            wrap the given model and override its behavior of `.eval()` and `.train()`.
        data_loader: an iterable object with a length.
            The elements it generates will be the inputs to the model.
        evaluator: the evaluator(s) to run. Use `None` if you only want to benchmark,
            but don't want to do any evaluation.

    Returns:
        The return value of `evaluator.evaluate()`
    """
    num_devices = get_world_size()
    logger = logging.getLogger(__name__)
    logger.info("Start inference on {} batches".format(len(data_loader)))

    total = len(data_loader)  # inference data loader must have a fixed length
    if evaluator is None:
        # create a no-op evaluator
        evaluator = DatasetEvaluators([])
    if isinstance(evaluator, abc.MutableSequence):
        evaluator = DatasetEvaluators(evaluator)
    evaluator.reset()

    num_warmup = min(5, total - 1)
    start_time = time.perf_counter()
    total_data_time = 0
    total_compute_time = 0
    total_eval_time = 0
    with ExitStack() as stack:
        if isinstance(model, nn.Module):
            stack.enter_context(inference_context(model))
        stack.enter_context(torch.no_grad())

        start_data_time = time.perf_counter()
        for idx, inputs in enumerate(data_loader):
            total_data_time += time.perf_counter() - start_data_time
            if idx == num_warmup:
                start_time = time.perf_counter()
                total_data_time = 0
                total_compute_time = 0
                total_eval_time = 0

            start_compute_time = time.perf_counter()
            outputs = model(inputs)
            if torch.cuda.is_available():
                torch.cuda.synchronize()
            total_compute_time += time.perf_counter() - start_compute_time

            start_eval_time = time.perf_counter()
            evaluator.process(inputs, outputs)
            total_eval_time += time.perf_counter() - start_eval_time

            iters_after_start = idx + 1 - num_warmup * int(idx >= num_warmup)
            data_seconds_per_iter = total_data_time / iters_after_start
            compute_seconds_per_iter = total_compute_time / iters_after_start
            eval_seconds_per_iter = total_eval_time / iters_after_start
            total_seconds_per_iter = (time.perf_counter() - start_time) / iters_after_start
            if idx >= num_warmup * 2 or compute_seconds_per_iter > 5:
                eta = datetime.timedelta(seconds=int(total_seconds_per_iter * (total - idx - 1)))
                log_every_n_seconds(
                    logging.INFO,
                    (
                        f"Inference done {idx + 1}/{total}. "
                        f"Dataloading: {data_seconds_per_iter:.4f} s/iter. "
                        f"Inference: {compute_seconds_per_iter:.4f} s/iter. "
                        f"Eval: {eval_seconds_per_iter:.4f} s/iter. "
                        f"Total: {total_seconds_per_iter:.4f} s/iter. "
                        f"ETA={eta}"
                    ),
                    n=5,
                    name='mask2former'
                )
            start_data_time = time.perf_counter()

    # Measure the time only for this worker (before the synchronization barrier)
    total_time = time.perf_counter() - start_time
    total_time_str = str(datetime.timedelta(seconds=total_time))
    # NOTE this format is parsed by grep
    logger.info(
        "Total inference time: {} ({:.6f} s / iter per device, on {} devices)".format(
            total_time_str, total_time / (total - num_warmup), num_devices
        )
    )
    total_compute_time_str = str(datetime.timedelta(seconds=int(total_compute_time)))
    logger.info(
        "Total inference pure compute time: {} ({:.6f} s / iter per device, on {} devices)".format(
            total_compute_time_str, total_compute_time / (total - num_warmup), num_devices
        )
    )
    evaluator.epe = evaluator.epe/len(data_loader)
    evaluator.one_pix_err = evaluator.one_pix_err/len(data_loader)
    evaluator.three_pix_err = evaluator.three_pix_err/len(data_loader)
    results = evaluator.evaluate()
    # An evaluator may return None when not in main process.
    # Replace it by an empty dict instead to make it easier for downstream code to handle
    if results is None:
        results = {}
    return results


@contextmanager
def inference_context(model):
    """
    A context where the model is temporarily changed to eval mode,
    and restored to previous mode afterwards.

    Args:
        model: a torch Module
    """
    training_mode = model.training
    model.eval()
    yield
    model.train(training_mode)

class SceneFlowStereoEvaluator(SemSegEvaluator):

    """
    Semantic seg evaluator with overwride of process method
    for extra preprocessing for stereo disparity maps
    """

    def reset(self):
        self._conf_matrix = np.zeros((self._num_classes + 1, self._num_classes + 1), dtype=np.int64)
        # self._conf_matrix = np.zeros((192 + 1, 192 + 1), dtype=np.int64)
        self._predictions = []
        self.epe = 0
        self.one_pix_err = 0
        self.three_pix_err = 0

    def process(self, inputs, outputs):
        """
        Args:
            inputs: the inputs to a model.
                It is a list of dicts. Each dict corresponds to an image and
                contains keys like "height", "width", "file_name".
            outputs: the outputs of a model. It is either list of semantic segmentation predictions
                (Tensor [H, W]) or list of dicts with key "sem_seg" that contains semantic
                segmentation prediction in the same format.
        """
        for input, output in zip(inputs, outputs):
            output = output["sem_seg"].to(self._cpu_device).squeeze() #* 4.0
            # output = output["sem_seg"].argmax(dim=0).to(self._cpu_device)+1
            # pred = np.array(output, dtype=np.int)[0]
            gt = read_disparity(self.input_file_to_gt_file[input["file_name"]])
            if len(gt.shape)==3:
                gt = gt[:,:,0:3]
            # gt = np.array(gt.round(), dtype=np.long)
            gt = np.array(gt, dtype=np.float) #/4.0
            # gt = np.ceil(np.array(gt, dtype=np.long))
            # gt = np.ceil(np.array(gt, dtype=np.long) / 4.0)
            # pred = np.array(output.round(), dtype=np.long) #* 4.0
            pred = np.array(output, dtype=np.float)

            # gt[gt > self._num_classes] = 0
            # pred[pred > self._num_classes] = 0

            gt[gt > 192] = 0
            pred[pred > 192] = 0

            valid_mask = gt > 0

            # gt[gt == self._ignore_label] = self._num_classes

            if np.any(valid_mask):
                disp_error = np.abs(pred[valid_mask]-gt[valid_mask])
                self.epe += np.mean(disp_error)
                self.one_pix_err += np.sum(disp_error > 1.0 ) / np.sum(valid_mask)
                self.three_pix_err += np.sum(disp_error > 3.0 ) / np.sum(valid_mask)

            pred = np.array(torch.round(output)/4, dtype=np.int)
            

            gt = np.ceil(gt/4.0)
            gt = gt.astype(np.int)
            # gt = np.array(gt.round(), dtype=np.int)


            gt[gt > self._num_classes] = 0
            pred[pred > self._num_classes] = 0

            # gt[gt > 192] = 0
            # pred[pred > 192] = 0

            valid_mask = gt > 0

            # gt[gt == self._ignore_label] = self._num_classes

            # if np.any(valid_mask):
            #     disp_error = np.abs(pred[valid_mask]-gt[valid_mask])
            #     self.epe += np.mean(disp_error)
            #     self.one_pix_err += np.sum(disp_error > 1.0 ) / np.sum(valid_mask)
            #     self.three_pix_err += np.sum(disp_error > 3.0 ) / np.sum(valid_mask)
                

            self._conf_matrix += np.bincount(
                (self._num_classes + 1) * pred.reshape(-1) + gt.reshape(-1),
                minlength=self._conf_matrix.size,
            ).reshape(self._conf_matrix.shape)

            # self._conf_matrix += np.bincount(
            #     (192 + 1) * pred.reshape(-1) + gt.reshape(-1),
            #     minlength=self._conf_matrix.size,
            # ).reshape(self._conf_matrix.shape)

            self._predictions.extend(self.encode_json_sem_seg(pred, input["file_name"])) 

    def evaluate(self):
        """
        Evaluates standard semantic segmentation metrics (http://cocodataset.org/#stuff-eval)
        and additional stereo metric EPE:

        * Mean intersection-over-union averaged across classes (mIoU)
        * Frequency Weighted IoU (fwIoU)
        * Mean pixel accuracy averaged across classes (mACC)
        * Pixel Accuracy (pACC)
        * End-to-End point error (epe)
        """
        if self._distributed:
            synchronize()
            conf_matrix_list = all_gather(self._conf_matrix)
            self._predictions = all_gather(self._predictions)
            epe_list = all_gather(self.epe)
            self.epe = np.mean(np.array(epe_list))
            one_pix_err_list = all_gather(self.one_pix_err)
            self.one_pix_err = np.mean(np.array(one_pix_err_list))
            three_pix_err_list = all_gather(self.three_pix_err)
            self.three_pix_err = np.mean(np.array(three_pix_err_list))
            self._predictions = list(itertools.chain(*self._predictions))
            if not is_main_process():
                return

            self._conf_matrix = np.zeros_like(self._conf_matrix)
            for conf_matrix in conf_matrix_list:
                self._conf_matrix += conf_matrix

        if self._output_dir:
            PathManager.mkdirs(self._output_dir)
            file_path = os.path.join(self._output_dir, "sem_seg_predictions.json")
            with PathManager.open(file_path, "w") as f:
                f.write(json.dumps(self._predictions))

        acc = np.full(self._num_classes, np.nan, dtype=np.float)
        iou = np.full(self._num_classes, np.nan, dtype=np.float)
        tp = self._conf_matrix.diagonal()[:-1].astype(np.float)
        pos_gt = np.sum(self._conf_matrix[:-1, :-1], axis=0).astype(np.float)
        class_weights = pos_gt / np.sum(pos_gt)
        pos_pred = np.sum(self._conf_matrix[:-1, :-1], axis=1).astype(np.float)
        acc_valid = pos_gt > 0
        acc[acc_valid] = tp[acc_valid] / pos_gt[acc_valid]
        iou_valid = (pos_gt + pos_pred) > 0
        union = pos_gt + pos_pred - tp
        iou[acc_valid] = tp[acc_valid] / union[acc_valid]
        macc = np.sum(acc[acc_valid]) / np.sum(acc_valid)
        miou = np.sum(iou[acc_valid]) / np.sum(iou_valid)
        fiou = np.sum(iou[acc_valid] * class_weights[acc_valid])
        pacc = np.sum(tp) / np.sum(pos_gt)

        res = {}
        res["epe"] = self.epe
        res["error_1pix"] = self.one_pix_err
        res["error_3pix"] = self.three_pix_err
        res["mIoU"] = 100 * miou
        res["fwIoU"] = 100 * fiou
        for i, name in enumerate(self._class_names):
            res["IoU-{}".format(name)] = 100 * iou[i]
        res["mACC"] = 100 * macc
        res["pACC"] = 100 * pacc
        for i, name in enumerate(self._class_names):
            res["ACC-{}".format(name)] = 100 * acc[i]

        if self._output_dir:
            file_path = os.path.join(self._output_dir, "sem_seg_evaluation.pth")
            with PathManager.open(file_path, "wb") as f:
                torch.save(res, f)
        results = OrderedDict({"sem_seg": res})
        self._logger.info(results)
        return results

CITYSCAPES_STEREO_SPLITS = {
    "cityscapes_stereo_train": ("cityscapes/leftImg8bit/train/", "cityscapes/rightImg8bit/train/", "cityscapes/disparity/train/"),
    "cityscapes_stereo_train_extra": ("cityscapes/leftImg8bit/train_extra/", "cityscapes/rightImg8bit/train_extra/", "cityscapes/disparity/train_extra/"),
    "cityscapes_stereo_val": ("cityscapes/leftImg8bit/val/", "cityscapes/rightImg8bit/val/", "cityscapes/disparity/val/"),
    "cityscapes_stereo_test": ("cityscapes/leftImg8bit/test/", "cityscapes/rightImg8bit/test/", "cityscapes/disparity/test/"),
}

def register_cityscapes_stereo(root):
    for key, (image_left_dir, image_right_dir, gt_dir) in CITYSCAPES_STEREO_SPLITS.items():
        image_left_dir = os.path.join(root, image_left_dir)
        image_right_dir = os.path.join(root, image_right_dir)
        gt_dir = os.path.join(root, gt_dir)

        # set stuff classes metadata
        stuff_classes = range(0,192,1)
        stuff_classes = [str(label) for label in stuff_classes]

        DatasetCatalog.register(
            key, lambda x=image_left_dir, y=gt_dir: load_cityscapes_stereo(x, y)
        )
        MetadataCatalog.get(key).set(
            image_left_dir=image_left_dir,
            image_right_dir=image_right_dir,
            gt_dir=gt_dir,
            ignore_label=0,
            stuff_classes=stuff_classes,
        )

SCENEFLOW_STEREO_SPLITS = {
    "sceneflow_train": "/home/Datasets/sceneflow/lists/sceneflow_train.list",
    "sceneflow_test": "/home/Datasets/sceneflow/lists/sceneflow_test.list",
}

def register_sceneflow_stereo(cfg):
    root = cfg.DATASETS.ROOT
    for key, file_list in SCENEFLOW_STEREO_SPLITS.items():
        image_dir = os.path.join(root, 'frames_finalpass')
        disp_dir = os.path.join(root, 'disparity')
        max_disp = cfg.MODEL.SEM_SEG_HEAD.NUM_CLASSES

        # set stuff classes metadata
        stuff_classes = range(1,max_disp+1,1)
        stuff_classes = [str(label) for label in stuff_classes]

        DatasetCatalog.register(
            key, lambda x=image_dir, y=disp_dir, z=file_list: load_sceneflow_stereo(x, y, z)
        )
        MetadataCatalog.get(key).set(
            image_dir=image_dir,
            gt_dir=disp_dir,
            ignore_label=0,
            stuff_classes=stuff_classes,
        )

def load_sceneflow_stereo(image_dir, disp_dir, file_list):
    """
    Args:
        image_dir (str): path to the raw dataset. e.g., "~/cityscapes/leftImg8bit/train".
        disp_dir (str): path to the raw disparity. e.g., "~/cityscapes/disparity/train".

    Returns:
        list[dict]: a list of dict, each has "file_name", "file_name_right" and
            "sem_seg_file_name".
    """
    ret = []
    with open(file_list, 'r') as f:
        img_list = f.read().splitlines()

    for image_file in img_list:
        left_image_file = os.path.join(image_dir, image_file)
        if not PathManager.isfile(left_image_file):
            continue
        right_image_file = left_image_file.replace('left', 'right')
        if not PathManager.isfile(right_image_file):
            continue
        disp_file = os.path.splitext(os.path.join(disp_dir, image_file))[0] + '.pfm'
        if not PathManager.isfile(disp_file):
            continue

        im = Image.open(left_image_file)
        (width, height) = im.size

        ret.append(
            {
                "file_name": left_image_file,
                "file_name_right": right_image_file,
                "sem_seg_file_name": disp_file,
                "width": width,
                "height": height
            }
        )
    assert len(ret), f"No images found in {image_dir}!"
    assert PathManager.isfile(
        ret[0]["sem_seg_file_name"]
    ), "Disparity map not found"  # noqa
    return ret


KITTI_STEREO_SPLITS = {
    "kitti_train": "/home/Datasets/kitti_stereo_2015/kitti2015_train180.list",
    "kitti_val": "/home/Datasets/kitti_stereo_2015/kitti2015_val20.list",
}

def register_kitti_stereo(cfg):
    root = cfg.DATASETS.ROOT
    for key, file_list in KITTI_STEREO_SPLITS.items():
        image_dir = os.path.join(root, 'image_2')
        disp_dir = os.path.join(root, 'disp_occ_0')
        max_disp = cfg.MODEL.SEM_SEG_HEAD.NUM_CLASSES

        # set stuff classes metadata
        stuff_classes = range(1,max_disp+1,1)
        stuff_classes = [str(label) for label in stuff_classes]

        DatasetCatalog.register(
            key, lambda x=image_dir, y=disp_dir, z=file_list: load_kitti_stereo(x, y, z)
        )
        MetadataCatalog.get(key).set(
            image_dir=image_dir,
            gt_dir=disp_dir,
            ignore_label=0,
            stuff_classes=stuff_classes,
        )

def load_kitti_stereo(image_dir, disp_dir, file_list):
    """
    Args:
        image_dir (str): path to the raw dataset. e.g., "~/cityscapes/leftImg8bit/train".
        disp_dir (str): path to the raw disparity. e.g., "~/cityscapes/disparity/train".

    Returns:
        list[dict]: a list of dict, each has "file_name", "file_name_right" and
            "sem_seg_file_name".
    """
    ret = []
    with open(file_list, 'r') as f:
        img_list = f.read().splitlines()

    for image_file in img_list:
        left_image_file = os.path.join(image_dir, image_file)
        if not PathManager.isfile(left_image_file):
            continue
        right_image_file = left_image_file.replace('image_2', 'image_3')
        if not PathManager.isfile(right_image_file):
            continue
        disp_file = os.path.join(disp_dir, image_file)
        if not PathManager.isfile(disp_file):
            continue

        im = Image.open(left_image_file)
        (width, height) = im.size

        ret.append(
            {
                "file_name": left_image_file,
                "file_name_right": right_image_file,
                "sem_seg_file_name": disp_file,
                "width": width,
                "height": height
            }
        )
    assert len(ret), f"No images found in {image_dir}!"
    assert PathManager.isfile(
        ret[0]["sem_seg_file_name"]
    ), "Disparity map not found"  # noqa
    return ret

# def register_sceneflow_stereo(cfg):
#     root = cfg.DATASETS.ROOT
#     for key, (image_dir, disp_dir) in SCENEFLOW_STEREO_SPLITS.items():
#         image_dir = os.path.join(root, image_dir)
#         disp_dir = os.path.join(root, disp_dir)
#         max_disp = cfg.MODEL.SEM_SEG_HEAD.NUM_CLASSES-1

#         # set stuff classes metadata
#         stuff_classes = range(0,max_disp+1,1)
#         stuff_classes = [str(label) for label in stuff_classes]

#         DatasetCatalog.register(
#             key, lambda x=image_dir, y=disp_dir: load_sceneflow_stereo(x, y)
#         )
#         MetadataCatalog.get(key).set(
#             image_dir=image_dir,
#             gt_dir=disp_dir,
#             ignore_label=0,
#             stuff_classes=stuff_classes,
#         )

# def load_sceneflow_stereo(image_dir, disp_dir):
#     """
#     Args:
#         image_dir (str): path to the raw dataset. e.g., "~/cityscapes/leftImg8bit/train".
#         disp_dir (str): path to the raw disparity. e.g., "~/cityscapes/disparity/train".

#     Returns:
#         list[dict]: a list of dict, each has "file_name", "file_name_right" and
#             "sem_seg_file_name".
#     """
#     ret = []
#     # gt_dir is small and contain many small files. make sense to fetch to local first
#     disp_dir = PathManager.get_local_path(disp_dir)
#     for left_image_file, right_image_file, disp_file in get_sceneflow_files(image_dir, disp_dir):
#         im = Image.open(left_image_file)
#         (width, height) = im.size
#         ret.append(
#             {
#                 "file_name": left_image_file,
#                 "file_name_right": right_image_file,
#                 "sem_seg_file_name": disp_file,
#                 "width": width,
#                 "height": height
#             }
#         )
#     assert len(ret), f"No images found in {image_dir}!"
#     assert PathManager.isfile(
#         ret[0]["sem_seg_file_name"]
#     ), "Disparity map not found"  # noqa
#     return ret

# def get_sceneflow_files(image_dir, disp_dir):
#     logger = logging.getLogger(__name__)
#     files = []
#     # scan through the directory
#     subsets = PathManager.ls(image_dir)
#     for sub in subsets:
#         sub_img_dir = os.path.join(image_dir, sub)
#         sub_disp_dir = os.path.join(disp_dir, sub)
#         for seq in PathManager.ls(sub_img_dir):
#             left_image_dir = os.path.join(sub_img_dir, seq, 'left')
#             for basename in PathManager.ls(left_image_dir):
#                 left_image_file = os.path.join(left_image_dir, basename)
#                 if not PathManager.isfile(left_image_file):
#                     continue
#                 right_image_file = left_image_file.replace('left', 'right')
#                 if not PathManager.isfile(right_image_file):
#                     continue

#                 # suffix = "leftImg8bit.png"
#                 # assert basename.endswith(suffix), basename
#                 # basename = basename[: -len(suffix)]

#                 disp_file = os.path.join(sub_disp_dir, seq, 'left', os.path.splitext(basename)[0] + '.pfm')
#                 if not PathManager.isfile(disp_file):
#                     continue


#                 files.append((left_image_file, right_image_file, disp_file))
#     assert len(files), "No images found in {}".format(image_dir)
#     for f in files[0]:
#         assert PathManager.isfile(f), f
#     return files

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
        # imgcv = cv2.imread(file, cv2.IMREAD_UNCHANGED).astype(np.float32)/256
        img = np.asarray(img)/256.0
        return img
    elif file.endswith('.pfm'): return readPFM(file)[0]
    else: raise Exception('don\'t know how to read %s' % file)


def setup(args):
    """
    Create configs and perform basic setups.
    """
    cfg = get_cfg()
    # for poly lr schedule
    add_deeplab_config(cfg)
    add_maskformer2_config(cfg)
    cfg.merge_from_file(args.config_file)
    cfg.merge_from_list(args.opts)
    cfg.freeze()
    default_setup(cfg, args)
    # Setup logger for "mask_former" module
    setup_logger(output=cfg.OUTPUT_DIR, distributed_rank=comm.get_rank(), name="mask2former")
    return cfg


def main(args):
    cfg = setup(args)
    if "sceneflow_train" in cfg.DATASETS.TRAIN:
    # register_cityscapes_stereo(cfg.DATASETS.ROOT)
        register_sceneflow_stereo(cfg)
    elif "kitti_train" in cfg.DATASETS.TRAIN:
        register_kitti_stereo(cfg)
    else:
        raise Exception('Dataset not registered') 

    if args.eval_only:
        model = Trainer.build_model(cfg)
        DetectionCheckpointer(model, save_dir=cfg.OUTPUT_DIR).resume_or_load(
            cfg.MODEL.WEIGHTS, resume=args.resume
        )
        res = Trainer.test(cfg, model)
        if cfg.TEST.AUG.ENABLED:
            res.update(Trainer.test_with_TTA(cfg, model))
        if comm.is_main_process():
            verify_results(cfg, res)
        return res

    trainer = Trainer(cfg)
    trainer.resume_or_load(resume=args.resume)
    return trainer.train()


if __name__ == "__main__":
    args = default_argument_parser().parse_args()
    print("Command Line Args:", args)
    launch(
        main,
        args.num_gpus,
        num_machines=args.num_machines,
        machine_rank=args.machine_rank,
        dist_url=args.dist_url,
        args=(args,),
    )
