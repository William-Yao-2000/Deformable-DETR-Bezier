# ------------------------------------------------------------------------
# Deformable DETR
# Copyright (c) 2020 SenseTime. All Rights Reserved.
# Licensed under the Apache License, Version 2.0 [see LICENSE for details]
# ------------------------------------------------------------------------
# Modified from DETR (https://github.com/facebookresearch/detr)
# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
# ------------------------------------------------------------------------

"""
Train and eval functions used in main.py
"""
import math
import os
import sys
from typing import Iterable

import torch
import util.misc as utils
from datasets.coco_eval import CocoEvaluator
from datasets.panoptic_eval import PanopticEvaluator
from datasets.data_prefetcher import data_prefetcher, to_cuda

# all_lst = []

def _reduce_and_update(metric_logger, loss_dict, weight_dict, is_training=True):
    """
    reduce the loss dict and update the metric logger in training or evaluation

    loss_dict: {'c': {...}, 'b': {...}}
    weight_dict: {'c': {...}, 'b': {...}}
    """
    # reduce losses over all GPUs for logging purposes
    # TODO: 为啥要这样做？这是啥意思？
    # -- the character part --
    loss_dict_reduced_c = utils.reduce_dict(loss_dict['c'])
    loss_dict_reduced_unscaled_c = {f'char_{k}_unscaled': v
                                    for k, v in loss_dict_reduced_c.items()}
    loss_dict_reduced_scaled_c = {f'char_{k}': v * weight_dict['c'][k]
                                for k, v in loss_dict_reduced_c.items() if k in weight_dict['c']}
    losses_reduced_scaled_c = sum(loss_dict_reduced_scaled_c.values())
    loss_value_c = losses_reduced_scaled_c.item()

    # -- the bezier curve part --
    loss_dict_reduced_b = utils.reduce_dict(loss_dict['b'])
    loss_dict_reduced_unscaled_b = {f'bezier_{k}_unscaled': v
                                    for k, v in loss_dict_reduced_b.items()}
    loss_dict_reduced_scaled_b = {f'bezier_{k}': v * weight_dict['b'][k]
                                for k, v in loss_dict_reduced_b.items() if k in weight_dict['b']}
    losses_reduced_scaled_b = sum(loss_dict_reduced_scaled_b.values())
    loss_value_b = losses_reduced_scaled_b.item()

    loss_value = loss_value_c + loss_value_b

    # 记录数据
    metric_logger.update(loss=loss_value)
    # TODO: 太多的话可以吧 unscaled 的部分去掉
    metric_logger.update(char_loss=loss_value_c, **loss_dict_reduced_scaled_c, **loss_dict_reduced_unscaled_c)
    metric_logger.update(bezier_loss=loss_value_b, **loss_dict_reduced_scaled_b, **loss_dict_reduced_unscaled_b)
    metric_logger.update(char_class_error=loss_dict_reduced_c['class_error'])
    metric_logger.update(bezier_class_error=loss_dict_reduced_b['class_error'])

    if is_training:
        if not math.isfinite(loss_value):
            print("Loss is {}, stopping training".format(loss_value))
            print("character part:")
            print(loss_dict_reduced_c)
            print("bezier curve part:")
            print(loss_dict_reduced_b)
            sys.exit(1)


def train_one_epoch(model: torch.nn.Module, criterion: torch.nn.Module,
                    data_loader: Iterable, optimizer: torch.optim.Optimizer,
                    device: torch.device, epoch: int, max_norm: float = 0):
    # set model mode to train
    model.train()
    criterion.train()
    metric_logger = utils.MetricLogger(delimiter="  ")
    metric_logger.add_meter('lr', utils.SmoothedValue(window_size=1, fmt='{value:.6f}'))
    metric_logger.add_meter('char_class_error', utils.SmoothedValue(window_size=1, fmt='{value:.2f}'))
    metric_logger.add_meter('bezier_class_error', utils.SmoothedValue(window_size=1, fmt='{value:.2f}'))
    metric_logger.add_meter('grad_norm', utils.SmoothedValue(window_size=1, fmt='{value:.2f}'))
    header = 'Epoch: [{}]'.format(epoch)
    # print_cnt = 0
    print_freq = 500

    prefetcher = data_prefetcher(data_loader, device, prefetch=True)
    samples, targets = prefetcher.next()

    # for samples, targets in metric_logger.log_every(data_loader, print_freq, header):
    for _ in metric_logger.log_every(range(len(data_loader)), print_freq, header):
        # 计算损失
        outputs = model(samples)
        # print("\noutputs:\n")
        # print(outputs)
        loss_dict = criterion(outputs, targets)
        # {'c': {...}, 'b': {...}}
        # print("\nloss dict:\n")
        # print(loss_dict)
        weight_dict = criterion.weight_dict
        # {'c': {...}, 'b': {...}}
        # print("\nweight_dict:\n")
        # print(weight_dict)
        losses_char = sum(loss_dict['c'][k] * weight_dict['c'][k] for k in loss_dict['c'].keys() if k in weight_dict['c'])
        losses_bezier = sum(loss_dict['b'][k] * weight_dict['b'][k] for k in loss_dict['b'].keys() if k in weight_dict['b'])
        losses = losses_char + losses_bezier
        # print("\nchar loss:", losses_char, "\nbezier loss:", losses_bezier)
        # print("\nlosses:\n")
        # print(losses)
        # norm_losses = losses_char / losses_char.detach() + losses_bezier / losses_bezier.detach()
        # print("\nnorm_losses:\n")
        # print(norm_losses)
        # better?: losses = losses_char / losses_char.detach() + losses_bezier / losses_bezier.detach()

        _reduce_and_update(metric_logger, loss_dict, weight_dict, is_training=True)

        # 梯度清零
        optimizer.zero_grad()
        # 计算梯度，并加以处理（clip...)
        losses.backward()
        if max_norm > 0:
            grad_total_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm)
        else:
            grad_total_norm = utils.get_total_grad_norm(model.parameters(), max_norm)
        # 反向传播
        optimizer.step()

        metric_logger.update(lr=optimizer.param_groups[0]["lr"])
        metric_logger.update(grad_norm=grad_total_norm)

        samples, targets = prefetcher.next()  # 这里用了 prefetcher 来生成数据

        # log_lst = [float(epoch), float(print_cnt * print_freq)]
        # for _, meter in metric_logger.meters.items():
        #     log_lst.append(float(meter.value))
        # all_lst.append(log_lst)

        # print_cnt += 1
        
    # gather the stats from all processes
    metric_logger.synchronize_between_processes()
    print("Averaged stats:", metric_logger)
    return {k: meter.global_avg for k, meter in metric_logger.meters.items()}


@torch.no_grad()
def evaluate(model, criterion, postprocessors, data_loader, base_ds, device, output_dir):
    # 设置模型模式为测试模式
    model.eval()
    criterion.eval()

    metric_logger = utils.MetricLogger(delimiter="  ")
    metric_logger.add_meter('char_class_error', utils.SmoothedValue(window_size=1, fmt='{value:.2f}'))
    metric_logger.add_meter('bezier_class_error', utils.SmoothedValue(window_size=1, fmt='{value:.2f}'))
    header = 'Test:'

    iou_types = tuple(k for k in ('segm', 'bbox') if k in postprocessors.keys())
    coco_evaluator = CocoEvaluator(base_ds, iou_types)
    # coco_evaluator.coco_eval[iou_types[0]].params.iouThrs = [0, 0.1, 0.5, 0.75]
    print_freq = 200

    panoptic_evaluator = None
    if 'panoptic' in postprocessors.keys():
        panoptic_evaluator = PanopticEvaluator(
            data_loader.dataset.ann_file,
            data_loader.dataset.ann_folder,
            output_dir=os.path.join(output_dir, "panoptic_eval"),
        )

    for samples, targets in metric_logger.log_every(data_loader, print_freq, header):
        samples, targets = to_cuda(samples, targets, device, non_blocking=False)

        outputs = model(samples)
        loss_dict = criterion(outputs, targets)
        weight_dict = criterion.weight_dict

        _reduce_and_update(metric_logger, loss_dict, weight_dict, is_training=False)

        orig_target_sizes = torch.stack([t["orig_size"] for t in targets], dim=0)
        results = postprocessors['bbox'](outputs, orig_target_sizes)
        # results = [{'scores': s, 'labels': l, 'boxes': b} for s, l, b in zip(scores, labels, boxes)]
        if 'segm' in postprocessors.keys():
            target_sizes = torch.stack([t["size"] for t in targets], dim=0)
            results = postprocessors['segm'](results, outputs, orig_target_sizes, target_sizes)
        res = {target['image_id'].item(): output for target, output in zip(targets, results)}
        if coco_evaluator is not None:
            coco_evaluator.update(res)

        if panoptic_evaluator is not None:
            res_pano = postprocessors["panoptic"](outputs, target_sizes, orig_target_sizes)
            for i, target in enumerate(targets):
                image_id = target["image_id"].item()
                file_name = f"{image_id:012d}.png"
                res_pano[i]["image_id"] = image_id
                res_pano[i]["file_name"] = file_name

            panoptic_evaluator.update(res_pano)

    # gather the stats from all processes
    metric_logger.synchronize_between_processes()
    print("Averaged stats:", metric_logger)
    if coco_evaluator is not None:
        coco_evaluator.synchronize_between_processes()
    if panoptic_evaluator is not None:
        panoptic_evaluator.synchronize_between_processes()

    # accumulate predictions from all images
    if coco_evaluator is not None:
        coco_evaluator.accumulate()
        coco_evaluator.summarize()
    panoptic_res = None
    if panoptic_evaluator is not None:
        panoptic_res = panoptic_evaluator.summarize()
    stats = {k: meter.global_avg for k, meter in metric_logger.meters.items()}
    if coco_evaluator is not None:
        if 'bbox' in postprocessors.keys():
            stats['coco_eval_bbox'] = coco_evaluator.coco_eval['bbox'].stats.tolist()
        if 'segm' in postprocessors.keys():
            stats['coco_eval_masks'] = coco_evaluator.coco_eval['segm'].stats.tolist()
    if panoptic_res is not None:
        stats['PQ_all'] = panoptic_res["All"]
        stats['PQ_th'] = panoptic_res["Things"]
        stats['PQ_st'] = panoptic_res["Stuff"]
    return stats, coco_evaluator
