# ------------------------------------------------------------------------
# Deformable DETR
# Copyright (c) 2020 SenseTime. All Rights Reserved.
# Licensed under the Apache License, Version 2.0 [see LICENSE for details]
# ------------------------------------------------------------------------
# Modified from DETR (https://github.com/facebookresearch/detr)
# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
# ------------------------------------------------------------------------

"""
COCO dataset which returns image_id for evaluation.

Mostly copy-paste from https://github.com/pytorch/vision/blob/13b35ff/references/detection/coco_utils.py
"""
from pathlib import Path

import torch
import torch.utils.data
from pycocotools import mask as coco_mask

from .torchvision_datasets import CocoDetection as TvCocoDetection
from util.misc import get_local_rank, get_local_size
import datasets.transforms as T


class CocoDetection(TvCocoDetection):  # 坑人！注意 21 行，这个继承的不是真正的 TvCocoDetection
    def __init__(self, img_folder, ann_file, transforms, return_masks, cache_mode=False, local_rank=0, local_size=1):
        super(CocoDetection, self).__init__(img_folder, ann_file,
                                            cache_mode=cache_mode, local_rank=local_rank, local_size=local_size)
        self._transforms = transforms
        self.prepare = ConvertCocoPolysToMask(return_masks)

    def __getitem__(self, idx):
        img, target = super(CocoDetection, self).__getitem__(idx)
        image_id = self.ids[idx]
        target = {'image_id': image_id, 'annotations': target}
        # target['annotations']: 列表，列表中的元素是 ann object dict
        img, target = self.prepare(img, target)  # 输入：图片和字典    输出：图片和处理过的字典
        # target
        #   key: 'image_id', 'boxes', 'labels', 'area', 'iscrowd', 'orig_size', 'size'
        if self._transforms is not None:
            img, target = self._transforms(img, target)  # 因为这个转换函数是经过复写的，所以可以输入 img 和 target
        return img, target


def convert_coco_poly_to_mask(segmentations, height, width):
    masks = []
    for polygons in segmentations:
        rles = coco_mask.frPyObjects(polygons, height, width)
        mask = coco_mask.decode(rles)
        if len(mask.shape) < 3:
            mask = mask[..., None]
        mask = torch.as_tensor(mask, dtype=torch.uint8)
        mask = mask.any(dim=2)
        masks.append(mask)
    if masks:
        masks = torch.stack(masks, dim=0)
    else:
        masks = torch.zeros((0, height, width), dtype=torch.uint8)
    return masks


class ConvertCocoPolysToMask(object):
    def __init__(self, return_masks=False):
        self.return_masks = return_masks

    def __call__(self, image, target):
        w, h = image.size  # 坑人..

        image_id = target["image_id"]
        image_id = torch.tensor([image_id])

        anno = target["annotations"]  # list(ann obj dict)

        anno = [obj for obj in anno if 'iscrowd' not in obj or obj['iscrowd'] == 0]
        # 只选择没有键 iscrowd 或者键 iscrowd 对应的值为 0 的字典项

        boxes = [obj["bbox"] for obj in anno]  # 每一项都是 4 个坐标, (x1, y1, w, h)
        # guard against no boxes via resizing
        boxes = torch.as_tensor(boxes, dtype=torch.float32).reshape(-1, 4)
        boxes[:, 2:] += boxes[:, :2]  # 将宽和高转换为右下角坐标
        boxes[:, 0::2].clamp_(min=0, max=w)
        boxes[:, 1::2].clamp_(min=0, max=h)
        # [num_boxes, 4]

        classes = [obj["category_id"] for obj in anno]
        classes = torch.tensor(classes, dtype=torch.int64)  # [num_boxes,]

        if self.return_masks:
            segmentations = [obj["segmentation"] for obj in anno]
            masks = convert_coco_poly_to_mask(segmentations, h, w)

        keypoints = None
        if anno and "keypoints" in anno[0]:
            keypoints = [obj["keypoints"] for obj in anno]
            keypoints = torch.as_tensor(keypoints, dtype=torch.float32)
            num_keypoints = keypoints.shape[0]
            if num_keypoints:
                keypoints = keypoints.view(num_keypoints, -1, 3)

        # 去除无效的 box
        keep = (boxes[:, 3] > boxes[:, 1]) & (boxes[:, 2] > boxes[:, 0])
        boxes = boxes[keep]
        classes = classes[keep]
        if self.return_masks:
            masks = masks[keep]
        if keypoints is not None:
            keypoints = keypoints[keep]

        target = {}
        target["boxes"] = boxes  # [num_boxes_, 4]
        target["labels"] = classes  # [num_boxes_,]
        if self.return_masks:
            target["masks"] = masks
        target["image_id"] = image_id
        if keypoints is not None:
            target["keypoints"] = keypoints

        # for conversion to coco api
        area = torch.tensor([obj["area"] for obj in anno])
        iscrowd = torch.tensor([obj["iscrowd"] if "iscrowd" in obj else 0 for obj in anno])
        target["area"] = area[keep]  # segmentation area
        target["iscrowd"] = iscrowd[keep]

        target["orig_size"] = torch.as_tensor([int(h), int(w)])
        target["size"] = torch.as_tensor([int(h), int(w)])

        # target key: 'image_id', 'boxes', 'labels', 'area', 'iscrowd', 'orig_size', 'size'
        return image, target


def make_coco_transforms(image_set):

    normalize = T.Compose([
        T.ToTensor(),
        T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])  # TODO: 为什么要设置为这些数字？
    ])

    scales = [480, 512, 544, 576, 608, 640, 672, 704, 736, 768, 800]
    img_max_size = 1024  # original: 1333

    if image_set == 'train':
        return T.Compose([
            T.RandomHorizontalFlip(),  # TODO: 场景文本检测可能不需要？
            T.RandomSelect(
                T.RandomResize(scales, max_size=img_max_size),
                T.Compose([
                    T.RandomResize([400, 500, 600]),
                    T.RandomSizeCrop(384, 600),
                    T.RandomResize(scales, max_size=img_max_size),
                ])
            ),
            normalize,
        ])

    if image_set == 'val':
        return T.Compose([
            T.RandomResize([800], max_size=img_max_size),
            normalize,
        ])

    raise ValueError(f'unknown {image_set}')


def build(image_set, args):
    root = Path(args.synthtext_path)
    assert root.exists(), f'provided Synthtext path {root} does not exist'
    mode = 'instances'
    PATHS = {
        "train": (root / "SynthText", root / "annotations" / 'synthtext-rec_train2.json'),
        "val": (root / "SynthText", root / "annotations" / 'synthtext-rec_val2.json'),
    }

    img_folder, ann_file = PATHS[image_set]  # image folder & annotation file
    dataset = CocoDetection(img_folder, ann_file, transforms=make_coco_transforms(image_set), return_masks=args.masks,
                            cache_mode=args.cache_mode, local_rank=get_local_rank(), local_size=get_local_size())
    return dataset
