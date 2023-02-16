# ------------------------------------------------------------------------
# Deformable DETR
# Copyright (c) 2020 SenseTime. All Rights Reserved.
# Licensed under the Apache License, Version 2.0 [see LICENSE for details]
# ------------------------------------------------------------------------
# Modified from DETR (https://github.com/facebookresearch/detr)
# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
# ------------------------------------------------------------------------

"""
Transforms and data augmentation for both image + bbox.
"""
import random

import PIL
import torch
import torchvision.transforms as T
import torchvision.transforms.functional as F

from util.box_ops import box_xyxy_to_cxcywh
from util.misc import interpolate


def crop(image, target, region):
    cropped_image = F.crop(image, *region)

    target = target.copy()
    i, j, h, w = region  # (top, left, height, width)

    # TODO: should we do something wrt the original size?
    target["size"] = torch.tensor([h, w])  # 将 ground truth 的 size 修改为裁剪后的
    
    max_size = torch.as_tensor([w, h], dtype=torch.float32)

    # -- handle character part --
    tgt_char = target["char"]
    fields_char = ["labels", "area", "iscrowd"]

    if "boxes" in tgt_char:
        boxes = tgt_char["boxes"]  # [num_boxes, 4]
        cropped_boxes = boxes - torch.as_tensor([j, i, j, i])  # ground truth box 坐标平移
        cropped_boxes = torch.min(cropped_boxes.reshape(-1, 2, 2), max_size)  # 裁掉平移后box大于图片高宽的部分
        cropped_boxes = cropped_boxes.clamp(min=0)  # 裁掉平移后小于0的部分
        area = (cropped_boxes[:, 1, :] - cropped_boxes[:, 0, :]).prod(dim=1)  # 用裁剪后的box重新计算面积
        tgt_char["boxes"] = cropped_boxes.reshape(-1, 4)  # 更新
        tgt_char["area"] = area
        fields_char.append("boxes")

    if "masks" in tgt_char:
        # FIXME should we update the area here if there are no boxes?
        tgt_char['masks'] = tgt_char['masks'][:, i:i + h, j:j + w]
        fields_char.append("masks")

    # remove elements for which the boxes or masks that have zero area
    if "boxes" in tgt_char or "masks" in tgt_char:
        # favor boxes selection when defining which elements to keep
        # this is compatible with previous implementation
        if "boxes" in tgt_char:
            cropped_boxes = tgt_char['boxes'].reshape(-1, 2, 2)
            # 去除面积为 0 的 box
            keep_char = torch.all(cropped_boxes[:, 1, :] > cropped_boxes[:, 0, :], dim=1)
        else:
            keep_char = tgt_char['masks'].flatten(1).any(1)

        for field in fields_char:
            tgt_char[field] = tgt_char[field][keep_char]

    target["char"] = tgt_char

    # -- handle bezier part --
    tgt_bezier = target["bezier"]
    fields_bezier = ["labels"]
    keep_bezier = None

    if "points" in tgt_bezier:
        points = tgt_bezier["points"]  # [num_bezier, 8]
        assert points.shape[-1] == 8
        cropped_points = points - torch.as_tensor([j, i]*4)
        tgt_bezier["points"] = cropped_points
        fields_bezier.append("points")

        # TODO: it's hard to decide which bezier curve should be kept
        # 这里的策略是 4 个控制点都在图片内部才保留该 bezier 曲线 
        xs, ys = cropped_points[:, 0::2], cropped_points[:, 1::2]
        keep_bezier_x = torch.logical_and(torch.all(xs >= 0, dim=1), torch.all(xs <= w, dim=1))
        keep_bezier_y = torch.logical_and(torch.all(ys >= 0, dim=1), torch.all(ys <= h, dim=1))
        keep_bezier = torch.logical_and(keep_bezier_x, keep_bezier_y)

    if keep_bezier is not None:
        for field in fields_bezier:
            tgt_bezier[field] = tgt_bezier[field][keep_bezier]

    target["bezier"] = tgt_bezier

    return cropped_image, target


def hflip(image, target):
    flipped_image = F.hflip(image)

    w, h = image.size

    target = target.copy()

    # -- handle char part --
    tgt_char = target["char"]
    if "boxes" in tgt_char:
        boxes = tgt_char["boxes"]  # (x1, y1, x2, y2)
        boxes = boxes[:, [2, 1, 0, 3]] * torch.as_tensor([-1, 1, -1, 1]) + torch.as_tensor([w, 0, w, 0])  # 标签翻转
        tgt_char["boxes"] = boxes

    if "masks" in tgt_char:
        tgt_char['masks'] = tgt_char['masks'].flip(-1)
    
    target["char"] = tgt_char

    # -- handle bezier part --
    tgt_bezier = target["bezier"]
    if "points" in tgt_bezier:
        points = tgt_bezier["points"]
        points = points * torch.as_tensor([-1, 1]*4) + torch.as_tensor([w, 0]*4)  # 点的顺序无需改变
        tgt_bezier["points"] = points
    
    target["bezier"] = tgt_bezier

    return flipped_image, target


def resize(image, target, size, max_size=None):
    # size can be min_size (scalar) or (w, h) tuple

    def get_size_with_aspect_ratio(image_size, size, max_size=None):
        """
        Args:
            image_size: (w, h)
            size: int
            max_size: int

        Returns:
            (oh, ow)
        """
        w, h = image_size
        if max_size is not None:
            min_original_size = float(min((w, h)))
            max_original_size = float(max((w, h)))
            if max_original_size / min_original_size * size > max_size:  # 按照最长边等比例缩放
                size = int(round(max_size * min_original_size / max_original_size))

        if (w <= h and w == size) or (h <= w and h == size):
            return (h, w)

        if w < h:
            ow = size
            oh = int(size * h / w)
        else:
            oh = size
            ow = int(size * w / h)

        return (oh, ow)  # 等比例缩放，短边的大小为 size 的大小 或者 长边的大小为 max_size 大小

    def get_size(image_size, size, max_size=None):
        if isinstance(size, (list, tuple)):
            return size[::-1]
        else:
            return get_size_with_aspect_ratio(image_size, size, max_size)

    # 修改图片大小
    size = get_size(image.size, size, max_size)  # (oh, ow)
    rescaled_image = F.resize(image, size)

    if target is None:
        return rescaled_image, None

    ratios = tuple(float(s) / float(s_orig) for s, s_orig in zip(rescaled_image.size, image.size))
    ratio_width, ratio_height = ratios  # 宽和高的比例

    # 修改 target 属性
    target = target.copy()

    h, w = size
    target["size"] = torch.tensor([h, w])

    # -- handle char part --
    tgt_char = target["char"]
    if "boxes" in tgt_char:
        boxes = tgt_char["boxes"]
        scaled_boxes = boxes * torch.as_tensor([ratio_width, ratio_height, ratio_width, ratio_height])
        tgt_char["boxes"] = scaled_boxes

    if "area" in tgt_char:
        area = tgt_char["area"]
        scaled_area = area * (ratio_width * ratio_height)
        tgt_char["area"] = scaled_area

    if "masks" in tgt_char:
        tgt_char['masks'] = interpolate(
            tgt_char['masks'][:, None].float(), size, mode="nearest")[:, 0] > 0.5

    target["char"] = tgt_char

    # -- handle bezier part --
    tgt_bezier = target["bezier"]
    if "points" in tgt_bezier:
        points = tgt_bezier["points"]
        scaled_points = points * torch.as_tensor([ratio_width, ratio_height] * 4)
        tgt_bezier["points"] = scaled_points

    target["bezier"] = tgt_bezier

    return rescaled_image, target


def pad(image, target, padding):
    # assumes that we only pad on the bottom right corners
    padded_image = F.pad(image, (0, 0, padding[0], padding[1]))
    if target is None:
        return padded_image, None
    target = target.copy()
    # should we do something wrt the original size?
    target["size"] = torch.tensor(padded_image[::-1])
    tgt_char = target["char"]
    if "masks" in tgt_char:
        tgt_char['masks'] = torch.nn.functional.pad(target['masks'], (0, padding[0], 0, padding[1]))
    target["char"] = tgt_char
    return padded_image, target


class RandomCrop(object):
    def __init__(self, size):
        self.size = size

    def __call__(self, img, target):
        region = T.RandomCrop.get_params(img, self.size)
        return crop(img, target, region)


class RandomSizeCrop(object):
    # 修改为随机裁剪，宽和高最小为原图的一半
    def __call__(self, img: PIL.Image.Image, target: dict):
        # 修改：主要是 self.max_size 一般来说太小了
        w = random.randint(img.width // 2, img.width)
        h = random.randint(img.height // 2, img.height)
        region = T.RandomCrop.get_params(img, [h, w])  # (top, left, height, width)
        return crop(img, target, region)


class CenterCrop(object):
    def __init__(self, size):
        self.size = size

    def __call__(self, img, target):
        image_width, image_height = img.size
        crop_height, crop_width = self.size
        crop_top = int(round((image_height - crop_height) / 2.))
        crop_left = int(round((image_width - crop_width) / 2.))
        return crop(img, target, (crop_top, crop_left, crop_height, crop_width))


class RandomHorizontalFlip(object):
    def __init__(self, p=0.5):
        self.p = p

    def __call__(self, img, target):
        if random.random() < self.p:
            return hflip(img, target)
        return img, target


class RandomResize(object):
    def __init__(self, sizes, max_size=None):
        assert isinstance(sizes, (list, tuple))
        self.sizes = sizes  # list (int)
        self.max_size = max_size

    def __call__(self, img, target=None):
        size = random.choice(self.sizes)  # int
        return resize(img, target, size, self.max_size)


class RandomPad(object):
    def __init__(self, max_pad):
        self.max_pad = max_pad

    def __call__(self, img, target):
        pad_x = random.randint(0, self.max_pad)
        pad_y = random.randint(0, self.max_pad)
        return pad(img, target, (pad_x, pad_y))


class RandomSelect(object):
    """
    Randomly selects between transforms1 and transforms2,
    with probability p for transforms1 and (1 - p) for transforms2
    """
    def __init__(self, transforms1, transforms2, p=0.5):
        self.transforms1 = transforms1
        self.transforms2 = transforms2
        self.p = p

    def __call__(self, img, target):
        if random.random() < self.p:
            return self.transforms1(img, target)
        return self.transforms2(img, target)


class ToTensor(object):
    def __call__(self, img, target):
        return F.to_tensor(img), target


class RandomErasing(object):

    def __init__(self, *args, **kwargs):
        self.eraser = T.RandomErasing(*args, **kwargs)

    def __call__(self, img, target):
        return self.eraser(img), target


class Normalize(object):
    """
    为图片和target做归一化
    """
    def __init__(self, mean, std):
        self.mean = mean
        self.std = std

    def __call__(self, image, target=None):
        image = F.normalize(image, mean=self.mean, std=self.std)
        if target is None:
            return image, None
        target = target.copy()
        h, w = image.shape[-2:]
        # -- handle char part --
        tgt_char = target["char"]
        if "boxes" in tgt_char:
            boxes = tgt_char["boxes"]
            boxes = box_xyxy_to_cxcywh(boxes)  # 转成 cx, cy, w, h 形式
            boxes = boxes / torch.tensor([w, h, w, h], dtype=torch.float32)  # 归一化
            tgt_char["boxes"] = boxes
        target["char"] = tgt_char

        # -- handle bezier part --
        tgt_bezier = target["bezier"]
        if "points" in tgt_bezier:
            points = tgt_bezier["points"]
            points = points / torch.tensor([w, h]*4, dtype=torch.float32)
            tgt_bezier["points"] = points
        target["bezier"] = tgt_bezier

        return image, target


class Compose(object):
    def __init__(self, transforms):
        self.transforms = transforms

    def __call__(self, image, target):
        for t in self.transforms:
            image, target = t(image, target)
        return image, target

    def __repr__(self):
        format_string = self.__class__.__name__ + "("
        for t in self.transforms:
            format_string += "\n"
            format_string += "    {0}".format(t)
        format_string += "\n)"
        return format_string
