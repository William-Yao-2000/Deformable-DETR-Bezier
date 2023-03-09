# ------------------------------------------------------------------------
# Deformable DETR
# Copyright (c) 2020 SenseTime. All Rights Reserved.
# Licensed under the Apache License, Version 2.0 [see LICENSE for details]
# ------------------------------------------------------------------------
# Modified from DETR (https://github.com/facebookresearch/detr)
# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
# ------------------------------------------------------------------------

from pathlib import Path

import torch
import torch.utils.data
from pycocotools import mask as coco_mask

from .torchvision_datasets import CocoDetection as TvCocoDetection
from util.misc import get_local_rank, get_local_size, nested_tensor_from_tensor_list
import datasets.transforms as T
from .coco import make_coco_transforms

import json
import os
import os.path
from PIL import Image


class ImgTest:  # 坑人！注意 21 行，这个继承的不是真正的 TvCocoDetection
    def __init__(self, img_folder, img_lst_file, transforms):
        self.root = img_folder
        with open(img_lst_file, 'r') as fp:
            self.img_names = json.load(fp)  # a list
        self._transforms = transforms

    def __getitem__(self, idx):
        path = self.img_names[idx]
        path_ = os.path.join(self.root, path)
        img = Image.open(path_).convert('RGB')
        if self._transforms is not None:
            img, _ = self._transforms(img, None)
        inform = {
            'shape': torch.tensor(img.shape[-2:]),
            'path': path_
        }
        return img, inform
    
    def __len__(self):
        return len(self.img_names)


def build_dataset_test(args):
    if args.dataset_name == 'synthtext':
        root = Path(args.synthtext_path)
        root_img = Path("/DATACENTER/raid0/yaowenhao/proj/Deformable-DETR-Synthtext-recog/data/synthtext/SynthText")
        PATHS = (root_img, root / "test" / 'test_img_file.json')
    elif args.dataset_name == 'icdar-2015':
        root = Path(args.icdar_2015_path)
        root_img = Path("./data/icdar-2015/test")
        PATHS = (root_img, root / "test_anno" / 'test_img_file.json')
    assert root.exists(), f'provided Synthtext path {root} does not exist'

    img_folder, img_lst_file = PATHS  # image folder & annotation file
    dataset = ImgTest(img_folder, img_lst_file, transforms=make_coco_transforms('test'))
    return dataset


# 好像没什么用了
def collate_fn(batch):
    batch = list(zip(*batch))
    batch[0] = nested_tensor_from_tensor_list(batch[0])
    return tuple(batch)