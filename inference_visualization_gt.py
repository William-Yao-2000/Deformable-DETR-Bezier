# ------------------------------------------------------------------------
# Deformable DETR
# Copyright (c) 2020 SenseTime. All Rights Reserved.
# Licensed under the Apache License, Version 2.0 [see LICENSE for details]
# ------------------------------------------------------------------------
# Modified from DETR (https://github.com/facebookresearch/detr)
# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
# ------------------------------------------------------------------------


import argparse
import datetime
import json
import random
import time
from pathlib import Path
from PIL import Image

import numpy as np
import torch
from torch.utils.data import DataLoader
import torchvision.transforms as T
import torchvision.transforms.functional as F
import datasets
from util import box_ops
import util.misc as utils
from util import visualizer
import datasets.samplers as samplers
from datasets import build_dataset, get_coco_api_from_dataset
from engine import evaluate, train_one_epoch
from models import build_model
from datasets.coco import make_coco_transforms

import scipy
from scipy import io


def main(img_path_gt):
    transforms = None
    transforms = make_coco_transforms('val')
    print(transforms)

    gt_mat_path = "./data/synthtext/SynthText/gt.mat"
    vis_gt(img_path_gt, transforms, gt_mat_path)


def get_bbox_gt(coordinates: np.ndarray, img_size):
    assert coordinates.ndim == 3  # [2, 4, char_num]
    print("img_size:", img_size)
    coordinates = np.around(coordinates, 2)
    x_coor, y_coor = coordinates[0], coordinates[1]  # [4, char_num]
    x_min, x_max = np.min(x_coor, axis=0), np.max(x_coor, axis=0)  # [char_num,]
    y_min, y_max = np.min(y_coor, axis=0), np.max(y_coor, axis=0)
    x_min, x_max = torch.tensor(x_min) / img_size[0], torch.tensor(x_max) / img_size[0]
    y_min, y_max = torch.tensor(y_min) / img_size[1], torch.tensor(y_max) / img_size[1]
    return torch.stack((x_min, y_min, x_max, y_max), dim=-1)


def get_string(text_lst):
    res_str = ""
    for word in text_lst:
        for c in word:
            if c not in (' ', '\t', '\n'):
                res_str += c
    return res_str


def vis_gt(img_path_gt: str, transforms, gt_mat_path):
    origin_img = Image.open(img_path_gt).convert('RGB')
    origin_img_size = torch.tensor(origin_img.size)
    print("origin_img_size:", origin_img_size)
    img_gt, _ = transforms(origin_img, None)
    img_gt = img_gt.unsqueeze(0)
    print(img_gt.shape)
    print('img_gt:\n', img_gt)
    img_gt_size = torch.tensor(img_gt.shape)[-2:]
    print(img_gt_size)
    
    features = scipy.io.loadmat(gt_mat_path)
    total = len(features["wordBB"][0])
    img_path_find = '/'.join(img_path_gt.split('/')[-2:])
    print("img_path_find:", img_path_find)
    for i in range(total):
        if features["imnames"][0][i][0] == img_path_find:
            print("HAHA!!")
            coordinates = features['charBB'][0][i]
            # 产生的bbox是xxyy，未经过归一化
            coordinates = get_bbox_gt(coordinates, origin_img_size)
            text_lst = features["txt"][0][i]
            label_string = get_string(text_lst)
            print("shape of coordinates:", coordinates.shape)
            print("len of label string:", len(label_string))
            print(label_string)
            assert len(label_string) == coordinates.shape[0]
            break
    vslzr = visualizer.COCOVisualizer()
    boxes_gt = box_ops.box_xyxy_to_cxcywh(coordinates)
    # print('select mask:', select_mask)
    pred_dict = {
        'boxes': boxes_gt,  # xywh, [0,1]
        'size': img_gt_size,
        'box_label': [c for c in label_string]
    }
    print(pred_dict['boxes'])
    img_cpu = img_gt.squeeze(0).to('cpu')
    print(img_cpu.shape)
    vslzr.visualize(img_path_gt, img_cpu, pred_dict, savedir=f'./vis/gt_temp', dpi=200)


if __name__ == '__main__':
    img_path_gt = './data/synthtext/SynthText/8/ballet_3_0.jpg'
    main(img_path_gt)
