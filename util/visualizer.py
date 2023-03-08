# -*- coding: utf-8 -*-
'''
@File    :   visualizer.py
@Time    :   2022/04/05 11:39:33
@Author  :   Shilong Liu 
@Contact :   liusl20@mail.tsinghua.edu.cn; slongliu86@gmail.com
Modified from COCO evaluator
'''

import os, sys
from textwrap import wrap
import torch
import numpy as np
import cv2
import datetime

import matplotlib.pyplot as plt
from matplotlib.collections import PatchCollection
from matplotlib.patches import Polygon
from pycocotools import mask as maskUtils
from matplotlib import transforms

def renorm(img: torch.FloatTensor, mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]) \
        -> torch.FloatTensor:
    # img: tensor(3,H,W) or tensor(B,3,H,W)
    # return: same as img
    assert img.dim() == 3 or img.dim() == 4, "img.dim() should be 3 or 4 but %d" % img.dim() 
    if img.dim() == 3:
        assert img.size(0) == 3, 'img.size(0) shoule be 3 but "%d". (%s)' % (img.size(0), str(img.size()))
        img_perm = img.permute(1,2,0)
        mean = torch.Tensor(mean)
        std = torch.Tensor(std)
        img_res = img_perm * std + mean
        return img_res.permute(2,0,1)
    else: # img.dim() == 4
        assert img.size(1) == 3, 'img.size(1) shoule be 3 but "%d". (%s)' % (img.size(1), str(img.size()))
        img_perm = img.permute(0,2,3,1)
        mean = torch.Tensor(mean)
        std = torch.Tensor(std)
        img_res = img_perm * std + mean
        return img_res.permute(0,3,1,2)

class ColorMap():
    def __init__(self, basergb=[255,255,0]):
        self.basergb = np.array(basergb)
    def __call__(self, attnmap):
        # attnmap: h, w. np.uint8.
        # return: h, w, 4. np.uint8.
        assert attnmap.dtype == np.uint8
        h, w = attnmap.shape
        res = self.basergb.copy()
        res = res[None][None].repeat(h, 0).repeat(w, 1) # h, w, 3
        attn1 = attnmap.copy()[..., None] # h, w, 1
        res = np.concatenate((res, attn1), axis=-1).astype(np.uint8)
        return res


class COCOVisualizer():
    def __init__(self) -> None:
        pass

    def visualize(self, img_path, img, tgt_c, tgt_b, tgt_p, res_lst,
                  caption=None, dpi=120, savedir=None, show_in_console=True):
        """
        img: tensor(3, H, W)
        tgt: make sure they are all on cpu.
            must have items: 'image_id', 'boxes', 'size'
        """
        plt.figure(dpi=dpi)
        plt.rcParams['font.size'] = '5'
        ax = plt.gca()
        img = renorm(img).permute(1, 2, 0)
        # if os.environ.get('IPDB_SHILONG_DEBUG', None) == 'INFO':
        #     import ipdb; ipdb.set_trace()
        ax.imshow(img)
        
        # self.addtgt_c(tgt_c, res_lst)
        self.addtgt_b(tgt_b)
        self.addtgt_p(tgt_p)
        if show_in_console:
            plt.show()

        if savedir is not None:
            if caption is None:
                # savename = '{}/{}-{}.png'.format(savedir, int(tgt['image_id']), str(datetime.datetime.now()).replace(' ', '-'))
                # savename = '{}/visualization-{}.png'.format(savedir, str(datetime.datetime.now()).replace(' ', '-'))
                # path = "./data/synthtext/SynthText/8/ballet_3_0.jpg"
                dir_num, file_name = img_path.split('/')[-2:]
                file_name = file_name[:-4]
                savename = '{}/vis-{}-{}.png'.format(savedir, dir_num, file_name)
            else:
                savename = '{}/{}-{}-{}.png'.format(savedir, caption, int(tgt['image_id']), str(datetime.datetime.now()).replace(' ', '-'))
            print("savename: {}".format(savename))
            os.makedirs(os.path.dirname(savename), exist_ok=True)
            plt.savefig(savename)
        plt.close()

    def addtgt_c(self, tgt, res_lst):
        """
        - tgt: dict. args:
            - boxes: num_boxes, 4. xywh, [0,1].
            - box_label: num_boxes.
        """
        assert 'boxes' in tgt
        ax = plt.gca()
        H, W = tgt['size'].tolist() 
        numbox = tgt['boxes'].shape[0]

        color_pool = [(np.random.random((1, 3))*0.6+0.4).tolist()[0] for _ in range(len(res_lst))]
        color = [(np.random.random((1, 3))*0.6+0.4).tolist()[0] for _ in range(numbox)]
        for i, lst in enumerate(res_lst):
            for k in lst:
                color[k] = color_pool[i]
        
        polygons = []
        boxes = []
        for box in tgt['boxes'].cpu():
            unnormbbox = box * torch.Tensor([W, H, W, H])
            unnormbbox[:2] -= unnormbbox[2:] / 2
            [bbox_x, bbox_y, bbox_w, bbox_h] = unnormbbox.tolist()
            boxes.append([bbox_x, bbox_y, bbox_w, bbox_h])
            poly = [[bbox_x, bbox_y], [bbox_x, bbox_y+bbox_h], [bbox_x+bbox_w, bbox_y+bbox_h], [bbox_x+bbox_w, bbox_y]]
            np_poly = np.array(poly).reshape((4,2))
            polygons.append(Polygon(np_poly))
            # c = (np.random.random((1, 3))*0.6+0.4).tolist()[0]
            # color.append(c)

        p = PatchCollection(polygons, facecolor=color, linewidths=0, alpha=0.1)
        ax.add_collection(p)
        p = PatchCollection(polygons, facecolor='none', edgecolors=color, linewidths=0.5)
        ax.add_collection(p)


        if 'box_label' in tgt:
            assert len(tgt['box_label']) == numbox, f"{len(tgt['box_label'])} = {numbox}, "
            for idx, bl in enumerate(tgt['box_label']):
                _string = str(bl)
                bbox_x, bbox_y, bbox_w, bbox_h = boxes[idx]
                # ax.text(bbox_x, bbox_y, _string, color='black', bbox={'facecolor': 'yellow', 'alpha': 1.0, 'pad': 1})
                ax.text(bbox_x, bbox_y-8, _string, color='black', bbox={'facecolor': color[idx], 'alpha': 0.6, 'pad': 1})

        if 'caption' in tgt:
            ax.set_title(tgt['caption'], wrap=True)
    
    def addtgt_b(self, tgt):
        """
        - tgt: dict. args:
            - curves: num_curves, 8. x1y1x2y2x3y3x4y4, [0,1].
            - size: img size.
        """
        assert 'curves' in tgt
        ax = plt.gca()
        H, W = tgt['size'].tolist() 
        numcurve = tgt['curves'].shape[0]

        for curve in tgt['curves'].cpu():
            unnormcurve = curve * torch.Tensor([W, H]*4)
            [x1, y1, x2, y2, x3, y3, x4, y4] = unnormcurve.tolist()
            points = [[x1, y1], [x2, y2], [x4, y4]]
            c = (np.random.random((1, 3))*0.6+0.4).tolist()[0]
            for x, y in points:
                ax.scatter(x, y, s=3, c=np.array(c).reshape(1, -1))
        
    def addtgt_p(self, tgt):
        assert 'polygons' in tgt
        ax = plt.gca()
        numpoly = tgt['polygons'].shape[0]

        colors = []
        polygons = []
        for polygon in tgt['polygons'].cpu():
            # polygon: [n*2, 2]
            unnormpoly = polygon
            x_lst = unnormpoly[:, 0].tolist()
            y_lst = unnormpoly[:, 1].tolist()
            assert len(x_lst) == len(y_lst)
            poly = [[x, y] for (x, y) in zip(x_lst, y_lst)]
            np_poly = np.array(poly).reshape((-1,2))
            polygons.append(Polygon(np_poly))
            c = (np.random.random((1, 3))*0.6+0.4).tolist()[0]
            colors.append(c)

        p = PatchCollection(polygons, facecolor=colors, linewidths=0, alpha=0.1)
        ax.add_collection(p)
        p = PatchCollection(polygons, facecolor='none', edgecolors=colors, linewidths=0.5)
        ax.add_collection(p)

        if 'poly_label' in tgt:
            assert len(tgt['poly_label']) == numpoly, f"{len(tgt['poly_label'])} = {numpoly}, "
            for idx, bl in enumerate(tgt['poly_label']):
                _string = str(bl)
                _poly = tgt['polygons'][idx]
                start_x, start_y = _poly[0][0].item(), _poly[0][1].item()
                ax.text(start_x, start_y-8, _string, color='black', bbox={'facecolor': colors[idx], 'alpha': 0.5, 'pad': 1})
