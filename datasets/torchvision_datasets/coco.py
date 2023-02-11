# ------------------------------------------------------------------------
# Deformable DETR
# Copyright (c) 2020 SenseTime. All Rights Reserved.
# Licensed under the Apache License, Version 2.0 [see LICENSE for details]
# ------------------------------------------------------------------------
# Modified from torchvision
# ------------------------------------------------------------------------

"""
Copy-Paste from torchvision, but add utility of caching images on memory
"""
from torchvision.datasets.vision import VisionDataset
from PIL import Image
import os
import os.path
import tqdm
from io import BytesIO

from pycocotools.coco import COCO
from collections import defaultdict
import itertools
from time import time


def _isArrayLike(obj):
    return hasattr(obj, '__iter__') and hasattr(obj, '__len__')


class COCOBezier(COCO):
    """
    重写 COCO 类，适配增加 bezier 曲线后的 annotation file
    """
    def __init__(self, annotation_file=None):
        super(COCOBezier, self).__init__(annotation_file=annotation_file)
        self.anns_bezier, self.cats_bezier = dict(), dict()
        self.imgToAnnsBezier, self.catBezierToImgs = defaultdict(list), defaultdict(list)
        if annotation_file is not None:
            print('loading annotations (bezier part) into memory...')
            self.createBezierIndex()


    def createBezierIndex(self):
        start = time()
        print('creating index (bezier part)...')
        anns_bezier, cats_bezier = {}, {}
        imgToAnnsBezier, catBezierToImgs = defaultdict(list), defaultdict(list)
        if 'annotations_bezier' in self.dataset:
            for annb in self.dataset['annotations_bezier']:
                imgToAnnsBezier[annb['image_id']].append(annb)
                anns_bezier[annb['id']] = annb
        
        if 'categories_bezier' in self.dataset:
            for catb in self.dataset['categories_bezier']:
                cats_bezier[catb['id']] = catb
        
        if 'annotations_bezier' in self.dataset and 'categories_bezier' in self.dataset:
            for annb in self.dataset['annotations_bezier']:
                catBezierToImgs[annb['category_bezier_id']].append(annb['image_id'])

        print('index (bezier part) created, time: {:.2f}'.format(time()-start))

        self.anns_bezier = anns_bezier
        self.cats_bezier = cats_bezier
        self.imgToAnnsBezier = imgToAnnsBezier
        self.catBezierToImgs = catBezierToImgs

    def getAnnBezierIds(self, imgIds=[]):
        """
        只考虑了传入imgIds的情况
        """
        # from pycocotools.coco import _isArrayLike  # TODO: 这样子行不行呢
        imgIds = imgIds if _isArrayLike(imgIds) else [imgIds]

        if len(imgIds) == 0:
            annbs = self.dataset['annotations_bezier']
        else:
            lists = [self.imgToAnnsBezier[imgId] for imgId in imgIds 
                     if imgId in self.imgToAnnsBezier]
            annbs = list(itertools.chain.from_iterable(lists))
        # iscrowd == None
        ids = [annb['id'] for annb in annbs]
        return ids

    def loadAnnsBezier(self, ids=[]):
        if _isArrayLike(ids):
            return [self.anns_bezier[id] for id in ids]
        elif type(ids) == int:
            return [self.anns_bezier[ids]]


class CocoDetection(VisionDataset):
    """`MS Coco Detection <http://mscoco.org/dataset/#detections-challenge2016>`_ Dataset.
    Args:
        root (string): Root directory where images are downloaded to.
        annFile (string): Path to json annotation file.
        transform (callable, optional): A function/transform that  takes in an PIL image
            and returns a transformed version. E.g, ``transforms.ToTensor``
        target_transform (callable, optional): A function/transform that takes in the
            target and transforms it.
        transforms (callable, optional): A function/transform that takes input sample and its target as entry
            and returns a transformed version.
    """

    def __init__(self, root, annFile, transform=None, target_transform=None, transforms=None,
                 cache_mode=False, local_rank=0, local_size=1):
        super(CocoDetection, self).__init__(root, transforms, transform, target_transform)
        self.coco = COCOBezier(annFile)
        self.ids = list(sorted(self.coco.imgs.keys()))  # 将 annotation file 中所有图片的 id 构成的列表进行排序
        self.cache_mode = cache_mode
        self.local_rank = local_rank
        self.local_size = local_size
        if cache_mode:
            self.cache = {}
            self.cache_images()

    def cache_images(self):
        self.cache = {}
        for index, img_id in zip(tqdm.trange(len(self.ids)), self.ids):
            if index % self.local_size != self.local_rank:
                continue
            path = self.coco.loadImgs(img_id)[0]['file_name']
            with open(os.path.join(self.root, path), 'rb') as f:
                self.cache[path] = f.read()

    def get_image(self, path):
        if self.cache_mode:
            if path not in self.cache.keys():
                with open(os.path.join(self.root, path), 'rb') as f:
                    self.cache[path] = f.read()
            return Image.open(BytesIO(self.cache[path])).convert('RGB')
        return Image.open(os.path.join(self.root, path)).convert('RGB')

    def __getitem__(self, index):
        """
        Args:
            index (int): Index
        Returns:
            tuple: Tuple (image, target_char, target_bezier). 
            target_char is the object returned by ``coco.loadAnns``
            target_bezier is the object returned by ``coco.loadAnnsBezier``
        """
        coco = self.coco
        img_id = self.ids[index]  # self.ids 是已经排好序的列表
        ann_ids = coco.getAnnIds(imgIds=img_id)
        target_char = coco.loadAnns(ann_ids)  # annotation 项的字典构成的列表
        annb_ids = coco.getAnnBezierIds(imgIds=img_id)
        target_bezier = coco.loadAnnsBezier(annb_ids)

        path = coco.loadImgs(img_id)[0]['file_name']  # 获取图片路径

        img = self.get_image(path)
        if self.transforms is not None:  # TODO: need to revise?
            img, target_char, target_bezier = self.transforms(img, target_char, target_bezier)

        return img, target_char, target_bezier  # 图片以及2个列表

    def __len__(self):  # annotation file 中图片总数量
        return len(self.ids)
