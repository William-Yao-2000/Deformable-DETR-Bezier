# ------------------------------------------------------------------------
# Deformable DETR
# Copyright (c) 2020 SenseTime. All Rights Reserved.
# Licensed under the Apache License, Version 2.0 [see LICENSE for details]
# ------------------------------------------------------------------------
# Modified from DETR (https://github.com/facebookresearch/detr)
# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
# ------------------------------------------------------------------------

"""
Deformable DETR model and criterion classes.
"""
import torch
import torch.nn.functional as F
from torch import nn
import math

from util import box_ops
from util.misc import (NestedTensor, nested_tensor_from_tensor_list, inverse_sigmoid)

from .backbone import build_backbone
from .matcher import build_matcher
from .segmentation import (DETRsegm, PostProcessPanoptic, PostProcessSegm)
from .deformable_transformer import build_deforamble_transformer
from .set_criterion import build_set_criterion
import copy


def _get_clones(module, N):
    return nn.ModuleList([copy.deepcopy(module) for _ in range(N)])


class DeformableDETR(nn.Module):
    """ This is the Deformable DETR module that performs object detection """
    def __init__(self, backbone, transformer, num_classes, num_queries, num_feature_levels,
                 aux_loss=True, with_box_refine=False, two_stage=False):
        """ Initializes the model.
        Parameters:
            backbone: torch module of the backbone to be used. See backbone.py
            transformer: torch module of the transformer architecture. See deformable_transformer.py
            num_classes: number of object and bezier classes.
                         (num_classes_char, num_classes_bezier)
            num_queries: number of object and bezier queries, i.e. detection slot. This is the maximal 
                         number of objects and bezier curves DETR can detect in a single image. 
                         For COCO, we recommend 100 queries.
                         (num_queries_char, num_queries_bezier)
            num_feature_levels: number of features in different scales
            aux_loss: True if auxiliary decoding losses (loss at each decoder layer) are to be used.
            with_box_refine: iterative bounding box refinement
            two_stage: two-stage Deformable DETR
        """
        super(DeformableDETR, self).__init__()
        self.num_queries_c = num_queries[0]  # num_queries_characters
        self.num_queries_b = num_queries[1]  # num_qureies_bezier_curves
        self.transformer = transformer
        hidden_dim = transformer.d_model  # default=256
        # 线性层，输出分类结果  sigmoid/softmax 在损失函数里面
        self.class_embed_c = nn.Linear(hidden_dim, num_classes[0])
        # 3层MLP，输出回归框的位置
        # parameters: (input_dim, hidden_dim, output_dim, num_layers)  
        self.bbox_embed = MLP(hidden_dim, hidden_dim, 4, 3)  # (cx, cy, w, h)
        self.class_embed_b = nn.Linear(hidden_dim, num_classes[1])
        self.point_embed = MLP(hidden_dim, hidden_dim, 8, 3)  # (x1, y1, x2, y2, x3, y3, x4, y4)
        self.num_feature_levels = num_feature_levels  # 不同 scale 特征图的数量
        # 嵌入，将 num_queries 个元素嵌入到 hidden_dim*2 维的空间中
        if not two_stage:
            self.query_embed_c = nn.Embedding(num_queries[0], hidden_dim*2)  # TODO: 为什么要嵌入到 hidden_dim*2 这个大小呢
            self.query_embed_b = nn.Embedding(num_queries[1], hidden_dim*2)
        # 提取不同大小的特征图？
        # 感觉这一块应该是 backbone 与 transformer 之间的连接
        # 实际上初始化了 self.input_proj 
        if num_feature_levels > 1:
            num_backbone_outs = len(backbone.strides)  # 骨干网络输出的特征图的数量，3
            input_proj_list = []
            # 直接从骨干网络的每层输出中用 1*1 卷积获取 256 通道的特征图
            for _ in range(num_backbone_outs):
                in_channels = backbone.num_channels[_]
                input_proj_list.append(nn.Sequential(
                    nn.Conv2d(in_channels, hidden_dim, kernel_size=1),  # param: (in_channels, out_channels, kernel_size)
                    nn.GroupNorm(32, hidden_dim),
                ))
            # 从骨干网络的最后一层输出继续做 stride=2 的卷积，得到更小的特征图
            for _ in range(num_feature_levels - num_backbone_outs):
                input_proj_list.append(nn.Sequential(
                    nn.Conv2d(in_channels, hidden_dim, kernel_size=3, stride=2, padding=1),
                    nn.GroupNorm(32, hidden_dim),
                ))
                in_channels = hidden_dim
            self.input_proj = nn.ModuleList(input_proj_list)
        else:
            self.input_proj = nn.ModuleList([
                nn.Sequential(
                    nn.Conv2d(backbone.num_channels[0], hidden_dim, kernel_size=1),  
                    # TODO: 为什么不是-1？
                    # 已解决，因为这种情况下 bacbone.num_channels=[2048]
                    nn.GroupNorm(32, hidden_dim),
                )])
        self.backbone = backbone
        self.aux_loss = aux_loss
        self.with_box_refine = with_box_refine
        self.two_stage = two_stage

        # 模型参数初始化
        prior_prob = 0.01
        bias_value = -math.log((1 - prior_prob) / prior_prob)  # 分类网络初始化的 trick，在 focal loss 中见过
        self.class_embed_c.bias.data = torch.ones(num_classes[0]) * bias_value
        nn.init.constant_(self.bbox_embed.layers[-1].weight.data, 0)
        nn.init.constant_(self.bbox_embed.layers[-1].bias.data, 0)
        self.class_embed_b.bias.data = torch.ones(num_classes[1]) * bias_value
        nn.init.constant_(self.point_embed.layers[-1].weight.data, 0)
        nn.init.constant_(self.point_embed.layers[-1].bias.data, 0)
        for proj in self.input_proj:
            nn.init.xavier_uniform_(proj[0].weight, gain=1)
            nn.init.constant_(proj[0].bias, 0)

        # 感觉下面这段代码是为了让模型能够输出所有 decoder 中间层的输出结果
        # if two-stage, the **last** class_embed and bbox_embed is for region proposal generation
        num_pred_c = (transformer.decoder_c.num_layers + 1) if two_stage else transformer.decoder_c.num_layers
        num_pred_b = (transformer.decoder_b.num_layers + 1) if two_stage else transformer.decoder_b.num_layers
        # TODO: box refine 和 two-stage 的内容还没看
        if with_box_refine:
            # 深复制
            self.class_embed_c = _get_clones(self.class_embed_c, num_pred_c)
            self.bbox_embed = _get_clones(self.bbox_embed, num_pred_c)
            nn.init.constant_(self.bbox_embed[0].layers[-1].bias.data[2:], -2.0)
            # hack implementation for iterative bounding box refinement
            self.transformer.decoder_c.bbox_embed = self.bbox_embed

            self.class_embed_b = _get_clones(self.class_embed_b, num_pred_b)
            self.point_embed = _get_clones(self.point_embed, num_pred_b)
            # hack implementation for iterative bounding box refinement
            self.transformer.decoder_b.point_embed = self.point_embed
        else:
            # 将输出 bounding-box 长和宽的偏移设置为-2.0，反正最后还要再通过 sigmoid 函数
            nn.init.constant_(self.bbox_embed.layers[-1].bias.data[2:], -2.0)
            # 感觉是浅复制，所有 decoder 层共享子网络模型参数
            self.class_embed_c = nn.ModuleList([self.class_embed_c for _ in range(num_pred_c)])
            self.bbox_embed = nn.ModuleList([self.bbox_embed for _ in range(num_pred_c)])
            self.class_embed_b = nn.ModuleList([self.class_embed_b for _ in range(num_pred_b)])
            self.point_embed = nn.ModuleList([self.point_embed for _ in range(num_pred_b)])
            # TODO: 尚未知道 self.transformer 的结构如何，因此还需要修改
            self.transformer.decoder_c.bbox_embed = None
        if two_stage:
            # hack implementation for two-stage
            self.transformer.decoder_c.class_embed_c = self.class_embed_c
            self.transformer.decoder_b.class_embed_b = self.class_embed_b
            for box_embed in self.bbox_embed:
                nn.init.constant_(box_embed.layers[-1].bias.data[2:], 0.0)

    def forward(self, samples: NestedTensor):
        """ The forward expects a NestedTensor, which consists of:
               - samples.tensor: batched images, of shape [batch_size x 3 x H x W]
               - samples.mask: a binary mask of shape [batch_size x H x W], containing 1 on padded pixels

            # TODO: 要修改
            It returns a dict with the following elements:
               - "pred_logits": the classification logits (including no-object) for all queries.
                                Shape= [batch_size x num_queries x (num_classes + 1)]
               - "pred_boxes": The **normalized** boxes coordinates for all queries, represented as
                               (center_x, center_y, height, width). These values are normalized in [0, 1],
                               relative to the size of each individual image (disregarding possible padding).
                               See PostProcess for information on how to retrieve the unnormalized bounding box.
               - "aux_outputs": Optional, only returned when auxilary losses are activated. It is a list of
                                dictionnaries containing the two above keys for each decoder layer.
        """
        # get NestedTensor (mask is used in transformer to restrict attention location?)
        if not isinstance(samples, NestedTensor):
            samples = nested_tensor_from_tensor_list(samples)
        
        # backbone
        features, pos = self.backbone(samples)  # features 是骨干网络中间层的输出以及 mask，pos 是每层的位置编码
                                                # features[l]: [bs, c_l, h_l, w_l], pos[l]: [bs, d_model, h_l, w_l]

        # backbone to transformer input projection
        srcs = []
        masks = []
        for l, feat in enumerate(features):
            src, mask = feat.decompose()  # [bs, c_l, h_l, w_l], [bs, h_l, w_l]
            srcs.append(self.input_proj[l](src))  # self.input_proj[l](src): [bs, d_model, h_l, w_l]
            masks.append(mask)
            assert mask is not None
        if self.num_feature_levels > len(srcs):  # 在backbone最后一层的基础上生成更小的特征图
            _len_srcs = len(srcs)
            for l in range(_len_srcs, self.num_feature_levels):
                if l == _len_srcs:
                    src = self.input_proj[l](features[-1].tensors)
                else:
                    src = self.input_proj[l](srcs[-1])
                m = samples.mask  # [bs, max_h, max_w]
                # 对 mask 下采样，使其与 src 的宽、高相同（前两维强制不采样）
                # 这一步在 backbone 里面也有
                mask = F.interpolate(m[None].float(), size=src.shape[-2:]).to(torch.bool)[0]  # m[None] 相当于 m.unsqueeze(dim=0)
                pos_l = self.backbone[1](NestedTensor(src, mask)).to(src.dtype)  # 输出当前层的位置编码，[bs, d_model, h_l, w_l]
                srcs.append(src)
                masks.append(mask)
                pos.append(pos_l)
                # len(srcs) == len(masks) == len(pos) == self.num_feature_levels
                # srcs[l]: [bs, d_model, h_l, w_l]
                # masks[l]: [bs, h_l, w_l]
                # pos[l]: [bs, d_model, h_l, w_l]

        # transformer (encoder + decoder)
        query_embeds = None
        if not self.two_stage:
            query_embeds_c = self.query_embed_c.weight
            query_embeds_b = self.query_embed_b.weight
            query_embeds = {"c": query_embeds_c, "b": query_embeds_b}

        hs, init_reference, inter_references, enc_outputs_class, enc_outputs_coord_unact = \
            self.transformer(srcs, masks, pos, query_embeds)

        # FFN, output classification and regression
        outputs_classes_c, outputs_classes_b = [], []
        outputs_coords_c, outputs_coords_b = [], []
        # 每层都输出结果
        # -- the character box part --
        for lvl in range(hs["c"].shape[0]):
            if lvl == 0:
                reference_c = init_reference["c"]
            else:
                reference_c = inter_references["c"][lvl - 1]
            reference_c = inverse_sigmoid(reference_c)
            outputs_class_c = self.class_embed_c[lvl](hs["c"][lvl])  # [bs, num_queries_c, num_classes_c]
            tmp_c = self.bbox_embed[lvl](hs["c"][lvl])  # [bs, num_queries_c, 4]
            if reference_c.shape[-1] == 4:
                tmp_c += reference_c
            else:
                assert reference_c.shape[-1] == 2
                tmp_c[..., :2] += reference_c
            outputs_coord_c = tmp_c.sigmoid()
            outputs_classes_c.append(outputs_class_c)
            outputs_coords_c.append(outputs_coord_c)
        outputs_class_c = torch.stack(outputs_classes_c)
        outputs_coord_c = torch.stack(outputs_coords_c)

        # -- the bezier curve part --
        for lvl in range(hs["b"].shape[0]):
            if lvl == 0:
                reference_b = init_reference["b"]
            else:
                reference_b = inter_references["b"][lvl - 1]
            reference_b = inverse_sigmoid(reference_b)
            assert reference_b.shape[-1] == 2
            outputs_class_b = self.class_embed_b[lvl](hs["b"][lvl])  # [bs, num_queries_b, num_classes_b]
            tmp_b = self.point_embed[lvl](hs["b"][lvl])  # [bs, num_queries_b, 8]
            tmp_b += torch.cat([reference_b]*4, -1)
            outputs_coord_b = tmp_b.sigmoid()
            outputs_classes_b.append(outputs_class_b)
            outputs_coords_b.append(outputs_coord_b)
        outputs_class_b = torch.stack(outputs_classes_b)
        outputs_coord_b = torch.stack(outputs_coords_b)

        out = {
            'char': {
                'pred_logits': outputs_class_c[-1], 'pred_boxes': outputs_coord_c[-1]
            },
            'bezier': {
                'pred_logits': outputs_class_b[-1], 'pred_points': outputs_coord_b[-1]
            }
        }

        if self.aux_loss:
            out['char']['aux_outputs'] = self._set_aux_loss(outputs_class_c, outputs_coord_c, mode='c')
            out['bezier']['aux_outputs'] = self._set_aux_loss(outputs_class_b, outputs_coord_b, mode='b')
            # TODO: need to revise
            pass

        if self.two_stage:
            # activation
            enc_outputs_coord_c = enc_outputs_coord_unact["c"].sigmoid()
            enc_outputs_coord_b = enc_outputs_coord_unact["b"].sigmoid()
            # TODO: enc_outputs_class 应该怎么办呢？是保留原有的所有输出，还是说只保留topk的输出？
            out['char']['enc_outputs'] = {'pred_logits': enc_outputs_class["c"], 'pred_boxes': enc_outputs_coord_c}
            out['bezier']['enc_outputs'] = {'pred_logits': enc_outputs_class["b"], 'pred_points': enc_outputs_coord_b}
        return out  # a dictionary

    @torch.jit.unused
    def _set_aux_loss(self, outputs_class_x, outputs_coord_x, mode):
        # this is a workaround to make torchscript happy, as torchscript
        # doesn't support dictionary with non-homogeneous values, such
        # as a dict having both a Tensor and a list.
        assert mode in ('c', 'b')
        pred_xx = 'pred_boxes' if mode == 'c' else 'pred_points'
        return [{'pred_logits': a, pred_xx: b}
                for a, b in zip(outputs_class_x[:-1], outputs_coord_x[:-1])]


class PostProcess(nn.Module):
    """ This module converts the model's output into the format expected by the coco api"""

    @torch.no_grad()
    def forward(self, outputs, target_sizes):
        """ Perform the computation
        Parameters:
            outputs: raw outputs of the model
            target_sizes: tensor of dimension [batch_size x 2] containing the size of each images of the batch
                          For evaluation, this must be the original image size (before any data augmentation)
                          For visualization, this should be the image size after data augment, but before padding
        """
        out_logits, out_bbox = outputs['char']['pred_logits'], outputs['char']['pred_boxes']
        # [bs, num_queries, num_classes], [bs, num_queries, 4]

        assert len(out_logits) == len(target_sizes)
        assert target_sizes.shape[1] == 2

        prob = out_logits.sigmoid()
        # 取每个 batch 的所有 queries 在所有类上预测结果最高的 100 个值
        # TODO: 根据 synthtext 的特点，改成 300！！！
        topk_values, topk_indexes = torch.topk(prob.view(out_logits.shape[0], -1), 300, dim=1)  # [bs, 300]
        scores = topk_values
        topk_boxes = topk_indexes // out_logits.shape[2]
        labels = topk_indexes % out_logits.shape[2]
        boxes = box_ops.box_cxcywh_to_xyxy(out_bbox)
        boxes = torch.gather(boxes, 1, topk_boxes.unsqueeze(-1).repeat(1,1,4))  # 根据索引来取得对应结果

        # and from relative [0, 1] to absolute [0, height] coordinates
        img_h, img_w = target_sizes.unbind(1)
        scale_fct = torch.stack([img_w, img_h, img_w, img_h], dim=1)
        boxes = boxes * scale_fct[:, None, :]

        results = [{'scores': s, 'labels': l, 'boxes': b} for s, l, b in zip(scores, labels, boxes)]

        return results


class MLP(nn.Module):
    """ Very simple multi-layer perceptron (also called FFN)"""

    def __init__(self, input_dim, hidden_dim, output_dim, num_layers):
        super().__init__()
        self.num_layers = num_layers
        h = [hidden_dim] * (num_layers - 1)
        self.layers = nn.ModuleList(nn.Linear(n, k) for n, k in zip([input_dim] + h, h + [output_dim]))  # 这个写法挺妙的，简洁！

    def forward(self, x):
        for i, layer in enumerate(self.layers):
            x = F.relu(layer(x)) if i < self.num_layers - 1 else layer(x)  # 最后一层不用 activation function
        return x


def build(args):
    assert args.dataset_file in ('coco', 'coco_panoptic', 'synthtext')
    num_classes = 0
    if args.dataset_file == 'coco':
        num_classes = 91
    elif args.dataset_file == 'coco_panoptic':
        num_classes = 250
    elif args.dataset_file == 'synthtext':  # 数据集中类的数目，synthtext recog: 94
        num_classes = (95, 2)
    device = torch.device(args.device)

    # 创建 deformable-DETR 模型
    backbone = build_backbone(args)  # backbone + positional embedding
    transformer = build_deforamble_transformer(args)
    model = DeformableDETR(
        backbone,
        transformer,
        num_classes=num_classes,  # coco: 91
        num_queries=args.num_queries,  # default=(330, 100)
        num_feature_levels=args.num_feature_levels,  # 4
        aux_loss=args.aux_loss,  # True
        with_box_refine=args.with_box_refine,  # False
        two_stage=args.two_stage,  # False
    )
    if args.masks:
        model = DETRsegm(model, freeze_detr=(args.frozen_weights is not None))

    # 定义匹配器及损失函数
    matcher = build_matcher(args)  # 匈牙利二分图匹配，a dict
    
    # weight_dict
    weight_dict = {"c": {}, "b": {}}
    weight_dict["c"] = {'loss_ce': args.cls_c_loss_coef,
                        'loss_bbox': args.bbox_loss_coef, 
                        'loss_giou': args.giou_loss_coef}
    if args.masks:
        weight_dict["c"]["loss_mask"] = args.mask_loss_coef
        weight_dict["c"]["loss_dice"] = args.dice_loss_coef
    weight_dict["b"] = {'loss_ce': args.cls_b_loss_coef,
                        'loss_point': args.point_loss_coef}
    origin_weight_dict = copy.deepcopy(weight_dict)
    # TODO this is a hack
    if args.aux_loss:
        aux_weight_dict_c, aux_weight_dict_b = {}, {}
        for i in range(args.dec_layers - 1):  # TODO: 以后两个decoder可能层数不一样，注意就是了
            aux_weight_dict_c.update({k + f'_{i}': v for k, v in origin_weight_dict["c"].items()})
            aux_weight_dict_b.update({k + f'_{i}': v for k, v in origin_weight_dict["b"].items()})
        weight_dict["c"].update(aux_weight_dict_c)
        weight_dict["b"].update(aux_weight_dict_b)
    if args.two_stage:
        weight_dict["c"].update({k + f'_enc': v for k, v in origin_weight_dict["c"].items()})
        weight_dict["b"].update({k + f'_enc': v for k, v in origin_weight_dict["b"].items()})

    # losses
    losses = {"c": [], "b": []}
    losses["c"] = ['labels', 'boxes', 'cardinality']
    if args.masks:
        losses["c"] += ["masks"]
    losses["b"] = ['labels', 'points', 'cardinality']

    # criterion
    criterion = build_set_criterion(num_classes, matcher, weight_dict, losses, 
                                    focal_alpha=args.focal_alpha)
    criterion.to(device)

    # 定义后处理器
    postprocessors = {'bbox': PostProcess()}
    if args.masks:
        postprocessors['segm'] = PostProcessSegm()
        if args.dataset_file == "coco_panoptic":
            is_thing_map = {i: i <= 90 for i in range(201)}
            postprocessors['panoptic'] = PostProcessPanoptic(is_thing_map, threshold=0.85)

    return model, criterion, postprocessors
