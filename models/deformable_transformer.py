# ------------------------------------------------------------------------
# Deformable DETR
# Copyright (c) 2020 SenseTime. All Rights Reserved.
# Licensed under the Apache License, Version 2.0 [see LICENSE for details]
# ------------------------------------------------------------------------
# Modified from DETR (https://github.com/facebookresearch/detr)
# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
# ------------------------------------------------------------------------

import copy
from typing import Optional, List
import math

import torch
import torch.nn.functional as F
from torch import nn, Tensor
from torch.nn.init import xavier_uniform_, constant_, uniform_, normal_

from util.misc import inverse_sigmoid
from models.ops.modules import MSDeformAttn


class DeformableTransformer(nn.Module):
    def __init__(self, d_model=256, nhead=8,
                 num_encoder_layers=6, num_decoder_layers=6, dim_feedforward=1024, dropout=0.1,
                 activation="relu", return_intermediate_dec=False,
                 num_feature_levels=4, dec_n_points=4, enc_n_points=4,
                 two_stage=False, two_stage_num_proposals=(330, 100)):
        super().__init__()

        self.d_model = d_model
        self.nhead = nhead
        self.two_stage = two_stage
        self.two_stage_num_proposals = two_stage_num_proposals

        encoder_layer = DeformableTransformerEncoderLayer(d_model, dim_feedforward,
                                                          dropout, activation,
                                                          num_feature_levels, nhead, enc_n_points)
        self.encoder = DeformableTransformerEncoder(encoder_layer, num_encoder_layers)

        decoder_layer = DeformableTransformerDecoderLayer(d_model, dim_feedforward,
                                                          dropout, activation,
                                                          num_feature_levels, nhead, dec_n_points)
        self.decoder_c = DeformableTransformerDecoder(decoder_layer, num_decoder_layers, return_intermediate_dec)
        self.decoder_b = DeformableTransformerDecoder(decoder_layer, num_decoder_layers, return_intermediate_dec)

        self.level_embed = nn.Parameter(torch.Tensor(num_feature_levels, d_model))  # 层级编码

        if two_stage:
            self.enc_output_c = nn.Linear(d_model, d_model)
            self.enc_output_norm_c = nn.LayerNorm(d_model)
            self.pos_trans_c = nn.Linear(d_model * 2, d_model * 2)
            self.pos_trans_norm_c = nn.LayerNorm(d_model * 2)

            self.enc_output_b = nn.Linear(d_model, d_model)
            self.enc_output_norm_b = nn.LayerNorm(d_model)
            self.pos_trans_b = nn.Linear(d_model * 4, d_model * 2)  # 是4不是2！
            self.pos_trans_norm_b = nn.LayerNorm(d_model * 2)
        else:
            self.reference_points_c = nn.Linear(d_model, 2)
            self.reference_points_b = nn.Linear(d_model, 2)

        self._reset_parameters()

    # 参数初始化
    def _reset_parameters(self):
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)
        for m in self.modules():
            if isinstance(m, MSDeformAttn):
                m._reset_parameters()
        if not self.two_stage:
            xavier_uniform_(self.reference_points_c.weight.data, gain=1.0)
            constant_(self.reference_points_c.bias.data, 0.)
            xavier_uniform_(self.reference_points_b.weight.data, gain=1.0)
            constant_(self.reference_points_b.bias.data, 0.)
        normal_(self.level_embed)

    def get_proposal_pos_embed(self, proposals):
        """
        Args:
            proposals: [bs, num_queries_c, 4] or [bs, num_queries_b, 8]
        """
        num_pos_feats = 128
        temperature = 10000
        scale = 2 * math.pi

        dim_t = torch.arange(num_pos_feats, dtype=torch.float32, device=proposals.device)  # [num_pos_feats,]
        dim_t = temperature ** (2 * (dim_t // 2) / num_pos_feats)  # [num_pos_feats,]
        proposals = proposals.sigmoid() * scale  # [bs, num_queries_c, 4] or [bs, num_queries_b, 8]
        pos = proposals[:, :, :, None] / dim_t
        # [bs, num_queries_c, 4, num_pos_feats] or [bs, num_queries_b, 8, num_pos_feats]
        pos = torch.stack((pos[:, :, :, 0::2].sin(), pos[:, :, :, 1::2].cos()), dim=4).flatten(2)
        # [bs, num_queries_c, 4, num_pos_feats//2, 2] --> [bs, num_queries_c, num_pos_feats*4] or
        # [bs, num_queries_b, 8, num_pos_feats//2, 2] --> [bs, num_queries_b, num_pos_feats*8]
        return pos

    def gen_encoder_output_proposals(self, memory, memory_padding_mask, spatial_shapes):
        """获取线性投影后的memory及初始化的proposal

        Args:
            memory: [bs, \sum_{l=0}^{L-1} h_l * w_l, c]
            memory_padding_mask: [bs, \sum_{l=0}^{L-1} h_l * w_l]
            spatial_shapes: [L, 2]

        Returns:
            output_memory: 经过线性投射和归一化的（字符或Bezier曲线）的memory。
                           包含的key: "c", "b"
                "c": [bs, \sum_{l=0}^{L-1} h_l * w_l, c]    c == d_model
                "b": [bs, \sum_{l=0}^{L-1} h_l * w_l, c]
            output_proposals: 每个sample的每个位置的特征对应的初始化的（字符或Bezier曲线）proposal。
                              包含的key: "c", "b"
                "c": [bs, \sum_{l=0}^{L-1} h_l * w_l, 4]
                "b": [bs, \sum_{l=0}^{L-1} h_l * w_l, 8]
        """
        N_, S_, C_ = memory.shape
        base_scale = 4.0
        proposals_c, proposals_b = [], []
        _cur = 0
        for lvl, (H_, W_) in enumerate(spatial_shapes):
            mask_flatten_ = memory_padding_mask[:, _cur:(_cur + H_ * W_)].view(N_, H_, W_, 1)
            # [bs, h_l * w_l] --> [bs, h_l, w_l, 1]
            valid_H = torch.sum(~mask_flatten_[:, :, 0, 0], 1)  # [bs,]
            valid_W = torch.sum(~mask_flatten_[:, 0, :, 0], 1)  # [bs,]

            grid_y, grid_x = torch.meshgrid(torch.linspace(0, H_ - 1, H_, dtype=torch.float32, device=memory.device),
                                            torch.linspace(0, W_ - 1, W_, dtype=torch.float32, device=memory.device))
            # grid_y, grid_x: [h_l, w_l]
            grid = torch.cat([grid_x.unsqueeze(-1), grid_y.unsqueeze(-1)], -1)  # [h_l, w_l, 2]
            # 每个点的坐标

            scale = torch.cat([valid_W.unsqueeze(-1), valid_H.unsqueeze(-1)], 1).view(N_, 1, 1, 2)  # 每个sample的有效尺寸，[bs, 1, 1, 2]
            grid = (grid.unsqueeze(0).expand(N_, -1, -1, -1) + 0.5) / scale  # 归一化，[bs, h_l, w_l, 2]
            wh = torch.ones_like(grid) * 0.05 * (2.0 ** lvl)  # [bs, h_l, w_l, 2]
            # 对应原始图片的宽高（不同层次的特征对应不同大小的初始候选框）

            proposal_c = torch.cat((grid, wh), -1).view(N_, -1, 4)
            # [bs, h_l, w_l, 4] --> [bs, h_l * w_l, 4]
            proposals_c.append(proposal_c)
            proposal_b = torch.cat([grid]*4, -1).view(N_, -1, 8)
            # [bs, h_l, w_l, 8] --> [bs, h_l * w_l, 8]
            proposals_b.append(proposal_b)
            _cur += (H_ * W_)
        # output_proposals = {}
        output_proposals_c = torch.cat(proposals_c, 1)  # [bs, \sum_{l=0}^{L-1} h_l * w_l, 4]
        # 舍弃靠近边界的点
        output_proposals_valid_c = ((output_proposals_c > 0.01) & (output_proposals_c < 0.99)).all(-1, keepdim=True)
        # 编码，这是sigmoid函数的反函数
        output_proposals_c = torch.log(output_proposals_c / (1 - output_proposals_c))  # [bs, \sum_{l=0}^{L-1} h_l * w_l, 4]
        output_proposals_c = output_proposals_c.masked_fill(memory_padding_mask.unsqueeze(-1), float('inf'))
        # [bs, \sum_{l=0}^{L-1} h_l * w_l, 4]
        output_proposals_c = output_proposals_c.masked_fill(~output_proposals_valid_c, float('inf'))
        # [bs, \sum_{l=0}^{L-1} h_l * w_l, 4]

        output_proposals_b = torch.cat(proposals_b, 1)  # [bs, \sum_{l=0}^{L-1} h_l * w_l, 8]
        output_proposals_valid_b = ((output_proposals_b > 0.01) & (output_proposals_b < 0.99)).all(-1, keepdim=True)
        output_proposals_b = torch.log(output_proposals_b / (1 - output_proposals_b))  # [bs, \sum_{l=0}^{L-1} h_l * w_l, 8]
        output_proposals_b = output_proposals_b.masked_fill(memory_padding_mask.unsqueeze(-1), float('inf'))
        # [bs, \sum_{l=0}^{L-1} h_l * w_l, 8]
        output_proposals_b = output_proposals_b.masked_fill(~output_proposals_valid_b, float('inf'))
        # [bs, \sum_{l=0}^{L-1} h_l * w_l, 8]

        output_proposals = {"c": output_proposals_c, "b": output_proposals_b}

        output_memory_c, output_memory_b = memory, memory  # [bs, \sum_{l=0}^{L-1} h_l * w_l, c]
        
        output_memory_c = output_memory_c.masked_fill(memory_padding_mask.unsqueeze(-1), float(0))
        output_memory_c = output_memory_c.masked_fill(~output_proposals_valid_c, float(0))
        output_memory_c = self.enc_output_norm_c(self.enc_output_c(output_memory_c))
        # [bs, \sum_{l=0}^{L-1} h_l * w_l, c]

        output_memory_b = output_memory_b.masked_fill(memory_padding_mask.unsqueeze(-1), float(0))
        output_memory_b = output_memory_b.masked_fill(~output_proposals_valid_b, float(0))
        output_memory_b = self.enc_output_norm_b(self.enc_output_b(output_memory_b))
        # [bs, \sum_{l=0}^{L-1} h_l * w_l, c]

        output_memory = {"c": output_memory_c, "b": output_memory_b}
        
        return output_memory, output_proposals

    def get_valid_ratio(self, mask):
        _, H, W = mask.shape  # [bs, h_l, w_l]
        valid_H = torch.sum(~mask[:, :, 0], 1)  # [bs,]
        valid_W = torch.sum(~mask[:, 0, :], 1)  # [bs,]
        valid_ratio_h = valid_H.float() / H
        valid_ratio_w = valid_W.float() / W
        valid_ratio = torch.stack([valid_ratio_w, valid_ratio_h], -1)  # [bs, 2]
        return valid_ratio

    def forward(self, srcs, masks, pos_embeds, query_embed=None):
        # now the query_embed is a dict
        assert self.two_stage or query_embed is not None

        # prepare input for encoder
        # 展平、连接
        src_flatten = []
        mask_flatten = []
        lvl_pos_embed_flatten = []
        spatial_shapes = []
        for lvl, (src, mask, pos_embed) in enumerate(zip(srcs, masks, pos_embeds)):
            # src: [bs, d_model, h_l, w_l]
            # mask: [bs, h_l, w_l]
            # pos_embed: [bs, d_model, h_l, w_l]
            bs, c, h, w = src.shape
            spatial_shape = (h, w)
            spatial_shapes.append(spatial_shape)
            src = src.flatten(2).transpose(1, 2)  # [bs, h_l * w_l, c]    c == d_model == 256
            mask = mask.flatten(1)  # [bs, h_l * w_l]
            pos_embed = pos_embed.flatten(2).transpose(1, 2)  # [bs, h_l * w_l, c]
            # 位置编码+层级编码
            lvl_pos_embed = pos_embed + self.level_embed[lvl].view(1, 1, -1)  # [bs, h_l * w_l, c]，用到了广播机制
            src_flatten.append(src)
            mask_flatten.append(mask)
            lvl_pos_embed_flatten.append(lvl_pos_embed)
        src_flatten = torch.cat(src_flatten, 1)  # [bs, \sum_{l=0}^{L-1} h_l * w_l, c]
        mask_flatten = torch.cat(mask_flatten, 1)  # [bs, \sum_{l=0}^{L-1} h_l * w_l]
        lvl_pos_embed_flatten = torch.cat(lvl_pos_embed_flatten, 1)  # [bs, \sum_{l=0}^{L-1} h_l * w_l, c]
        spatial_shapes = torch.as_tensor(spatial_shapes, dtype=torch.long, device=src_flatten.device)  # [L, 2]
        # 将多层特征图展平连接后，每层的起点对应展平向量的里面的 index
        # 这里用 new_zeros 主要是为了保证 dtype 和 device 相同 
        level_start_index = torch.cat((spatial_shapes.new_zeros((1, )), spatial_shapes.prod(1).cumsum(0)[:-1]))  # [L,]（去掉了终点）
        # self.get_valid_ratio(m): mini-batch 中每张图片的宽和高上的有效值的比例，[bs, 2]
        # valid_ratios: 加了一个层数，在 dim=1 **扩维拼接**起来
        valid_ratios = torch.stack([self.get_valid_ratio(m) for m in masks], 1)  # [bs, L, 2]

        # encoder
        memory = self.encoder(src_flatten, spatial_shapes, level_start_index, valid_ratios, lvl_pos_embed_flatten, mask_flatten)
        # [bs, \sum_{l=0}^{L-1} h_l * w_l, c]

        # prepare input for decoder
        bs, _, c = memory.shape
        if self.two_stage:
            output_memory, output_proposals = self.gen_encoder_output_proposals(memory, mask_flatten, spatial_shapes)

            # hack implementation for two-stage Deformable DETR
            enc_outputs_class_c = self.decoder_c.class_embed_c[self.decoder_c.num_layers](output_memory["c"])
            # [bs, \sum_{l=0}^{L-1} h_l * w_l, num_classes_c]
            enc_outputs_coord_unact_c = self.decoder_c.bbox_embed[self.decoder_c.num_layers](output_memory["c"]) + output_proposals["c"]
            # [bs, \sum_{l=0}^{L-1} h_l * w_l, 4]
            enc_outputs_class_b = self.decoder_b.class_embed_b[self.decoder_b.num_layers](output_memory["b"])
            # [bs, \sum_{l=0}^{L-1} h_l * w_l, num_classes_b]
            enc_outputs_coord_unact_b = self.decoder_b.point_embed[self.decoder_b.num_layers](output_memory["b"]) + output_proposals["b"]
            # [bs, \sum_{l=0}^{L-1} h_l * w_l, 8]

            topk_c, topk_b = self.two_stage_num_proposals
            topk_proposals_c = torch.topk(enc_outputs_class_c[..., 0], topk_c, dim=1)[1]  # TODO: 这里究竟是为什么呢
            # [bs, \sum_{l=0}^{L-1} h_l * w_l, num_classes_c] --> [bs, \sum_{l=0}^{L-1} h_l * w_l] --> [bs, num_queries_c]
            # topk_c == num_queries_c
            # torch.topk 函数返回的是值和索引组成的元组，所以[1]表示取了最大值对应的索引 
            
            topk_coords_unact_c = torch.gather(enc_outputs_coord_unact_c, 1, topk_proposals_c.unsqueeze(-1).repeat(1, 1, 4))
            # [bs, num_queries_c, 4]
            topk_coords_unact_c = topk_coords_unact_c.detach()
            reference_points_c = topk_coords_unact_c.sigmoid()
            init_reference_out_c = reference_points_c

            topk_proposals_b = torch.topk(enc_outputs_class_b[..., 0], topk_b, dim=1)[1]
            # [bs, \sum_{l=0}^{L-1} h_l * w_l, num_classes_b] --> [bs, \sum_{l=0}^{L-1} h_l * w_l] --> [bs, num_queries_b]
            topk_coords_unact_b = torch.gather(enc_outputs_coord_unact_b, 1, topk_proposals_b.unsqueeze(-1).repeat(1, 1, 8))
            # [bs, num_queries_b, 8]
            topk_coords_unact_b = topk_coords_unact_b.detach()
            tcub_x, tcub_y = torch.mean(topk_coords_unact_b[..., 0::2], -1), torch.mean(topk_coords_unact_b[..., 1::2], -1)
            reference_points_b = torch.stack((tcub_x, tcub_y), -1)  # [bs, num_queries_b, 2]
            reference_points_b = reference_points_b.sigmoid()
            init_reference_out_b = reference_points_b

            pos_trans_out_c = self.pos_trans_norm_c(self.pos_trans_c(self.get_proposal_pos_embed(topk_coords_unact_c)))
            query_embed_c, tgt_c = torch.split(pos_trans_out_c, c, dim=2)
            pos_trans_out_b = self.pos_trans_norm_b(self.pos_trans_b(self.get_proposal_pos_embed(topk_coords_unact_b)))
            query_embed_b, tgt_b = torch.split(pos_trans_out_b, c, dim=2)

            enc_outputs_class = {"c": enc_outputs_class_c, "b": enc_outputs_class_b}
            enc_outputs_coord_unact = {"c": enc_outputs_coord_unact_c, "b": enc_outputs_coord_unact_b}
        else:
            # 切分前：
            # query_embed["c"]: [num_queries_c, d_model*2]
            # query_embed["b"]: [num_queries_b, d_model*2]
            # 切分？之前把 query embedding 到了 d_model*2 维，所以可以沿列切分为 2 块
            query_embed_c, tgt_c = torch.split(query_embed["c"], c, dim=1)  # c == d_model
            query_embed_b, tgt_b = torch.split(query_embed["b"], c, dim=1)
            # 切分后：
            # query_embed_c: [num_queries_c, d_model]
            # query_embed_b: [num_queries_b, d_model]
            query_embed_c = query_embed_c.unsqueeze(0).expand(bs, -1, -1)   # [bs, num_queries_c, d_model]
            tgt_c = tgt_c.unsqueeze(0).expand(bs, -1, -1)  # -1: 对应维度的尺寸不变
            query_embed_b = query_embed_b.unsqueeze(0).expand(bs, -1, -1)   # [bs, num_queries_b, d_model]
            tgt_b = tgt_b.unsqueeze(0).expand(bs, -1, -1)
            reference_points_c = self.reference_points_c(query_embed_c).sigmoid()  # 直接用一个线性层学习出参考点，[bs, num_queries_c, 2]
            reference_points_b = self.reference_points_b(query_embed_b).sigmoid()  # 直接用一个线性层学习出参考点，[bs, num_queries_b, 2]
            init_reference_out_c, init_reference_out_b = reference_points_c, reference_points_b

        # dual decoders
        hs_c, inter_references_c = self.decoder_c(tgt_c, reference_points_c, memory, spatial_shapes, 
                                                  level_start_index, valid_ratios, query_embed_c, mask_flatten)
        hs_b, inter_references_b = self.decoder_b(tgt_b, reference_points_b, memory, spatial_shapes, 
                                                  level_start_index, valid_ratios, query_embed_b, mask_flatten)

        inter_references_out_c, inter_references_out_b = inter_references_c, inter_references_b
        
        hs = {"c": hs_c, "b": hs_b}
        init_reference_out = {"c": init_reference_out_c, "b": init_reference_out_b}
        inter_references_out = {"c": inter_references_out_c, "b": inter_references_out_b}

        if self.two_stage:
            return hs, init_reference_out, inter_references_out, enc_outputs_class, enc_outputs_coord_unact
        return hs, init_reference_out, inter_references_out, None, None


class DeformableTransformerEncoderLayer(nn.Module):
    def __init__(self,
                 d_model=256, d_ffn=1024,
                 dropout=0.1, activation="relu",
                 n_levels=4, n_heads=8, n_points=4):
        super().__init__()

        # self attention
        self.self_attn = MSDeformAttn(d_model, n_levels, n_heads, n_points)  # TODO
        self.dropout1 = nn.Dropout(dropout)
        self.norm1 = nn.LayerNorm(d_model)

        # ffn
        self.linear1 = nn.Linear(d_model, d_ffn)
        self.activation = _get_activation_fn(activation)
        self.dropout2 = nn.Dropout(dropout)
        self.linear2 = nn.Linear(d_ffn, d_model)
        self.dropout3 = nn.Dropout(dropout)
        self.norm2 = nn.LayerNorm(d_model)

    @staticmethod
    def with_pos_embed(tensor, pos):
        return tensor if pos is None else tensor + pos

    def forward_ffn(self, src):
        src2 = self.linear2(self.dropout2(self.activation(self.linear1(src))))
        src = src + self.dropout3(src2)
        src = self.norm2(src)
        return src

    def forward(self, src, pos, reference_points, spatial_shapes, level_start_index, padding_mask=None):
        """forward function of deformable transformer ENCODER LAYER

        Args:
            src:                [bs, \sum_{l=0}^{L-1} h_l * w_l, d_model]
            pos:                [bs, \sum_{l=0}^{L-1} h_l * w_l, d_model]
            reference_points:   [bs, \sum_{l=0}^{L-1} h_l * w_l, L, 2]
            spatial_shapes:     [L, 2]
            level_start_index:  [L,]
            padding_mask:       [bs, \sum_{l=0}^{L-1} h_l * w_l]
        """
        # self attention
        src2 = self.self_attn(self.with_pos_embed(src, pos), reference_points, src, spatial_shapes, level_start_index, padding_mask)
        src = src + self.dropout1(src2)
        src = self.norm1(src)

        # ffn
        src = self.forward_ffn(src)

        return src


class DeformableTransformerEncoder(nn.Module):
    def __init__(self, encoder_layer, num_layers):
        super().__init__()
        self.layers = _get_clones(encoder_layer, num_layers)
        self.num_layers = num_layers

    @staticmethod
    def get_reference_points(spatial_shapes, valid_ratios, device):
        reference_points_list = []
        for lvl, (H_, W_) in enumerate(spatial_shapes):  # 好像懂了，这个spatial_shapes应该是每层特征图片的长宽组成的列表之类的

            ref_y, ref_x = torch.meshgrid(torch.linspace(0.5, H_ - 0.5, H_, dtype=torch.float32, device=device),
                                          torch.linspace(0.5, W_ - 0.5, W_, dtype=torch.float32, device=device))
            """
            torch.meshgrid: https://blog.csdn.net/weixin_39504171/article/details/106356977
            torch.linspace(start, end, steps)，steps指分割的端点数
            实际上就是取了每个像素的中心点
            假设 H=2, W=3
            ref_y = [[0.5, 0.5, 0.5],
                     [1.5, 1.5, 1.5]]
            ref_x = [[0.5, 1.5, 2.5],
                     [0.5, 1.5, 2.5]]
            shape: [h_l, w_l]
            """
            # valid_ratios: [bs, L, 2]
            # valid_ratios[:, None, lvl, 1]: [bs, 1]
            # ref_y.reshape(-1)[None]: [h_l * w_l,] --> [1, h_l * w_l]
            # 感觉是用每张图片 padding 前的大小来对 meshgrid 做归一化？
            ref_y = ref_y.reshape(-1)[None] / (valid_ratios[:, None, lvl, 1] * H_)  # [bs, h_l * w_l]
            ref_x = ref_x.reshape(-1)[None] / (valid_ratios[:, None, lvl, 0] * W_)
            ref = torch.stack((ref_x, ref_y), -1)  # [bs, h_l * w_l, 2]    每个batch中每张图片在该层按照原有大小归一化后的网格坐标
            reference_points_list.append(ref)
        reference_points = torch.cat(reference_points_list, 1)  # [bs, \sum_{l=0}^{L-1} h_l * w_l, 2]
        # reference_points[:, :, None]: [bs, \sum_{l=0}^{L-1} h_l * w_l, 1, 2]
        # valid_ratios[:, None]: [bs, 1, L, 2]
        reference_points = reference_points[:, :, None] * valid_ratios[:, None]
        return reference_points  # [bs, \sum_{l=0}^{L-1} h_l * w_l, L, 2]

    def forward(self, src, spatial_shapes, level_start_index, valid_ratios, pos=None, padding_mask=None):
        """forward function of deformable transformer ENCODER

        Args:
            src:                [bs, \sum_{l=0}^{L-1} h_l * w_l, d_model]
            spatial_shapes:     [L, 2]
            level_start_index:  [L,]
            valid_ratios:       [bs, L, 2]
            pos:                [bs, \sum_{l=0}^{L-1} h_l * w_l, d_model]
            padding_mask:       [bs, \sum_{l=0}^{L-1} h_l * w_l]
        """
        output = src
        reference_points = self.get_reference_points(spatial_shapes, valid_ratios, device=src.device) # [bs, \sum_{l=0}^{L-1} h_l * w_l, L, 2]
        for _, layer in enumerate(self.layers):
            output = layer(output, pos, reference_points, spatial_shapes, level_start_index, padding_mask)

        return output


class DeformableTransformerDecoderLayer(nn.Module):
    def __init__(self, d_model=256, d_ffn=1024,
                 dropout=0.1, activation="relu",
                 n_levels=4, n_heads=8, n_points=4):
        super().__init__()

        # cross attention
        self.cross_attn = MSDeformAttn(d_model, n_levels, n_heads, n_points)
        self.dropout1 = nn.Dropout(dropout)
        self.norm1 = nn.LayerNorm(d_model)

        # self attention
        self.self_attn = nn.MultiheadAttention(d_model, n_heads, dropout=dropout)
        self.dropout2 = nn.Dropout(dropout)
        self.norm2 = nn.LayerNorm(d_model)

        # ffn
        self.linear1 = nn.Linear(d_model, d_ffn)
        self.activation = _get_activation_fn(activation)
        self.dropout3 = nn.Dropout(dropout)
        self.linear2 = nn.Linear(d_ffn, d_model)
        self.dropout4 = nn.Dropout(dropout)
        self.norm3 = nn.LayerNorm(d_model)

    @staticmethod
    def with_pos_embed(tensor, pos):
        return tensor if pos is None else tensor + pos

    def forward_ffn(self, tgt):
        tgt2 = self.linear2(self.dropout3(self.activation(self.linear1(tgt))))
        tgt = tgt + self.dropout4(tgt2)
        tgt = self.norm3(tgt)
        return tgt

    def forward(self, tgt, query_pos, reference_points, src, src_spatial_shapes, level_start_index, src_padding_mask=None):
        """forward function of deformable transformer DECODER LAYER

        Args:
            tgt:                    [bs, num_queries, d_model]
            query_pos:              [bs, num_queries, d_model]
            reference_points:       [bs, num_queries, L, 2]
            src:                    [bs, \sum_{l=0}^{L-1} h_l * w_l, d_model]
            src_spatial_shapes:     [L, 2]
            level_start_index:      [L,]
            src_padding_mask:       [bs, \sum_{l=0}^{L-1} h_l * w_l]

        Returns:
            _type_: _description_
        """
        # self attention
        q = k = self.with_pos_embed(tgt, query_pos)
        tgt2 = self.self_attn(q.transpose(0, 1), k.transpose(0, 1), tgt.transpose(0, 1))[0].transpose(0, 1)  # TODO: ???
        tgt = tgt + self.dropout2(tgt2)
        tgt = self.norm2(tgt)

        # cross attention
        tgt2 = self.cross_attn(self.with_pos_embed(tgt, query_pos),
                               reference_points,
                               src, src_spatial_shapes, level_start_index, src_padding_mask)
        tgt = tgt + self.dropout1(tgt2)
        tgt = self.norm1(tgt)

        # ffn
        tgt = self.forward_ffn(tgt)

        return tgt


class DeformableTransformerDecoder(nn.Module):
    def __init__(self, decoder_layer, num_layers, return_intermediate=False):
        super().__init__()
        self.layers = _get_clones(decoder_layer, num_layers)
        self.num_layers = num_layers
        self.return_intermediate = return_intermediate
        # hack implementation for iterative bounding box refinement and two-stage Deformable DETR
        self.bbox_embed = None
        self.class_embed_c = None
        self.point_embed = None
        self.class_embed_b = None

    def forward(self, tgt, reference_points, src, src_spatial_shapes, src_level_start_index, src_valid_ratios,
                query_pos=None, src_padding_mask=None):
        """The forward function of deformable transformer DECODER

        Args:
            tgt: the input queries of decoder                       [bs, num_queries, d_model]
            reference_points: the reference point of each query     [bs, num_queries, 2]
            src: the output features of encoder                     [bs, \sum_{l=0}^{L-1} h_l * w_l, d_model]
            src_spatial_shapes:                                     [L, 2]
            src_level_start_index:                                  [L,]
            src_valid_ratios:                                       [bs, L, 2]
            query_pos: (learned) query positional embedding         [bs, num_queries, d_model]
            src_padding_mask: the padding mask of encoder input     [bs, \sum_{l=0}^{L-1} h_l * w_l]

        Returns:
            output and reference points
        """
        output = tgt

        intermediate = []
        intermediate_reference_points = []
        for lid, layer in enumerate(self.layers):
            if reference_points.shape[-1] == 4:
                reference_points_input = reference_points[:, :, None] \
                                         * torch.cat([src_valid_ratios, src_valid_ratios], -1)[:, None]
            else:
                assert reference_points.shape[-1] == 2
                # reference_points[:, :, None]: [bs, num_queries, 1, 2]
                # src_valid_ratios[:, None]: [bs, 1, L, 2]
                reference_points_input = reference_points[:, :, None] * src_valid_ratios[:, None]
                # mini-batch 中每个 query 在每一层上的参考点（因为没对应的点，所以是学习而来的）
                # [bs, num_queries, L, 2]
            output = layer(output, query_pos, reference_points_input, src, src_spatial_shapes, src_level_start_index, src_padding_mask)
            # [bs, num_queries, d_model]

            # hack implementation for iterative bounding box refinement or control point refinement
            if self.bbox_embed is not None:
                tmp = self.bbox_embed[lid](output)
                if reference_points.shape[-1] == 4:
                    new_reference_points = tmp + inverse_sigmoid(reference_points)
                    new_reference_points = new_reference_points.sigmoid()
                else:
                    assert reference_points.shape[-1] == 2
                    new_reference_points = tmp
                    new_reference_points[..., :2] = tmp[..., :2] + inverse_sigmoid(reference_points)
                    new_reference_points = new_reference_points.sigmoid()
                reference_points = new_reference_points.detach()
            elif self.point_embed is not None:
                assert reference_points.shape[-1] == 2
                tmp = self.point_embed[lid](output)  # [bs, num_queries_b, 8]
                tmp_x, tmp_y = torch.mean(tmp[..., 0::2], -1), torch.mean(tmp[..., 1::2], -1)
                tmp = torch.stack((tmp_x, tmp_y), -1)
                new_reference_points = tmp + inverse_sigmoid(reference_points)
                new_reference_points = new_reference_points.sigmoid()
                

            if self.return_intermediate:
                intermediate.append(output)
                intermediate_reference_points.append(reference_points)

        if self.return_intermediate:
            # [L, bs, num_queries, d_model]   [L, bs, num_queries, 2]
            return torch.stack(intermediate), torch.stack(intermediate_reference_points)

        return output, reference_points


def _get_clones(module, N):
    return nn.ModuleList([copy.deepcopy(module) for _ in range(N)])


def _get_activation_fn(activation):
    """Return an activation function given a string"""
    if activation == "relu":
        return F.relu
    if activation == "gelu":
        return F.gelu
    if activation == "glu":
        return F.glu
    raise RuntimeError(F"activation should be relu/gelu, not {activation}.")


def build_deforamble_transformer(args):
    return DeformableTransformer(
        d_model=args.hidden_dim,  # default=256 (ignored below)
        nhead=args.nheads,  # 8
        num_encoder_layers=args.enc_layers,  # 6
        num_decoder_layers=args.dec_layers,  # 6
        dim_feedforward=args.dim_feedforward,  # 1024
        dropout=args.dropout,  # 0.1
        activation="relu",
        return_intermediate_dec=True,
        num_feature_levels=args.num_feature_levels,  # 4
        dec_n_points=args.dec_n_points,  # 4
        enc_n_points=args.enc_n_points,  # 4
        # two_stage 会同时影响 transformer 和 deformableDETR，而 with_box_refine 只影响 deformableDETR
        two_stage=args.two_stage,  # False
        two_stage_num_proposals=args.num_queries)  # (330, 100)
