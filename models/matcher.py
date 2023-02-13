# ------------------------------------------------------------------------
# Deformable DETR
# Copyright (c) 2020 SenseTime. All Rights Reserved.
# Licensed under the Apache License, Version 2.0 [see LICENSE for details]
# ------------------------------------------------------------------------
# Modified from DETR (https://github.com/facebookresearch/detr)
# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
# ------------------------------------------------------------------------

"""
Modules to compute the matching cost and solve the corresponding LSAP.
"""
import torch
from scipy.optimize import linear_sum_assignment
from torch import nn

from util.box_ops import box_cxcywh_to_xyxy, generalized_box_iou


class HungarianMatcherChar(nn.Module):
    """This class computes an assignment **(CHARACTER PART)** between the targets and 
       the predictions of the network.

    For efficiency reasons, the targets don't include the no_object. Because of this, in general,
    there are more predictions than targets. In this case, we do a 1-to-1 matching of the best predictions,
    while the others are un-matched (and thus treated as non-objects).
    """

    def __init__(self,
                 cost_class_c: float = 1,
                 cost_bbox: float = 1,
                 cost_giou: float = 1):
        """Creates the matcher

        Params:
            cost_class_c: This is the relative weight of the character classification error in the matching cost
            cost_bbox: This is the relative weight of the L1 error of the bounding box coordinates in the matching cost
            cost_giou: This is the relative weight of the giou loss of the bounding box in the matching cost
        """
        super(HungarianMatcherChar, self).__init__()
        self.cost_class_c = cost_class_c
        self.cost_bbox = cost_bbox
        self.cost_giou = cost_giou
        assert cost_class_c != 0 or cost_bbox != 0 or cost_giou != 0, "all costs cant be 0"

    def forward(self, outputs, targets):
        """ Performs the matching

        Params:
            outputs: This is a dict that contains at least these entries:
                "char": a dict which contains the character prediction results
                    "pred_logits": Tensor of dim [batch_size, num_queries_c, num_classes_c] with the classification logits
                    "pred_boxes": Tensor of dim [batch_size, num_queries_c, 4] with the predicted box coordinates
                "bezier": a dict which contains the bezier curve prediction results
                    "pred_logits": Tensor of dim [batch_size, num_queries_b, num_classes_b] with the classification logits
                    "pred_points": Tensor of dim [batch_size, num_queries_b, 8] with the predicted control points coordinates

            targets: This is a list of targets (len(targets) = batch_size), where each target is a dict containing
                     at least these entries:
                "char": a dict which contains the character annotations in a image
                    "labels": Tensor of dim [num_target_boxes] (where num_target_boxes is the number of ground-truth
                            character boxes in the target) containing the character class labels
                    "boxes": Tensor of dim [num_target_boxes, 4] containing the target box coordinates
                            (normalized) (cx, cy, w, h)
                "bezier": a dict which contains the bezier curve annotations in a image
                    "labels": Tensor of dim [num_bezier_curves] containing the bezier curve class labels
                    "points": Tensor of dim [num_bezier_curves, 4] containing the control point coordinates
                            (normalized) (x1, y1, x2, y2, x3, y3, x4, y4)

        Returns:
            A list of size batch_size, containing tuples of (index_i, index_j) where:
                - index_i is the indices of the selected predictions (in order)
                - index_j is the indices of the corresponding selected targets (in order)
            For each batch element, it holds:
                len(index_i) = len(index_j) = min(num_queries_c, num_target_boxes)
        """
        with torch.no_grad():
            bs, num_queries_c = outputs["char"]["pred_logits"].shape[:2]

            # We flatten to compute the cost matrices in a batch
            out_prob_c = outputs["char"]["pred_logits"].flatten(0, 1).sigmoid()  # [bs * num_queries_c, num_classes_c]
            out_bbox = outputs["char"]["pred_boxes"].flatten(0, 1)  # [bs * num_queries_c, 4]

            # Also concat the target labels and boxes
            tgt_ids_c = torch.cat([v["char"]["labels"] for v in targets])  # [\sum_{i=0}^{bs-1} num_target_boxes_i,]
            tgt_bbox = torch.cat([v["char"]["boxes"] for v in targets])  # [\sum_{i=0}^{bs-1} num_target_boxes_i, 4]

            # Compute the classification cost.
            # alpha-focal loss
            alpha = 0.25
            gamma = 2.0
            neg_cost_class_c = (1 - alpha) * (out_prob_c ** gamma) * (-(1 - out_prob_c + 1e-8).log())
            pos_cost_class_c = alpha * ((1 - out_prob_c) ** gamma) * (-(out_prob_c + 1e-8).log())
            # cost_class_c[i, j]: out_prob_c 中第 i 个 query 在 tgt 中第 j 个字符目标对应的类上的二分类 focal loss
            # 感觉是将概率看作 num_classes_c 个二分类器啊
            cost_class_c = pos_cost_class_c[:, tgt_ids_c] - neg_cost_class_c[:, tgt_ids_c]
            # [bs * num_queries_c, \sum_{i=0}^{bs-1} num_target_boxes_i]

            # Compute the L1 cost between boxes
            # cost_bbox[i, j]: out_bbox 中第 i 个元素到 tgt_bbox 中第 j 个元素的 L1 距离
            cost_bbox = torch.cdist(out_bbox, tgt_bbox, p=1)  # [bs * num_queries_c, \sum_{i=0}^{bs-1} num_target_boxes_i]

            # Compute the giou cost betwen boxes
            cost_giou = -generalized_box_iou(box_cxcywh_to_xyxy(out_bbox),
                                             box_cxcywh_to_xyxy(tgt_bbox))  # [bs * num_queries_c, \sum_{i=0}^{bs-1} num_target_boxes_i]

            # Final cost matrix
            C = self.cost_bbox * cost_bbox + self.cost_class_c * cost_class_c + self.cost_giou * cost_giou
            C = C.view(bs, num_queries_c, -1).cpu()  # [bs, num_queries_c, \sum_{i=0}^{bs-1} num_target_boxes_i]

            sizes_c = [len(v["char"]["boxes"]) for v in targets]
            # 行和列进行匹配，使得匹配的二者对应元素最小
            indices_c = [linear_sum_assignment(c[i]) for i, c in enumerate(C.split(sizes_c, -1))]
            # c: [bs, num_queries_c, num_target_boxes_i]
            # c[i]: [num_queries_c, num_target_boxes_i]
            # 因为要将第 i 个预测的 batch 与第 i 个 target 的 batch 对应起来
            return [(torch.as_tensor(i, dtype=torch.int64), torch.as_tensor(j, dtype=torch.int64)) for i, j in indices_c]
            # result[k]: 第 k 个 batch 中预测结果与真实目标的对应关系，形状为 [num_target_boxes_k, 2]
            # e.g. result[1] == (tensor[0, 1, 3], tensor[2, 0, 1])


class HungarianMatcherBezier(nn.Module):
    """This class computes an assignment **(BEZIER PART)** between the targets and 
       the predictions of the network.
    """

    def __init__(self,
                 cost_class_b: float = 1,
                 cost_point: float = 1,):
        """Creates the matcher

        Params:
            cost_class_b: This is the relative weight of the bezier curve classification error in the matching cost
            cost_point: This is the relative weight of the **L2** error of the control point coordinates in the matching cost
        """
        super(HungarianMatcherBezier, self).__init__()
        self.cost_class_b = cost_class_b
        self.cost_point = cost_point
        assert cost_class_b != 0 or cost_point != 0, "all costs cant be 0"

    def forward(self, outputs, targets):
        """ Performs the matching

        Params:
            same as class HungarianMatcherChar

        Returns:
            A list of size batch_size, containing tuples of (index_i, index_j) where:
                - index_i is the indices of the selected predictions (in order)
                - index_j is the indices of the corresponding selected targets (in order)
            For each batch element, it holds:
                len(index_i) = len(index_j) = min(num_queries_b, num_target_bezier)
        """
        with torch.no_grad():
            bs, num_queries_b = outputs["bezier"]["pred_logits"].shape[:2]

            # We flatten to compute the cost matrices in a batch
            out_prob_b = outputs["bezier"]["pred_logits"].flatten(0, 1).sigmoid()  # [bs * num_queries_b, num_classes_b]
            out_point = outputs["bezier"]["pred_points"].flatten(0, 1)  # [bs * num_queries_b, 8]

            # Also concat the target labels and points
            tgt_ids_b = torch.cat([v["bezier"]["labels"] for v in targets])  # [\sum_{i=0}^{bs-1} num_target_bezier_i,]
            tgt_point = torch.cat([v["bezier"]["points"] for v in targets])  # [\sum_{i=0}^{bs-1} num_target_bezier_i, 8]

            # Compute the classification cost.
            # alpha-focal loss
            alpha = 0.25
            gamma = 2.0
            neg_cost_class_b = (1 - alpha) * (out_prob_b ** gamma) * (-(1 - out_prob_b + 1e-8).log())
            pos_cost_class_b = alpha * ((1 - out_prob_b) ** gamma) * (-(out_prob_b + 1e-8).log())
            # cost_class_b[i, j]: out_prob_b 中第 i 个 query 在 tgt 中第 j 个贝塞尔曲线目标对应的类上的二分类 focal loss
            cost_class_b = pos_cost_class_b[:, tgt_ids_b] - neg_cost_class_b[:, tgt_ids_b]
            # [bs * num_queries_b, \sum_{i=0}^{bs-1} num_target_bezier_i]

            # Compute the L2 cost between control points
            # cost_point[i, j]: out_point 中第 i 个元素到 tgt_point 中第 j 个元素的 **L2** 距离
            cost_point = torch.cdist(out_point, tgt_point, p=2)  # [bs * num_queries_b, \sum_{i=0}^{bs-1} num_target_bezier_i]

            # Final cost matrix
            C = self.cost_point * cost_point + self.cost_class_b * cost_class_b
            C = C.view(bs, num_queries_b, -1).cpu()  # [bs, num_queries_b, \sum_{i=0}^{bs-1} num_target_bezier_i]

            sizes_b = [len(v["bezier"]["points"]) for v in targets]
            # 行和列进行匹配，使得匹配的二者对应元素最小
            indices_b = [linear_sum_assignment(c[i]) for i, c in enumerate(C.split(sizes_b, -1))]
            # c: [bs, num_queries_b, num_target_bezier_i]
            # c[i]: [num_queries_b, num_target_bezier_i]
            # 因为要将第 i 个预测的 batch 与第 i 个 target 的 batch 对应起来
            return [(torch.as_tensor(i, dtype=torch.int64), torch.as_tensor(j, dtype=torch.int64)) for i, j in indices_b]
            # result[k]: 第 k 个 batch 中预测结果与真实目标的对应关系，形状为 [num_target_bezier_k, 2]
            # e.g. result[0] == (tensor[0, 1, 3], tensor[2, 0, 1])


def build_matcher(args):
    matcher_c = HungarianMatcherChar(cost_class_c=args.set_cost_class_c,
                                cost_bbox=args.set_cost_bbox,
                                cost_giou=args.set_cost_giou)
    matcher_b = HungarianMatcherBezier(cost_class_b=args.set_cost_class_b,
                                 cost_point=args.set_cost_point)
    return {"c": matcher_c, "b": matcher_b}
