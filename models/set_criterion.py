"""
Modules that computes the criterion
"""
import torch
import torch.nn.functional as F
from torch import nn

from util import box_ops
from util.misc import (nested_tensor_from_tensor_list,
                       accuracy, get_world_size, interpolate,
                       is_dist_avail_and_initialized)

from .segmentation import (dice_loss, sigmoid_focal_loss)
import copy


def _get_src_permutation_idx(indices):
    # permute predictions following indices
    # 将所有的 src 的索引连接在一起，并加上了对应 batch 的下标
    batch_idx = torch.cat([torch.full_like(src, i) for i, (src, _) in enumerate(indices)])  # [0,1,1,2,2,3,...]
    src_idx = torch.cat([src for (src, _) in indices])  # 将列表中分割的元组连接在一起
    return batch_idx, src_idx

def _get_tgt_permutation_idx(indices):
    # permute targets following indices
    # 将所有的 tgt 的索引连接在一起，并加上了对应 batch 的下标
    batch_idx = torch.cat([torch.full_like(tgt, i) for i, (_, tgt) in enumerate(indices)])
    tgt_idx = torch.cat([tgt for (_, tgt) in indices])
    return batch_idx, tgt_idx


class SetCriterion(nn.Module):
    def __init__(self, num_classes, matcher, weight_dict, losses, focal_alpha=0.25):
        """create the criterion

        Args:
            num_classes: tuple(int, int)
            matcher: a dict containing key 'c' and 'b'
            weight_dict: a dict containing key 'c' and 'b'
            losses: a dict containing key 'c' and 'b'
        """
        super(SetCriterion, self).__init__()
        self.weight_dict = weight_dict
        self.criterion_c = SetCriterionPart(num_classes[0], matcher['c'], weight_dict['c'], 
                                            losses['c'], mode='c', focal_alpha=focal_alpha)
        self.criterion_b = SetCriterionPart(num_classes[1], matcher['b'], weight_dict['b'], 
                                            losses['b'], mode='b', focal_alpha=focal_alpha)
    
    def forward(self, outputs, targets):
        # TODO: 貌似 outputs_without_aux 还可以优化
        losses_c = self.criterion_c(outputs, targets)
        losses_b = self.criterion_b(outputs, targets)
        return {"c": losses_c, "b": losses_b}


class SetCriterionPart(nn.Module):
    """ This class computes the loss (**CHARACTER or BEZIER PART**) for DETR.
    The process happens in two steps:
        1) we compute hungarian assignment between ground truth boxes and the outputs of the model
        2) we supervise each pair of matched ground-truth / prediction (supervise class and box)
    """
    def __init__(self, num_classes_x, matcher_x, weight_dict_x, losses_x, mode, focal_alpha=0.25):
        """ Create the character criterion.
        Parameters:
            num_classes_x: number of character or bezier curve object classes, 
                           **omitting** the special no-object category
            matcher_x: a module able to compute a matching between targets and proposals
            weight_dict_x: dict containing as key the names of the losses and as values their relative weight.
            losses_x: list of all the losses to be applied. See get_loss for list of available losses.
            focal_alpha: alpha in Focal Loss
        """
        super(SetCriterionPart, self).__init__()
        self.num_classes_x = num_classes_x
        self.matcher_x = matcher_x
        self.weight_dict_x = weight_dict_x
        self.losses_x = losses_x
        self.focal_alpha = focal_alpha
        assert mode in ('c', 'b')
        self.mode = mode

    def loss_labels(self, outputs, targets, indices_x, num_xx, log=True):
        """classification loss (NLL)
        targets dicts must contain the key "labels" containing a tensor of dim
        [nb_target_boxes] or [nb_target_bezier]
        """
        idx = _get_src_permutation_idx(indices_x)

        if self.mode == 'c':
            assert 'pred_logits' in outputs['char']
            src_logits = outputs['char']['pred_logits']  # [bs, num_queries_c, num_classes_c]
            target_classes_o = torch.cat([t["char"]["labels"][J] for t, (_, J) in zip(targets, indices_x)])
            # [\sum_{i=0}^{bs-1} num_target_boxes_i,]
        else:  # self.mode == 'b'
            assert 'pred_logits' in outputs['bezier']
            src_logits = outputs['bezier']['pred_logits']  # [bs, num_queries_b, num_classes_b]
            target_classes_o = torch.cat([t["bezier"]["labels"][J] for t, (_, J) in zip(targets, indices_x)])
            # # [\sum_{i=0}^{bs-1} num_target_bezier_i,]

        target_classes = torch.full(src_logits.shape[:2], self.num_classes_x,
                                    dtype=torch.int64, device=src_logits.device)
        # [bs, num_queries_c] or [bs, num_queries_b]
        # num_classes_c 这个数值应该是对应的空类？
        target_classes[idx] = target_classes_o

        target_classes_onehot = torch.zeros([src_logits.shape[0], src_logits.shape[1], src_logits.shape[2] + 1],
                                            dtype=src_logits.dtype, layout=src_logits.layout, device=src_logits.device)
        # [bs, num_queries_c, num_classes_c+1] or [bs, num_queries_b, num_classes_b+1]
        target_classes_onehot.scatter_(2, target_classes.unsqueeze(-1), 1)
        # scatter_ 的有关参考：https://yuyangyy.medium.com/understand-torch-scatter-b0fd6275331c

        target_classes_onehot = target_classes_onehot[:,:,:-1]  # 背景类为全0
        # [bs, num_queries_c, num_classes_c] or [bs, num_queries_b, num_classes_b]
        # 每个batch中每个query预测的结果对应的真实目标类别（以独热编码表示）
        loss_ce = sigmoid_focal_loss(src_logits, target_classes_onehot, num_xx, alpha=self.focal_alpha, gamma=2) * src_logits.shape[1]
        # TODO: 先不改这个关键字了
        losses = {'loss_ce': loss_ce}

        if log:
            # TODO this should probably be a separate loss, not hacked in this one here
            losses['class_error'] = 100 - accuracy(src_logits[idx], target_classes_o)[0]
        return losses

    @torch.no_grad()
    def loss_cardinality(self, outputs, targets, indices_x, num_xx):
        """ Compute the cardinality error, ie the absolute error in the number of predicted non-empty object
        This is not really a loss, it is intended for logging purposes only. It doesn't propagate gradients
        """
        if self.mode == 'c':
            pred_logits = outputs['char']['pred_logits']
            device = pred_logits.device
            tgt_lengths = torch.as_tensor([len(v['char']["labels"]) for v in targets], device=device)
        else:
            pred_logits = outputs['bezier']['pred_logits']
            device = pred_logits.device
            tgt_lengths = torch.as_tensor([len(v['bezier']["labels"]) for v in targets], device=device)
        # Count the number of predictions that are NOT "no-object" (which is the last class)
        card_pred = (pred_logits.argmax(-1) != pred_logits.shape[-1] - 1).sum(1)
        card_err = F.l1_loss(card_pred.float(), tgt_lengths.float())
        losses = {'cardinality_error': card_err}
        return losses

    def loss_boxes(self, outputs, targets, indices_c, num_boxes):
        """Compute the losses related to the bounding boxes, the L1 regression loss and the GIoU loss
           targets dicts must contain the key "boxes" containing a tensor of dim [nb_target_boxes, 4]
           The target boxes are expected in format (center_x, center_y, h, w), normalized by the image size.
        """
        assert 'pred_boxes' in outputs['char']
        idx = _get_src_permutation_idx(indices_c)
        # 根据 matcher 的结果找到 target 对应的预测 boxes
        src_boxes = outputs['char']['pred_boxes'][idx]  # [\sum_{i=0}^{bs-1} num_target_boxes_i, 4]
        target_boxes = torch.cat([t['char']['boxes'][i] for t, (_, i) in zip(targets, indices_c)], dim=0)

        loss_bbox = F.l1_loss(src_boxes, target_boxes, reduction='none')

        losses = {}
        losses['loss_bbox'] = loss_bbox.sum() / num_boxes

        # 取对角线元素计算 giou loss
        loss_giou = 1 - torch.diag(box_ops.generalized_box_iou(
            box_ops.box_cxcywh_to_xyxy(src_boxes),
            box_ops.box_cxcywh_to_xyxy(target_boxes)))
        losses['loss_giou'] = loss_giou.sum() / num_boxes
        return losses

    def loss_points(self, outputs, targets, indices_b, num_bezier):
        """Compute the losses related to the bounding boxes, the **L2** regression loss
           The target points are expected in format (x1, y1, x2, y2, x3, y3, x4, y4), 
           normalized by the image size.
        """
        assert 'pred_points' in outputs['bezier']
        idx = _get_src_permutation_idx(indices_b)
        # 根据 matcher 的结果找到 target 对应的预测 boxes
        src_points = outputs['bezier']['pred_points'][idx]  # [\sum_{i=0}^{bs-1} num_target_bezier_i, 8]
        target_points = torch.cat([t['bezier']['points'][i] for t, (_, i) in zip(targets, indices_b)], dim=0)

        loss_point = F.mse_loss(src_points, target_points, reduction='none')

        losses = {}
        losses['loss_point'] = loss_point.sum() / num_bezier

        return losses

    def loss_masks(self, outputs, targets, indices_c, num_boxes):
        """Compute the losses related to the masks: the focal loss and the dice loss.
           targets dicts must contain the key "masks" containing a tensor of dim [nb_target_boxes, h, w]
        """
        assert "pred_masks" in outputs['char']

        src_idx = _get_src_permutation_idx(indices_c)
        tgt_idx = _get_tgt_permutation_idx(indices_c)

        src_masks = outputs["char"]["pred_masks"]

        # TODO use valid to mask invalid areas due to padding in loss
        target_masks, valid = nested_tensor_from_tensor_list([t["char"]["masks"] for t in targets]).decompose()
        target_masks = target_masks.to(src_masks)

        src_masks = src_masks[src_idx]
        # upsample predictions to the target size
        src_masks = interpolate(src_masks[:, None], size=target_masks.shape[-2:],
                                mode="bilinear", align_corners=False)
        src_masks = src_masks[:, 0].flatten(1)

        target_masks = target_masks[tgt_idx].flatten(1)

        losses = {
            "loss_mask": sigmoid_focal_loss(src_masks, target_masks, num_boxes),
            "loss_dice": dice_loss(src_masks, target_masks, num_boxes),
        }
        return losses

    def get_loss(self, loss, outputs, targets, indices_x, num_xx, **kwargs):
        if self.mode == 'c':
            loss_map = {
                'labels': self.loss_labels,
                'cardinality': self.loss_cardinality,
                'boxes': self.loss_boxes,
                'masks': self.loss_masks
            }
        else:
            loss_map = {
                'labels': self.loss_labels,
                'cardinality': self.loss_cardinality,
                'points': self.loss_points,
            }
        assert loss in loss_map, f'do you really want to compute {loss} loss?'
        return loss_map[loss](outputs, targets, indices_x, num_xx, **kwargs)  # 返回某个字典用来 update

    def forward(self, outputs, targets):
        """ This performs the loss computation.
        Parameters:
             outputs: dict of dicts, containing key 'char' and 'bezier'
             targets: list of dicts, such that len(targets) == batch_size.
                      The expected keys in each dict depends on the losses applied, see each loss' doc
        """
        # TODO: 加上aux_loss后还需要修改
        outputs_without_aux = outputs
        # outputs_without_aux = {k: v for k, v in outputs.items() if k != 'aux_outputs' and k != 'enc_outputs'}
        # 'pred_logits': [bs, num_queries, num_classes]
        # 'pred_boxes': [bs, num_queries, 4]

        # Retrieve the matching between the outputs of the last layer and the targets
        indices_x = self.matcher_x(outputs_without_aux, targets)

        # Compute the average number of target boxes or bezier curves accross all nodes, for normalization purposes
        if self.mode == 'c':
            num_xx = sum(len(t["char"]["labels"]) for t in targets)
            _device = next(iter(outputs["char"].values())).device
            # mini-batch 中所有 target 的 box 的数量总和
        else:
            num_xx = sum(len(t["bezier"]["labels"]) for t in targets)
            _device = next(iter(outputs["bezier"].values())).device
            # mini-batch 中所有 target 的 bezier curve 的数量总和
        num_xx = torch.as_tensor([num_xx], dtype=torch.float, device=_device)
        if is_dist_avail_and_initialized():
            torch.distributed.all_reduce(num_xx)
        num_xx = torch.clamp(num_xx / get_world_size(), min=1).item()

        # Compute all the requested losses
        losses_x = {}
        for loss in self.losses_x:
            # 最基本的3个：labels, boxes, cardinality
            kwargs = {}
            losses_x.update(self.get_loss(loss, outputs, targets, indices_x, num_xx, **kwargs))

        # In case of auxiliary losses, we repeat this process with the output of each intermediate layer.
        # TODO: 还需要修改
        if 'aux_outputs' in outputs:
            for i, aux_outputs in enumerate(outputs['aux_outputs']):
                indices_c = self.matcher(aux_outputs, targets)
                for loss in self.losses_x:
                    if loss == 'masks':
                        # Intermediate masks losses are too costly to compute, we ignore them.
                        continue
                    kwargs = {}
                    if loss == 'labels':
                        # Logging is enabled only for the last layer
                        kwargs['log'] = False
                    l_dict = self.get_loss(loss, aux_outputs, targets, indices_c, num_xx, **kwargs)
                    l_dict = {k + f'_{i}': v for k, v in l_dict.items()}
                    losses_x.update(l_dict)

        # TODO: 还需要修改
        if 'enc_outputs' in outputs:
            enc_outputs = outputs['enc_outputs']
            bin_targets = copy.deepcopy(targets)
            for bt in bin_targets:
                bt['labels'] = torch.zeros_like(bt['labels'])
            indices_c = self.matcher(enc_outputs, bin_targets)
            for loss in self.losses_x:
                if loss == 'masks':
                    # Intermediate masks losses are too costly to compute, we ignore them.
                    continue
                kwargs = {}
                if loss == 'labels':
                    # Logging is enabled only for the last layer
                    kwargs['log'] = False
                l_dict = self.get_loss(loss, enc_outputs, bin_targets, indices_c, num_xx, **kwargs)
                l_dict = {k + f'_enc': v for k, v in l_dict.items()}
                losses_x.update(l_dict)

        return losses_x


def build_set_criterion(num_classes, matcher, weight_dict, losses, focal_alpha=0.25):
    return SetCriterion(num_classes, matcher, weight_dict, losses, focal_alpha)
