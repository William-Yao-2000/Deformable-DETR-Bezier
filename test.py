import torch
from scipy.optimize import linear_sum_assignment
import math
import torch.nn.functional as F

# a = torch.tensor([[2, 2], [3, 5], [9, 10]])
# print(a.shape)
# print(a[None].shape)
# print(a[None])


# mask1 = torch.tensor([[1,1,0,0],[1,1,0,0],[1,1,0,0],[0,0,0,0]]).unsqueeze(0)
# mask2 = torch.tensor([[1,0,0,0],[1,0,0,0],[0,0,0,0],[0,0,0,0]]).unsqueeze(0)
# mask3 = torch.tensor([[1,1,1,0],[1,1,1,0],[0,0,0,0],[0,0,0,0]]).unsqueeze(0)
# masks = torch.cat((mask1, mask2, mask3), dim=0)
# print(masks.shape)
# print(masks)

# _, H, W = masks.shape  # [bs, h, w]
# # print(masks[:,:,0])
# valid_H = torch.sum(masks[:, :, 0], 1) # [bs, ]
# valid_W = torch.sum(masks[:, 0, :], 1)
# # print(valid_H, valid_W)
# valid_ratio_h = valid_H.float() / H
# valid_ratio_w = valid_W.float() / W
# # print(valid_ratio_h, valid_ratio_w)
# # print(valid_ratio_h.shape)
# valid_ratios = torch.stack([valid_ratio_w, valid_ratio_h], -1)
# valid_ratios = valid_ratios.unsqueeze(1)
# print(valid_ratios.size())


# """
# Positional encoding
# """
# mask = torch.tensor([[[1, 1, 0],
#                       [1, 1, 0],
#                       [0, 0, 0],
#                       [0, 0, 0]],
#                      [[1, 1, 1],
#                       [1, 1, 1],
#                       [1, 1, 1],
#                       [0, 0, 0]],])
# assert mask is not None
# not_mask = mask
# y_embed = not_mask.cumsum(1, dtype=torch.float32)  # [bs, h, w]
# x_embed = not_mask.cumsum(2, dtype=torch.float32)  # [bs, h, w]
# print(y_embed)

# eps = 1e-6
# scale = 2 * math.pi
# y_embed = (y_embed - 0.5) / (y_embed[:, -1:, :] + eps) * scale
# x_embed = (x_embed - 0.5) / (x_embed[:, :, -1:] + eps) * scale
# print(y_embed)

# num_pos_feats, temperature = 128, 10000
# dim_t = torch.arange(num_pos_feats, dtype=torch.float32)  # [num_pos_feats,]
# dim_t = temperature ** (2 * (dim_t // 2) / num_pos_feats)  # [num_pos_feats,]
# print(dim_t.shape)

# pos_x = x_embed[:, :, :, None] / dim_t
# pos_y = y_embed[:, :, :, None] / dim_t
# print(pos_y.shape)
# pos_x = torch.stack((pos_x[:, :, :, 0::2].sin(), pos_x[:, :, :, 1::2].cos()), dim=4).flatten(3)
# pos_y = torch.stack((pos_y[:, :, :, 0::2].sin(), pos_y[:, :, :, 1::2].cos()), dim=4).flatten(3)
# print(pos_x[:, :, :, 0::2].sin().shape)
# pos = torch.cat((pos_y, pos_x), dim=3).permute(0, 3, 1, 2)  # [bs, c, h, w]




# ref_y, ref_x = torch.meshgrid(torch.linspace(0.5, H - 0.5, H, dtype=torch.float32),
#                               torch.linspace(0.5, W - 0.5, W, dtype=torch.float32))
# """
# torch.meshgrid: https://blog.csdn.net/weixin_39504171/article/details/106356977
# torch.linspace(start, end, steps)，steps指分割的端点数
# 实际上就是取了每个像素的中心点
# 假设 H=2, W=3
# ref_y = [[0.5, 0.5, 0.5],
#          [1.5, 1.5, 1.5]]
# ref_x = [[0.5, 1.5, 2.5],
#          [0.5, 1.5, 2.5]]
# """
# print(valid_ratios[:, None, 0, 1].shape)
# print(ref_y.reshape(-1)[None].shape)
# ref_y = ref_y.reshape(-1)[None] / (valid_ratios[:, None, 0, 1] * H)
# ref_x = ref_x.reshape(-1)[None] / (valid_ratios[:, None, 0, 0] * W)
# ref = torch.stack((ref_x, ref_y), -1)
# print(ref.shape)

# reference_points = ref[:, :, None] * valid_ratios[:, None]
# print(ref[:, :, None].shape)
# print(valid_ratios[:, None].shape)
# print(reference_points.shape)


# input_spatial_shapes = torch.tensor(((100,200),(50,100),(25,50)))
# offset_normalizer = torch.stack([input_spatial_shapes[..., 1], input_spatial_shapes[..., 0]], -1)
# print(offset_normalizer)


# out_prob = torch.tensor([[0.2, 0.5, 0.25, 0.05],[0.1, 0.6, 0.2, 0.1],[0.3, 0.4, 0.22, 0.08],[0.6, 0.1, 0.23, 0.07],[0.67, 0.03, 0.2, 0.1],[0.45, 0.25, 0.21, 0.09]])  # [bs * num_queries, num_classes]

# # Also concat the target labels
# tgt_ids = torch.tensor([1, 1, 1, 0, 0])  # [\sum_{i=0}^{bs-1} num_target_boxes_i,]

# # Compute the classification cost.
# # alpha-focal loss
# alpha = 0.25
# gamma = 2.0
# neg_cost_class = (1 - alpha) * (out_prob ** gamma) * (-(1 - out_prob + 1e-8).log())
# print(neg_cost_class.shape)
# pos_cost_class = alpha * ((1 - out_prob) ** gamma) * (-(out_prob + 1e-8).log())
# print(pos_cost_class)
# print(pos_cost_class[:, tgt_ids])
# cost_class = pos_cost_class[:, tgt_ids] - neg_cost_class[:, tgt_ids]


# C = torch.tensor([[0.2, 0.5, 0.25, 0.05],
#                   [0.1, 0.6, 0.2, 0.1],
#                   [0.3, 0.4, 0.22, 0.08],
#                   [0.6, 0.1, 0.23, 0.07],
#                   [0.67, 0.03, 0.2, 0.1],
#                   [0.45, 0.25, 0.21, 0.09]])  # [bs * num_queries, \sum_{i=0}^{bs-1} num_target_boxes_i]  1+3
# sizes = [1, 3]
# C = C.view(2, 3, -1).cpu()
# # print(C[(0,1,1,1),(1,0,1,2)])
# # topk_values, topk_indexes = torch.topk(C.view(C.shape[0], -1), 5, dim=1)
# # print(topk_values)
# # print(topk_indexes)

# print(C.split(sizes, -1)[0].shape)
# index = linear_sum_assignment(C.split(sizes, -1)[0][0])
# print(index)
# indices = [linear_sum_assignment(c[i]) for i, c in enumerate(C.split(sizes, -1))]
# print(indices)
# indices = [(torch.as_tensor(i, dtype=torch.int64), torch.as_tensor(j, dtype=torch.int64)) for i, j in indices]
# print(indices)

# batch_idx = torch.cat([torch.full_like(src, i) for i, (src, _) in enumerate(indices)])
# src_idx = torch.cat([src for (src, _) in indices])
# print(batch_idx)
# print(src_idx)
# tgt_idx = torch.cat([tgt for (_, tgt) in indices])
# print(tgt_idx)

# path = "./data/synthtext/SynthText/8/ballet_3_0.jpg"
# dir_num, file_name = path.split('/')[-2:]
# file_name = file_name[:-4]
# print(dir_num, file_name)


# spatial_shapes = torch.tensor([[224, 224], [112, 112], [56, 56], [28, 28]])
# print(spatial_shapes.shape)
# print(spatial_shapes.new_zeros((1, )))
# print(spatial_shapes.prod(1))
# level_start_index = torch.cat((spatial_shapes.new_zeros((1, )), spatial_shapes.prod(1).cumsum(0)[:-1]))
# print(level_start_index)


# bs, num_queries, num_classes = 2, 6, 5
# src = torch.randn(bs, num_queries, num_classes)
# tgt = torch.randint(low=0, high=num_classes, size=(bs, num_queries))
# tgt = torch.LongTensor(tgt)
# tgt_onehot = F.one_hot(tgt, num_classes=num_classes).float()
# print(src)
# print(tgt_onehot)
# ce_loss = F.binary_cross_entropy_with_logits(src, tgt_onehot, reduction="none")
# print(ce_loss)


# from PIL import Image
# img = Image.open("vis/gt_temp/vis-8-ballet_3_0.png").convert('RGB')
# print(img)


box_num_sum = 18
src_boxes = torch.rand(box_num_sum, 4)
tgt_boxes = torch.rand(box_num_sum, 4)
loss_bbox = F.l1_loss(src_boxes, tgt_boxes, reduction='none')
print(loss_bbox.shape)