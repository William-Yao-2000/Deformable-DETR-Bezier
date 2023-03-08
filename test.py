import torch
# from scipy.optimize import linear_sum_assignment
# import math
import torch.nn.functional as F
import bisect
import numpy as np
import cv2

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


# box_num_sum = 18
# src_boxes = torch.rand(box_num_sum, 4)
# tgt_boxes = torch.rand(box_num_sum, 4)
# loss_bbox = F.l1_loss(src_boxes, tgt_boxes, reduction='none')
# print(loss_bbox.shape)


# lst = [3, 2]
# print(lst*4)


# cropped_boxes = torch.randn(3, 2, 2)
# print(cropped_boxes)
# keep_char = torch.all(cropped_boxes[:, 1, :] > cropped_boxes[:, 0, :], dim=1)
# print(keep_char.shape)
# print(keep_char)

# cropped_points = torch.randn(4, 8)
# # keep_points = torch.all(0 <= cropped_points[:, 0::7] <= 0.5 and 0 <= cropped_points[:, 1::7] <= 0.5, dim=1)
# print(cropped_points)
# xs, ys = cropped_points[:, 0::2], cropped_points[:, 1::2]
# print(xs)
# # print(y1_y4)
# keep_x = torch.logical_and(torch.all(xs >= -1, dim=1), torch.all(xs <= 1, dim=1))
# print(keep_x)


# dct = {}
# dct['image_id'] = 1
# dct['char'] = {"area": 123}
# print(dct)
# dct_char = dct['char']
# dct_char['area'] = 456
# print(dct)


# points = torch.randn(5, 8)
# print(points)
# points = points * torch.as_tensor([-1, 1]*4)
# print(points)


# import random
# sizes = [400, 600, 800, 1000]
# print(random.choice(sizes))

# w, h = 800, 600
# x = torch.tensor([w, h]*4, dtype=torch.float32)
# print(x)


# tgt = {"char": [1], "bezier": [100]}

# def handle(tgt):
#     for k, v in tgt.items():
#         v[0] += 1 if v[0] == 1 else 0
# handle(tgt)
# print(tgt)


# num_queries = (330, 100)
# print(num_queries)
# print(sum(num_queries))


# mode = 'b'
# xx = 'char' if mode == 'c' else 'bezier'
# print(xx)


# tmp = torch.arange(48, dtype=torch.float32).view(2, 3, 8)
# print(tmp)
# tmp_x, tmp_y = torch.mean(tmp[..., 0::2], -1), torch.mean(tmp[..., 1::2], -1)
# print(tmp_x)
# print(tmp_y)
# tmp = torch.stack((tmp_x, tmp_y), -1)
# print(tmp.shape)
# print(tmp)

# tmp_cat = torch.cat([tmp]*4, -1)
# print(tmp_cat.shape)
# print(tmp_cat)


# a = torch.ones(4, 4)
# b = torch.randn(4, 1)
# print(a)
# print(b)
# print(a*b)


# enc_outputs_class = torch.randn(2, 5, 10)
# print(enc_outputs_class)
# topk = 2
# topk_proposals = torch.topk(enc_outputs_class[..., 0], topk, dim=1)[1]

# topk_class = torch.gather(enc_outputs_class, 1, topk_proposals.unsqueeze(-1).repeat(1, 1, 10))
# print(topk_class)


# grid = torch.randn(2, 3, 4, 2)
# result = torch.cat([grid]*4, -1).view(2, -1, 8)
# print(result.shape)


# def _get_src_permutation_idx(indices):
#     # permute predictions following indices
#     # 将所有的 src 的索引连接在一起，并加上了对应 batch 的下标
#     batch_idx = torch.cat([torch.full_like(src, i) for i, (src, _) in enumerate(indices)])  # [0,1,1,2,2,3,...]
#     src_idx = torch.cat([src for (src, _) in indices])  # 将列表中分割的元组连接在一起
#     return batch_idx, src_idx

# indices = [(torch.tensor([0,1,3]), torch.tensor([2,0,1])),
#            (torch.tensor([2,0]), torch.tensor([0,1]))]
# idx = _get_src_permutation_idx(indices)
# print(idx)

# num_classes = 6  # dataset class range: 1~5
# num_queries = 4
# src_logits = torch.randn(2, num_queries, num_classes)
# print(src_logits.shape)
# targets = list()
# targets.append({"char": {"labels": torch.tensor([3,4,5])} })
# targets.append({"char": {"labels": torch.tensor([1,4])} })
# target_classes_o = torch.cat([t["char"]["labels"][J] for t, (_, J) in zip(targets, indices)])
# print(target_classes_o)
# target_classes = torch.full(src_logits.shape[:2], 0,
#                             dtype=torch.int64, device=src_logits.device)
# # [bs, num_queries_c] or [bs, num_queries_b]
# print(target_classes)
# # num_classes_c 这个数值应该是对应的空类？ TODO: 好像有问题！
# target_classes[idx] = target_classes_o
# print(target_classes)

# target_classes_onehot = torch.zeros([src_logits.shape[0], src_logits.shape[1], src_logits.shape[2] + 1])
# # [bs, num_queries_c, num_classes_c+1] or [bs, num_queries_b, num_classes_b+1]
# print("\nNEXT tensors are target_class_onehot:")
# print(target_classes_onehot)
# print(target_classes_onehot.shape)
# target_classes_onehot.scatter_(2, target_classes.unsqueeze(-1), 1)
# # scatter_ 的有关参考：https://yuyangyy.medium.com/understand-torch-scatter-b0fd6275331c
# print(target_classes_onehot)
# print(target_classes_onehot.shape)

# target_classes_onehot = target_classes_onehot[:,:,:-1]  # 背景类为全0
# # [bs, num_queries_c, num_classes_c] or [bs, num_queries_b, num_classes_b]
# # 每个batch中每个query预测的结果对应的真实目标类别（以独热编码表示）
# print(target_classes_onehot)
# print(target_classes_onehot.shape)


"""
麻了，好像不能并行操作？
"""
# bs_range = torch.arange(3)
# num_boxes_range = torch.arange(5)
# grid_bs, grid_nb = torch.meshgrid(bs_range, num_boxes_range, indexing='ij')
# grid_idx = torch.stack((grid_bs, grid_nb), dim=-1)
# # print(grid_bs)
# # print(grid_nb)
# print(grid_idx.shape)
# # print(grid_idx)
# torch.manual_seed(1)
# boxes = torch.rand(3, 5, 4)  # [bs, num_boxes, 4]
# boxes_with_idx = torch.cat((boxes, grid_idx), dim=-1)
# print(boxes_with_idx)
# x1, x2, y1, y2 = boxes.unbind(-1)
# # print("")
# print(x1)
# print(x1.shape)
# sorted_x1_idx = torch.argsort(x1, dim=-1, descending=False).unsqueeze(-1).expand(-1, -1, boxes_with_idx.shape[2])
# sorted_x1_idx = torch.LongTensor(sorted_x1_idx)
# # print("")
# print(sorted_x1_idx.shape)
# print(sorted_x1_idx)
# # new_grid_idx_x1 = torch.cat((grid_idx, sorted_x1_idx), dim=-1)
# # print(new_grid_idx_x1)
# # new_boxes_x1 = boxes[new_grid_idx_x1]
# print(torch.gather(boxes_with_idx, 1, sorted_x1_idx)[..., [0,4,5]])
# # print(torch.gather(boxes_with_idx, 1, sorted_x1_idx))
# x = 0.5
# print(bisect.bisect_left)  # 呜呜呜


# boxes_idx = torch.arange(5).unsqueeze(-1)
# torch.manual_seed(1)
# boxes, _ = torch.rand(5, 4).sort(-1)  # [num_boxes, 4]
# boxes = torch.tensor(boxes, dtype=torch.float32)
# boxes_with_idx = torch.cat((boxes, boxes_idx), dim=-1)
# print(boxes_with_idx)
# x1, y1, x2, y2 = boxes.unbind(-1)
# # print("")
# print(x1)
# print(x1.shape)
# sorted_boxes_lst = []
# for i, coord in enumerate((x1, y1, x2, y2)):
#     sorted_coord_idx = torch.argsort(coord, dim=-1, descending=False).unsqueeze(-1).expand(-1, 5)
#     # [bs, num_boxes, 5]
#     sorted_coord_idx = torch.LongTensor(sorted_coord_idx)
#     sorted_boxes_with_idx = torch.gather(boxes_with_idx, 0, sorted_coord_idx)[..., [i,4]].numpy()
#     # [bs, num_boxes, 2]
#     sorted_boxes_lst.append(sorted_boxes_with_idx)

# for arr in sorted_boxes_lst:
#     print(arr, "\n")

# x, y = 0.5, 0.4
# valid_boxes = set()
# valid_boxes_subset = set()
# result_1_range = range(bisect.bisect_left(sorted_boxes_lst[0][:, 0], x))
# result_1 = [int(sorted_boxes_lst[0][k][1]) for k in result_1_range]
# result_1 = set(result_1)
# print(result_1)

# result_2_range = range(bisect.bisect_left(sorted_boxes_lst[1][:, 0], y))
# result_2 = [int(sorted_boxes_lst[1][k][1]) for k in result_2_range]
# result_2 = set(result_2)
# print(result_2)

# result_3_range = range(bisect.bisect_right(sorted_boxes_lst[2][:, 0], x), boxes.shape[0])
# result_3 = [int(sorted_boxes_lst[2][k][1]) for k in result_3_range]
# result_3 = set(result_3)
# print(result_3)

# result_4_range = range(bisect.bisect_right(sorted_boxes_lst[3][:, 0], y), boxes.shape[0])
# result_4 = [int(sorted_boxes_lst[3][k][1]) for k in result_4_range]
# result_4 = set(result_4)
# print(result_4)

# res = result_1 & result_2 & result_3 & result_4
# print(res, "\n\n")

# num_boxes = boxes.shape[0]
# res = set(range(num_boxes))
# for i, arr in enumerate(sorted_boxes_lst):
#     c = x if i % 2 == 0 else y
#     result_i_range = range(bisect.bisect_left(arr[:, 0], c)) if i < 2 else \
#         range(bisect.bisect_right(arr[:, 0], c), num_boxes)
#     result_i = [int(arr[k][1]) for k in result_i_range]
#     result_i = set(result_i)
#     print(result_i)
#     res &= result_i

# print(res)
# if len(res) > 1:
#     print("Warning: the len of set is {}".format(len(res)))
# for b in res:
#     print(b)
#     break


from skimage.draw import bezier_curve
import bezier
import math

img = np.zeros((512, 512, 3), np.uint8)
rectangles = [
    [60,60,80,80],
    [123,78,147,94],
    [168,70,188,92],
    [90,75,110,101],
    [378,405,398,427],
    [360,310,380,330],
    [390,335,410,361],
    [400,356,424,372],
    [398,380,418,402],
]
for rec in rectangles:
    cv2.rectangle(img, rec[:2], rec[2:], (55,255,155), thickness=2)
nodes1 = np.array([
    [60, 95, 130, 188],
    [65, 100, 90, 75],
])
curve1 = bezier.Curve(nodes1, degree=3)
nodes2 = np.array([
    [360, 420, 430, 377],
    [308, 350, 390, 428],
])
curve2 = bezier.Curve(nodes2, degree=3)
curves = [curve1, curve2]

n = 30
delta = 1/n
for i in range(n+1):
    for curve in curves:
        point = curve.evaluate(i*delta)
        x, y = round(point[0][0]), round(point[1][0])
        cv2.circle(img, (x,y), 2, (255, 50, 0), 2)


mapper = {0: 0, 1: 1}
res_lst = [[0,3,1,2],[5,6,7,8,4]]
points1 = torch.tensor(nodes1, dtype=torch.float32).transpose(0,1).reshape(1,8)
points2 = torch.tensor(nodes2, dtype=torch.float32).transpose(0,1).reshape(1,8)
points = torch.cat((points1, points2), dim=0)
boxes = torch.tensor(rectangles, dtype=torch.float32)
polygons = []

def get_poly_one_curve(bezier_idx, n=5):
    # TODO: 应该加一个 bezier_idx 到 self.res_lst 的每一项的 mapper
    #       因为有些曲线并没有穿过 bbox 
    assert bezier_idx in mapper
    box_lst = res_lst[mapper[bezier_idx]]
    bn = len(box_lst)
    ref_num = max(2, bn//3+1)  # 每个顶点处的长度是平均了附近的 ref_num 个 box 的有关长度得来的
    xs = points[bezier_idx][0::2].unsqueeze(0)  # [1, 4]
    ys = points[bezier_idx][1::2].unsqueeze(0)
    t = torch.linspace(0, 1, steps=n, dtype=xs.dtype, device=xs.device)  # [n,]
    c0, c1, c2, c3 = (1-t)**3, 3 * (1-t)**2 * t, 3 * t**2 * (1-t), t**3
    cs = torch.stack((c0, c1, c2, c3), dim=-1)  # [n, 4]
    x, y = (cs*xs).sum(dim=1), (cs*ys).sum(dim=1)  # [n,]
    # 切线系数
    q0, q1, q2, q3 = -3*t**2 + 6*t - 3, 9*t**2 - 12*t + 3, -9*t**2 + 6*t, 3*t**2
    qs = torch.stack((q0, q1, q2, q3), dim=-1)  # [n, 4]
    dx, dy = (qs*xs).sum(dim=1), (qs*ys).sum(dim=1)  # [n,]
    print(dx)
    print(dy)
    poly_vertex = torch.zeros(n*2, 2)
    tmp_boxes = boxes[box_lst, :]
    max_height = torch.max(tmp_boxes[:,3]-tmp_boxes[:,1])
    max_width = torch.max(tmp_boxes[:,2]-tmp_boxes[:,0])
    maxi = torch.max(max_height, max_width)
    for i in range(n):
        radius = maxi
        # 先暂时这样干吧
        rad = math.atan2(dx[i].item(), -dy[i].item())
        _dx, _dy = radius/2 * math.cos(rad), radius/2 * math.sin(rad)
        poly_vertex[i][0] = x[i] - _dx
        poly_vertex[i][1] = y[i] - _dy
        poly_vertex[n*2-i-1][0] = x[i] + _dx
        poly_vertex[n*2-i-1][1] = y[i] + _dy
    polygons.append(poly_vertex)

n = 5
get_poly_one_curve(0, n)
get_poly_one_curve(1, n)
for polygon in polygons:
    for i in range(n*2):
        xx, yy = round(polygon[i][0].item()), round(polygon[i][1].item())
        # xx, yy = round(x[i].item()), round(y[i].item())
        color = [round(255*(i/(n*2-1) * 0.6 + 0.4))] * 3
        # print(color)
        cv2.circle(img, (xx, yy), 2, color, 2)

cv2.imshow("rectangle", img)
cv2.waitKey(0)
cv2.destroyAllWindows()
