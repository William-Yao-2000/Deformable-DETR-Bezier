import torch
import numpy as np
from util import box_ops
import bisect
import math


class BezierConnector():
    """
    对一张图片进行结果连接
    """
    
    def __init__(self, boxes, labels, points, img_shape):
        """
        Args:
            boxes (torch.Tensor): xywh, range: [0, 1]
                                  shape: [num_boxes, 4]
            labels (list): labels of boxes
            points (torch.Tensor): x1y1x2y2x3y3x4y4, range: [0, 1]
                                   shape: [num_curves, 8]
            img_shape: [W, H]
        """
        self.num_boxes = boxes.shape[0]
        self.num_curves = points.shape[0]
        new_boxes = box_ops.box_cxcywh_to_xyxy(boxes)  # x1y1x2y2
        self.boxes = new_boxes * torch.cat([img_shape, img_shape], dim=0)  # [num_boxes, 4]
        self.labels = labels
        self.points = points * torch.cat([img_shape]*4, dim=0)  # [num_curves, 8]
        self.w, self.h = img_shape[0], img_shape[1]

        self.sorted_boxes_lst = []
        self.sort_boxes()
        self.used_box = set()
        self.res_lst = []
        self.mapper = {}  # bezier curve index 到 self.res_lst 的下标
        
        self.polygons = []
    
    def sort_boxes(self):
        boxes_idx = torch.arange(self.num_boxes, dtype=self.boxes.dtype, device=self.boxes.device)
        boxes_idx = boxes_idx.unsqueeze(-1)
        boxes_with_idx = torch.cat((self.boxes, boxes_idx), dim=-1)  # [num_boxes, 5]
        x1, y1, x2, y2 = self.boxes.unbind(-1)  # [num_boxes,]
        for i, coord in enumerate((x1, y1, x2, y2)):
            sorted_coord_idx = torch.argsort(coord, dim=-1, descending=False).unsqueeze(-1).expand(-1, 5)
            # [num_boxes, 5]
            # sorted_coord_idx = torch.LongTensor(sorted_coord_idx)
            sorted_boxes_with_idx = torch.gather(boxes_with_idx, 0, sorted_coord_idx)[..., [i,4]].cpu().detach().numpy()
            # [num_boxes, 2]
            self.sorted_boxes_lst.append(sorted_boxes_with_idx)
    
    def which_box(self, x, y):
        res = set(range(self.num_boxes))
        for i, arr in enumerate(self.sorted_boxes_lst):
            c = x if i % 2 == 0 else y
            result_i_range = range(bisect.bisect_left(arr[:, 0], c)) if i < 2 else \
                range(bisect.bisect_right(arr[:, 0], c), self.num_boxes)
            result_i = [int(arr[k][1]) for k in result_i_range]
            result_i = set(result_i)
            res &= result_i
        if len(res) > 1:
            print("Warning: the len of set is {}".format(len(res)))
        if len(res) == 0:
            return None
        for b in res:
            if b not in self.used_box:
                self.used_box.add(b)
                return b
        return None
    
    @staticmethod
    def gen_points(control_points, n=30):
        """
        生成**一条** Bezier 曲线上的点
        """
        xs = control_points[0::2].unsqueeze(0)  # [1, 4]
        ys = control_points[1::2].unsqueeze(0)
        t = torch.linspace(0, 1, steps=n, dtype=xs.dtype, device=xs.device)  # [n,]
        c0, c1, c2, c3 = (1-t)**3, 3 * (1-t)**2 * t, 3 * t**2 * (1-t), t**3
        cs = torch.stack((c0, c1, c2, c3), dim=-1)  # [n, 4]
        x, y = (cs*xs).sum(dim=1), (cs*ys).sum(dim=1)  # [n,]
        return x, y

    def group_sort_one_curve(self, bezier_idx, n=30):
        control_points = self.points[bezier_idx]
        points_x, points_y = self.gen_points(control_points, n)
        res_sublst = []
        for i in range(n):
            x, y = points_x[i].item(), points_y[i].item()
            box_idx = self.which_box(x, y)
            if box_idx is not None:
                res_sublst.append(box_idx)
        if len(res_sublst) > 0:
            self.mapper[bezier_idx] = len(self.res_lst)
            self.res_lst.append(res_sublst)

    def group_sort(self, n=30):
        for i in range(self.num_curves):
            self.group_sort_one_curve(i, n)
        for i, lst in enumerate(self.res_lst):
            s = ""
            for idx in lst:
                s += self.labels[idx]
            print("{}: {}".format(i, s))
        return self.res_lst
    
    def get_poly_one_curve(self, bezier_idx, n=5):
        assert bezier_idx in self.mapper
        box_lst = self.res_lst[self.mapper[bezier_idx]]
        bn = len(box_lst)
        ref_num = max(2, bn//3+1)  # 每个顶点处的长度是平均了附近的 ref_num 个 box 的有关长度得来的
        xs = self.points[bezier_idx][0::2].unsqueeze(0)  # [1, 4]
        ys = self.points[bezier_idx][1::2].unsqueeze(0)
        t = torch.linspace(0, 1, steps=n, dtype=xs.dtype, device=xs.device)  # [n,]
        c0, c1, c2, c3 = (1-t)**3, 3 * (1-t)**2 * t, 3 * t**2 * (1-t), t**3
        cs = torch.stack((c0, c1, c2, c3), dim=-1)  # [n, 4]
        x, y = (cs*xs).sum(dim=1), (cs*ys).sum(dim=1)  # [n,]
        # 切线系数
        q0, q1, q2, q3 = -3*t**2 + 6*t - 3, 9*t**2 - 12*t + 3, -9*t**2 + 6*t, 3*t**2
        qs = torch.stack((q0, q1, q2, q3), dim=-1)  # [n, 4]
        dx, dy = (qs*xs).sum(dim=1), (qs*ys).sum(dim=1)  # [n,]
        poly_vertex = torch.zeros(n*2, 2)
        tmp_boxes = self.boxes[box_lst, :]
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
        self.polygons.append(poly_vertex)

    def get_poly(self):
        self.polygons = []
        for bezier_idx, sub_lst_idx in self.mapper.items():
            word_len = len(self.res_lst[sub_lst_idx])
            self.get_poly_one_curve(bezier_idx)
        self.polygons = torch.stack(self.polygons, dim=0)
        print("POLYGON SHAPE:")
        print(self.polygons.shape)
        return self.polygons
