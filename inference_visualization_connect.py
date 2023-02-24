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
import cv2
import os



def get_args_parser():
    parser = argparse.ArgumentParser('Deformable DETR Detector', add_help=False)
    parser.add_argument('--lr', default=2e-5, type=float)  # origin: 2e-4
    parser.add_argument('--lr_backbone_names', default=["backbone.0"], type=str, nargs='+')
    parser.add_argument('--lr_backbone', default=2e-6, type=float)  # origin: 2e-5
    parser.add_argument('--lr_linear_proj_names', default=['reference_points', 'sampling_offsets'], type=str, nargs='+')
    parser.add_argument('--lr_linear_proj_mult', default=0.1, type=float)
    parser.add_argument('--batch_size', default=1, type=int)  # origin: 2
    parser.add_argument('--weight_decay', default=1e-4, type=float)
    parser.add_argument('--epochs', default=12, type=int)  # origin: 50
    parser.add_argument('--lr_drop', default=8, type=int)  # origin: 40
    parser.add_argument('--lr_drop_epochs', default=None, type=int, nargs='+')
    parser.add_argument('--clip_max_norm', default=0.1, type=float,
                        help='gradient clipping max norm')


    parser.add_argument('--sgd', action='store_true')

    # Variants of Deformable DETR
    parser.add_argument('--with_box_refine', default=False, action='store_true')
    parser.add_argument('--two_stage', default=False, action='store_true')

    # Model parameters
    parser.add_argument('--frozen_weights', type=str, default=None,
                        help="Path to the pretrained model. If set, only the mask head will be trained")

    # * Backbone
    parser.add_argument('--backbone', default='resnet50', type=str,
                        help="Name of the convolutional backbone to use")
    parser.add_argument('--dilation', action='store_true',
                        help="If true, we replace stride with dilation in the last convolutional block (DC5)")
    parser.add_argument('--position_embedding', default='sine', type=str, choices=('sine', 'learned'),
                        help="Type of positional embedding to use on top of the image features")
    parser.add_argument('--position_embedding_scale', default=2 * np.pi, type=float,
                        help="position / size * scale")
    parser.add_argument('--num_feature_levels', default=4, type=int, help='number of feature levels')

    # * Transformer
    parser.add_argument('--enc_layers', default=6, type=int,
                        help="Number of encoding layers in the transformer")
    parser.add_argument('--dec_layers', default=6, type=int,
                        help="Number of decoding layers in the transformer")
    parser.add_argument('--dim_feedforward', default=1024, type=int,
                        help="Intermediate size of the feedforward layers in the transformer blocks")
    parser.add_argument('--hidden_dim', default=256, type=int,
                        help="Size of the embeddings (dimension of the transformer)")
    parser.add_argument('--dropout', default=0.1, type=float,
                        help="Dropout applied in the transformer")
    parser.add_argument('--nheads', default=8, type=int,
                        help="Number of attention heads inside the transformer's attentions")
    parser.add_argument('--num_queries', default=(330, 100), type=tuple,
                        help="Number of query slots")  # origin: 300
    parser.add_argument('--dec_n_points', default=4, type=int)
    parser.add_argument('--enc_n_points', default=4, type=int)

    # * Segmentation
    parser.add_argument('--masks', action='store_true',
                        help="Train segmentation head if the flag is provided")

    # Loss
    # 如果传入--no_aux_loss，则 no_aux_loss 的值为 False，可以用 aux_loss 来访问该值
    parser.add_argument('--no_aux_loss', dest='aux_loss', action='store_false',
                        help="Disables auxiliary decoding losses (loss at each layer)")

    # * Matcher
    parser.add_argument('--set_cost_class_c', default=2, type=float,
                        help="Character class coefficient in the matching cost")
    parser.add_argument('--set_cost_bbox', default=5, type=float,
                        help="L1 box coefficient in the matching cost")
    parser.add_argument('--set_cost_giou', default=2, type=float,
                        help="giou box coefficient in the matching cost")

    parser.add_argument('--set_cost_class_b', default=2, type=float,
                        help="Bezier curve class coefficient in the matching cost")
    parser.add_argument('--set_cost_point', default=5, type=float,
                        help="L2 point coefficient in the matching cost")

    # * Loss coefficients
    # 各项损失的权重
    parser.add_argument('--mask_loss_coef', default=1, type=float)
    parser.add_argument('--dice_loss_coef', default=1, type=float)
    parser.add_argument('--cls_c_loss_coef', default=2, type=float)
    parser.add_argument('--bbox_loss_coef', default=5, type=float)
    parser.add_argument('--giou_loss_coef', default=2, type=float)

    parser.add_argument('--cls_b_loss_coef', default=2, type=float)
    parser.add_argument('--point_loss_coef', default=5, type=float)

    parser.add_argument('--focal_alpha', default=0.25, type=float)

    # dataset parameters
    parser.add_argument('--dataset_file', default='synthtext', choices=('coco', 'coco_panoptic', 'synthtext'))
    parser.add_argument('--coco_path', default='./data/coco', type=str)
    parser.add_argument('--coco_panoptic_path', type=str)
    parser.add_argument('--remove_difficult', action='store_true')

    parser.add_argument('--synthtext_path', default='./data/synthtext', type=str)

    parser.add_argument('--output_dir', default='',
                        help='path where to save, empty for no saving')
    parser.add_argument('--device', default='cuda',
                        help='device to use for training / testing')
    parser.add_argument('--seed', default=42, type=int)
    parser.add_argument('--resume', default='./exps/v002/checkpoint0014.pth', help='resume from checkpoint')
    parser.add_argument('--start_epoch', default=0, type=int, metavar='N',
                        help='start epoch')
    parser.add_argument('--eval', action='store_true')
    parser.add_argument('--num_workers', default=2, type=int)
    parser.add_argument('--cache_mode', default=False, action='store_true', help='whether to cache images on memory')

    parser.add_argument('--train_print_freq', default=500, type=int)
    parser.add_argument('--eval_print_freq', default=200, type=int)

    # inference parameters
    parser.add_argument('--inference_img_path', default='./data/synthtext/SynthText/8/ballet_3_0.jpg', type=str)

    return parser


def main(args):
    utils.init_distributed_mode(args)
    print("git:\n  {}\n".format(utils.get_sha()))

    if args.frozen_weights is not None:
        assert args.masks, "Frozen training is meant for segmentation only"
    print(args)

    device = torch.device(args.device)  # 用于训练/测试的设备  cuda

    # fix the seed for reproducibility
    seed = args.seed + utils.get_rank()
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)

    model, criterion, postprocessors = build_model(args)
    model.to(device)

    model_without_ddp = model
    n_parameters = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print('number of params:', n_parameters)

    # # dataset
    # dataset_train = build_dataset(image_set='train', args=args)
    # dataset_val = build_dataset(image_set='val', args=args)

    # if args.distributed:
    #     if args.cache_mode:
    #         sampler_train = samplers.NodeDistributedSampler(dataset_train)
    #         sampler_val = samplers.NodeDistributedSampler(dataset_val, shuffle=False)
    #     else:
    #         sampler_train = samplers.DistributedSampler(dataset_train)
    #         sampler_val = samplers.DistributedSampler(dataset_val, shuffle=False)
    # else:
    #     sampler_train = torch.utils.data.RandomSampler(dataset_train)
    #     sampler_val = torch.utils.data.SequentialSampler(dataset_val)

    # batch_sampler_train = torch.utils.data.BatchSampler(
    #     sampler_train, args.batch_size, drop_last=True)

    # # dataloader
    # data_loader_train = DataLoader(dataset_train, batch_sampler=batch_sampler_train,
    #                                collate_fn=utils.collate_fn, num_workers=args.num_workers,
    #                                pin_memory=True)
    # print("\n\n\nDATA LOADER TRAIN LEN:", len(data_loader_train))
    # data_loader_val = DataLoader(dataset_val, args.batch_size, sampler=sampler_val,
    #                              drop_last=False, collate_fn=utils.collate_fn, num_workers=args.num_workers,
    #                              pin_memory=True)
    # print("\n\n\nDATA LOADER VAL LEN:", len(data_loader_val))

    # lr_backbone_names = ["backbone.0", "backbone.neck", "input_proj", "transformer.encoder"]
    def match_name_keywords(n, name_keywords):
        out = False
        for b in name_keywords:
            if b in n:
                out = True
                break
        return out

    # for n, p in model_without_ddp.named_parameters():
    #     print(n)

    param_dicts = [
        {
            "params":
                [p for n, p in model_without_ddp.named_parameters()
                 if not match_name_keywords(n, args.lr_backbone_names) and not match_name_keywords(n, args.lr_linear_proj_names) and p.requires_grad],
            "lr": args.lr,
        },
        {
            "params": [p for n, p in model_without_ddp.named_parameters() if match_name_keywords(n, args.lr_backbone_names) and p.requires_grad],
            "lr": args.lr_backbone,
        },
        {
            "params": [p for n, p in model_without_ddp.named_parameters() if match_name_keywords(n, args.lr_linear_proj_names) and p.requires_grad],
            "lr": args.lr * args.lr_linear_proj_mult,
        }
    ]
    # optimizer
    if args.sgd:
        optimizer = torch.optim.SGD(param_dicts, lr=args.lr, momentum=0.9,
                                    weight_decay=args.weight_decay)
    else:
        optimizer = torch.optim.AdamW(param_dicts, lr=args.lr,
                                      weight_decay=args.weight_decay)
    lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer, args.lr_drop)

    if args.distributed:
        model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[args.gpu])
        model_without_ddp = model.module

    # if args.dataset_file == "coco_panoptic":
    #     # We also evaluate AP during panoptic training, on original coco DS
    #     coco_val = datasets.coco.build("val", args)
    #     base_ds = get_coco_api_from_dataset(coco_val)
    # else:
    #     base_ds = get_coco_api_from_dataset(dataset_val)

    if args.frozen_weights is not None:
        checkpoint = torch.load(args.frozen_weights, map_location='cpu')
        model_without_ddp.detr.load_state_dict(checkpoint['model'])

    output_dir = Path(args.output_dir)
    
    # 加载模型
    checkpoint = torch.load(args.resume, map_location='cpu')
    missing_keys, unexpected_keys = model_without_ddp.load_state_dict(checkpoint['model'], strict=False)
    unexpected_keys = [k for k in unexpected_keys if not (k.endswith('total_params') or k.endswith('total_ops'))]
    if len(missing_keys) > 0:
        print('Missing Keys: {}'.format(missing_keys))
    if len(unexpected_keys) > 0:
        print('Unexpected Keys: {}'.format(unexpected_keys))
    if not args.eval and 'optimizer' in checkpoint and 'lr_scheduler' in checkpoint and 'epoch' in checkpoint:
        import copy
        p_groups = copy.deepcopy(optimizer.param_groups)
        optimizer.load_state_dict(checkpoint['optimizer'])
        for pg, pg_old in zip(optimizer.param_groups, p_groups):
            pg['lr'] = pg_old['lr']
            pg['initial_lr'] = pg_old['initial_lr']
        # print(optimizer.param_groups)
        lr_scheduler.load_state_dict(checkpoint['lr_scheduler'])
        # todo: this is a hack for doing experiment that resume from checkpoint and also modify lr scheduler (e.g., decrease lr in advance).
        args.override_resumed_lr_drop = True
        if args.override_resumed_lr_drop:
            print('Warning: (hack) args.override_resumed_lr_drop is set to True, so args.lr_drop would override lr_drop in resumed lr_scheduler.')
            lr_scheduler.step_size = args.lr_drop
            lr_scheduler.base_lrs = list(map(lambda group: group['initial_lr'], optimizer.param_groups))
        lr_scheduler.step(lr_scheduler.last_epoch)
        args.start_epoch = checkpoint['epoch'] + 1
        print("start_epoch:", args.start_epoch)
    # check the resumed model
    # if not args.eval:
    #     test_stats, coco_evaluator = evaluate(
    #         model, criterion, postprocessors, data_loader_val, base_ds, device, args.output_dir
    #     )

    # 只验证而不训练的情况
    # if args.eval:
    #     test_stats, coco_evaluator = evaluate(model, criterion, postprocessors,
    #                                           data_loader_val, base_ds, device, args.output_dir)
    #     if args.output_dir:
    #         utils.save_on_master(coco_evaluator.coco_eval["bbox"].eval, output_dir / "eval.pth")
    #     return

    # load dictionary
    # TODO: 要改！！！因为一个v可能对应2个k（大小写字母）
    char2num = dict()
    with open("./vis/char2num_dict.json", 'r') as fp:
        char2num = json.load(fp)
    # print("char2num:", char2num)
    num2char = dict()
    for k, v in char2num.items():
        num2char[v] = k

    # inference
    print("------Start inference------")
    start_time = time.time()
    
    transforms = None
    # transforms = T.Compose([
    #     T.ToTensor(),
    #     T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    # ])
    transforms = make_coco_transforms('val')
    print(transforms)
    model_ver = args.resume.split('/')[-2]
    print('model version:', model_ver)
    inf_path = inference(model, args.inference_img_path, device, transforms, num2char, model_ver)
    # img: Image.open(BytesIO(self.cache[path])).convert('RGB')
    # TODO: 参考eval函数来写就行，注意要用 PostProcessor 来得到结果
    
    total_time = time.time() - start_time
    total_time_str = str(datetime.timedelta(seconds=int(total_time)))
    print('Inference time {}'.format(total_time_str))

    print("\n--------GT--------")
    gt_mat_path = "/DATACENTER/s/yaowenhao/proj/Deformable-DETR-SynthText-recog/data/synthtext/SynthText/gt.mat"
    gt_path = vis_gt(args.inference_img_path, transforms, gt_mat_path)
    # all_numpy_arr = np.array(all_lst)
    # np.save('log_numpy.npy', all_numpy_arr)

    img_inf, img_gt = cv2.imread(inf_path), cv2.imread(gt_path)
    img_connect = cv2.hconcat([img_inf, img_gt])
    final_save_path = f'./vis/connection/{model_ver}/' + inf_path.split('/')[-1]
    os.makedirs(os.path.dirname(final_save_path), exist_ok=True)
    cv2.imwrite(final_save_path, img_connect)
    print("final save path:", final_save_path)




def inference(model, img_path, device, transforms, num2char, model_ver):
    model.eval()
    origin_img = Image.open(img_path).convert('RGB')
    img, _ = transforms(origin_img, None)
    img = img.unsqueeze(0).to(device)
    print(img.shape)
    # print('img:\n', img)
    img_size = torch.tensor(img.shape)[-2:]
    print(img_size)
    
    output = model(img)
    out_logits_c, out_bbox = output['char']['pred_logits'], output['char']['pred_boxes']
    print('\nchar output shape:')
    print('pred_logits:', out_logits_c.shape)
    print('pred_boxes:', out_bbox.shape)

    # post process
    prob_c = out_logits_c.sigmoid()
    # TODO: 现在感觉最麻烦的就是这个地方了
    topk_values_c, topk_indexes_c = torch.topk(prob_c.view(out_logits_c.shape[0], -1), 100, dim=1)  # [bs, 20]
    scores_c = topk_values_c
    print('scores shape:', scores_c.shape)
    topk_boxes = topk_indexes_c // out_logits_c.shape[2]
    labels_c = topk_indexes_c % out_logits_c.shape[2]
    print('label shape:', labels_c.shape)
    print(labels_c)
    boxes = box_ops.box_cxcywh_to_xyxy(out_bbox)
    boxes = torch.gather(boxes, 1, topk_boxes.unsqueeze(-1).repeat(1,1,4))  # 根据索引来取得对应结果
    # print(boxes)
    threshold = 0.3
    
    vslzr = visualizer.COCOVisualizer()

    boxes = box_ops.box_xyxy_to_cxcywh(boxes)
    select_mask_c = scores_c > threshold
    # print('select mask:', select_mask)
    pred_dict_c = {
        'boxes': boxes[select_mask_c],  # xywh, [0,1]
        'size': img_size,
        'box_label': [num2char[c.item()] for c in labels_c[select_mask_c]],
        'caption': 'inference',
    }
    print(pred_dict_c['boxes'])
    
    out_logits_b, out_point = output['bezier']['pred_logits'], output['bezier']['pred_points']
    print('\nbezier output shape:')
    print('pred_logits:', out_logits_b.shape)
    print('pred_points:', out_point.shape)

    # post process
    prob_b = out_logits_b.sigmoid()
    # TODO: 现在感觉最麻烦的就是这个地方了
    topk_values_b, topk_indexes_b = torch.topk(prob_b.view(out_logits_b.shape[0], -1), 60, dim=1)  # [bs, 20]
    scores_b = topk_values_b
    print('scores shape:', scores_b.shape)
    topk_curves = topk_indexes_b // out_logits_b.shape[2]
    labels_b = topk_indexes_b % out_logits_b.shape[2]
    print('label shape:', labels_b.shape)
    print(labels_b)
    points = torch.gather(out_point, 1, topk_curves.unsqueeze(-1).repeat(1,1,8))  # 根据索引来取得对应结果
    # print(boxes)
    threshold = 0.3
    select_mask_b = scores_b > threshold
    # print('select mask:', select_mask)
    pred_dict_b = {
        'curves': points[select_mask_b],  # x1y1x2y2x3y3x4y4, [0,1]
        'size': img_size,
    }

    img_cpu = img.squeeze(0).to('cpu')
    print(img_cpu.shape)

    savedir = f'./vis/visual_result/{model_ver}'
    vslzr.visualize(img_path, img_cpu, pred_dict_c, pred_dict_b, savedir=savedir, dpi=200)

    dir_num, file_name = img_path.split('/')[-2:]
    file_name = file_name[:-4]
    savename = '{}/vis-{}-{}.png'.format(savedir, dir_num, file_name)
    return savename


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


def get_point_gt(coord_w_gt: np.ndarray, img_size):
    assert coord_w_gt.ndim == 3  # [2, 4, char_num]
    coord_w_gt = coord_w_gt.transpose(2, 1, 0)
    # [num_bezier, 4, 2]    num_bezier == num_words
    pt1, pt2, pt3, pt4 = np.split(coord_w_gt, 4, axis=1)
    # [num_bezier, 1, 2]
    pt_start, pt_end = ((pt1+pt4)/2).squeeze(1), ((pt2+pt3)/2).squeeze(1)
    # [num_bezier, 2]
    bp1, bp2, bp3, bp4 = pt_start, (pt_start*2+pt_end*1)/3, (pt_start*1+pt_end*2)/3, pt_end
    # [num_bezier, 2]
    bezier_points = np.concatenate((bp1, bp2, bp3, bp4), axis=1)
    bezier_points = np.array(bezier_points, dtype=np.float64)
    bezier_points = np.around(bezier_points, 2)
    bezier_points = torch.tensor(bezier_points, dtype=torch.float64)
    bezier_points = bezier_points / torch.tensor([img_size[0], img_size[1]]*4)
    return bezier_points


def get_string(text_lst):
    res_str = ""
    for word in text_lst:
        for c in word:
            if c not in (' ', '\t', '\n'):
                res_str += c
    return res_str


def vis_gt(img_path_gt: str, transforms, gt_mat_path):
    origin_img_gt = Image.open(img_path_gt).convert('RGB')
    origin_img_gt_size = torch.tensor(origin_img_gt.size)
    print("origin_img_gt_size:", origin_img_gt_size)
    img_gt, _ = transforms(origin_img_gt, None)
    img_gt = img_gt.unsqueeze(0)
    print(img_gt.shape)
    # print('img_gt:\n', img_gt)
    img_gt_size = torch.tensor(img_gt.shape)[-2:]
    print(img_gt_size)
    
    features = scipy.io.loadmat(gt_mat_path)
    total = len(features["wordBB"][0])
    img_path_find = '/'.join((img_path_gt).split('/')[-2:])
    print("img_path_find:", img_path_find)
    for i in range(total):
        if features["imnames"][0][i][0] == img_path_find:
            print("HAHA!!")
            coord_c_gt = features['charBB'][0][i]
            # 产生的bbox是xxyy，未经过归一化
            coord_c_gt = get_bbox_gt(coord_c_gt, origin_img_gt_size)
            text_lst = features["txt"][0][i]
            label_string = get_string(text_lst)
            print("shape of coordinates:", coord_c_gt.shape)
            print("len of label string:", len(label_string))
            print(label_string)
            assert len(label_string) == coord_c_gt.shape[0]
            coord_w_gt = features['wordBB'][0][i]
            if coord_w_gt.ndim == 2:
                coord_w_gt = coord_w_gt[..., None]
            bezier_points = get_point_gt(coord_w_gt, origin_img_gt_size)
            break

    vslzr = visualizer.COCOVisualizer()
    boxes_gt = box_ops.box_xyxy_to_cxcywh(coord_c_gt)
    # print('select mask:', select_mask)
    gt_dict_c = {
        'boxes': boxes_gt,  # xywh, [0,1]
        'size': img_gt_size,
        'box_label': [c for c in label_string],
        'caption': 'ground truth',
    }
    
    gt_dict_b = {
        'curves': bezier_points,
        'size': img_gt_size,
    }
    
    # print(pred_dict['boxes'])
    img_cpu = img_gt.squeeze(0).to('cpu')
    print(img_cpu.shape)
    savedir = './vis/gt_temp'
    vslzr.visualize(img_path_gt, img_cpu, gt_dict_c, gt_dict_b, savedir=savedir, dpi=200)

    dir_num, file_name = img_path_gt.split('/')[-2:]
    file_name = file_name[:-4]
    savename = '{}/vis-{}-{}.png'.format(savedir, dir_num, file_name)
    return savename



if __name__ == '__main__':
    parser = argparse.ArgumentParser('Deformable DETR training and evaluation script', parents=[get_args_parser()])
    args = parser.parse_args()
    if args.output_dir:
        Path(args.output_dir).mkdir(parents=True, exist_ok=True)
    main(args)
