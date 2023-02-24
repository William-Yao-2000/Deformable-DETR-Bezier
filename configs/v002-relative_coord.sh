#!/usr/bin/env bash

set -x

EXP_DIR=exps/v002-relative_coord
PY_ARGS=${@:1}

python -u main.py \
    --output_dir ${EXP_DIR} \
    --batch_size 2 \
    --lr 4e-5 \
    --lr_backbone 4e-6 \
    ${PY_ARGS}


# 主要修改的是 deformable-DETR 里面的 bezier control points 预测部分，变成了预测四个点到参考点的偏移量