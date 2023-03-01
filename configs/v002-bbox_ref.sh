#!/usr/bin/env bash

set -x

EXP_DIR=exps/v002-relative_coord-box_refine
PY_ARGS=${@:1}

python -u main.py \
    --output_dir ${EXP_DIR} \
    --batch_size 2 \
    --lr 4e-5 \
    --lr_backbone 4e-6 \
    --with_box_refine \
    ${PY_ARGS}


# 跑这个实验的时候，bezier曲线控制点预测偏移量的bug已经被改好了