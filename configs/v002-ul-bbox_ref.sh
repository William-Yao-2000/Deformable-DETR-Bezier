#!/usr/bin/env bash

set -x

EXP_DIR=exps/v002-relative_coord-ul-box_refine
PY_ARGS=${@:1}

python -u main.py \
    --output_dir ${EXP_DIR} \
    --batch_size 2 \
    --lr 4e-5 \
    --lr_backbone 4e-6 \
    --with_box_refine \
    --epochs 18 \
    --lr_drop 16 \
    ${PY_ARGS}


# 区分大小写，一共95个类