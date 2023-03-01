#!/usr/bin/env bash

set -x

EXP_DIR=exps/v002-2stage-bbox_ref
PY_ARGS=${@:1}

python -u main.py \
    --output_dir ${EXP_DIR} \
    --batch_size 2 \
    --lr 4e-5 \
    --lr_backbone 4e-6 \
    --with_box_refine \
    --two_stage \
    ${PY_ARGS}
