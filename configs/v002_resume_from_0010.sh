#!/usr/bin/env bash

set -x

EXP_DIR=exps/v002
PY_ARGS=${@:1}

python -u main_resume.py \
    --output_dir ${EXP_DIR} \
    --resume ${EXP_DIR}/checkpoint0010.pth \
    --batch_size 2 \
    --lr 4e-5 \
    --lr_backbone 4e-6 \
    --epochs 20 \
    --lr_drop 20 \
    ${PY_ARGS}
