#!/usr/bin/env bash

set -x

EXP_DIR=exps/r50_deformable_detr_330
PY_ARGS=${@:1}

python -u main_resume.py \
    --output_dir ${EXP_DIR} \
    --resume ${EXP_DIR}/checkpoint0005.pth \
    --epochs 8 \
    --lr_drop 6\
    ${PY_ARGS}
