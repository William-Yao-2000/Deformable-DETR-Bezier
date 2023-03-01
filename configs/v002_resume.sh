#!/usr/bin/env bash

set -x

EXP_DIR=exps/v002-absolute_coord-al-resume_from_11
PY_ARGS=${@:1}

python -u main_resume.py \
    --output_dir ${EXP_DIR} \
    --resume ${EXP_DIR}/checkpoint0011.pth \
    --batch_size 2 \
    --lr 4e-5 \
    --lr_backbone 4e-6 \
    --epochs 16 \
    --lr_drop 14 \
    ${PY_ARGS}
