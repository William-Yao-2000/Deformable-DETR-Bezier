#!/usr/bin/env bash

set -x

EXP_DIR=exps/v001
PY_ARGS=${@:1}

python -u main.py \
    --output_dir ${EXP_DIR} \
    --batch_size 1 \
    --lr 2e-5 \
    --lr_backbone 2e-6 \
    --train_print_freq 200 \
    --eval_print_freq 100 \
    ${PY_ARGS}
