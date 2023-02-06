#!/usr/bin/env bash

set -x

EXP_DIR=exps/v002
PY_ARGS=${@:1}

python -u main_eval.py \
    --resume ${EXP_DIR}/checkpoint0014.pth \
    --eval \
    ${PY_ARGS}
