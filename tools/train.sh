#!/usr/bin/env bash
export CUDA_VISIBLE_DEVICES=0
GPUS=1
PORT=${PORT:-29504}

# recon stage
CONFIG=configs/voxdet/VoxDet_train_p1.py

PYTHONPATH="$(dirname $0)/..":$PYTHONPATH \
python3 $(dirname "$0")/train.py --config $CONFIG

# det stage
# CONFIG=configs/voxdet/VoxDet_train_p2.py

# PYTHONPATH="$(dirname $0)/..":$PYTHONPATH \
# python3 $(dirname "$0")/train.py --config $CONFIG
