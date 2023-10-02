#!/usr/bin/env bash
export CUDA_VISIBLE_DEVICES=0,1,2,3
CONFIG=configs/voxdet/VoxDet_train_p2.py
GPUS=4
PORT=${PORT:-29501}

PYTHONPATH="$(dirname $0)/..":$PYTHONPATH \
python3 -m torch.distributed.launch --nproc_per_node=$GPUS --master_port=$PORT \
    $(dirname "$0")/train.py --config $CONFIG --launcher pytorch ${@:4}
