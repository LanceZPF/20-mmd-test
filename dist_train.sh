#!/usr/bin/env bash

PORT=${PORT:-29470}

PYTHONPATH="$(dirname $0)/..":$PYTHONPATH \
python -m torch.distributed.launch --nproc_per_node=4 --master_port=$PORT \
    $(dirname "$0")/fctrain.py --launcher pytorch ${@:3}
