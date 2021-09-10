#!/bin/bash

# distbuted training
python -m torch.distributed.run \
  --standalone --nnodes=1 --nproc_per_node=4 \
  main.py \
    --epochs 100 \
    --content-layers relu2_1
