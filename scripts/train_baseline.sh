#!/usr/bin/env bash

CUDA_VISIBLE_DEVICES=3 \
  python baseline.py \
  --dataset='KittiStereo' \
  --dataset_root='../data/KittiStereo' \
  --validation 1 \
  --batch_size=20 \
  --lr=0.001 \
  --epochs=45 \
  --input_height=320 \
  --input_width=1216 \
  --optimizer='SGD' \
  --logs_dir='logs/baseline' \
  --step_size=15

