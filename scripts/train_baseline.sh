#!/usr/bin/env bash

CUDA_VISIBLE_DEVICES=3 \
  python baseline.py \
  --dataset='KittiStereo' \
  --dataset_root='../data/KittiStereo' \
  --validation 1 \
  --batch_size=20 \
  --lr=0.0001 \
  --epochs=120 \
  --input_height=320 \
  --input_width=1216 \
  --optimizer='Adam' \
  --beta1=0.9 \
  --beta2=0.999 \
  --logs_dir='logs/baseline' \

