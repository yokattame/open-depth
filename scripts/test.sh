#!/usr/bin/env bash

CUDA_VISIBLE_DEVICES=0 \
  python baseline.py \
  --dataset='KittiStereo' \
  --dataset_root='../data/KittiStereo' \
  --batch_size=20 \
  --input_height=0 \
  --input_width=0 \
  --resume='logs/baseline/model_best.pth.tar' \
  --evaluate

