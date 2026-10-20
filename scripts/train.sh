#!/usr/bin/env bash
# 用法: bash scripts/train.sh [模型名] [epochs]
# 例:   bash scripts/train.sh yolov8n 100
set -e
cd "$(dirname "$0")/.."
MODEL=${1:-yolov8n}
EPOCHS=${2:-100}
~/miniconda3/envs/mathorcup/bin/yolo train \
  model=weights/${MODEL}.pt \
  data=configs/data.yaml \
  epochs=${EPOCHS} imgsz=640 batch=16 device=0 workers=8 \
  project=/home/mary/mathorcup/outputs name=${MODEL}_${EPOCHS}e exist_ok=true
