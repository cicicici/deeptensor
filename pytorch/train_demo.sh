#!/bin/bash

CUR_DIR=$(pwd)

DT2_ROOT='../../detectron2'

python $DT2_ROOT/tools/train_net.py --num-gpus 8 \
  --config-file $DT2_ROOT/configs/COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_1x.yaml

