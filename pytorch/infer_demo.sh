#!/bin/bash

CUR_DIR=$(pwd)

DT2_ROOT='../../detectron2'

python $DT2_ROOT/demo/demo.py --config-file $DT2_ROOT/configs/COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x.yaml \
  --input input.jpg \
  --output output \
  --opts MODEL.WEIGHTS detectron2://COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x/137849600/model_final_f10217.pkl

