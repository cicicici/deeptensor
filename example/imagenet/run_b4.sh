#!/bin/bash

CUR_DIR=$(pwd)

COMMIT=$(git rev-parse --short HEAD)
HOST=$(hostname | tr '-' '\n' | head -n 3 | tail -1)
DATE=$(date +%Y%m%d)

#LAUNCH=../launch/launch_gpu_4.sh
LAUNCH=../launch/launch_multi.sh
APP=imagenet_b4.py
TRACE=--trace

#INI_PRE=imagenet_l2_k
#INI_PRE=imagenet_l2_l
#INI_PRE=imagenet_l2_m
#INI_PRE=imagenet_l2_n
INI_PRE=imagenet_b4
INI=config/$INI_PRE".ini"

MAX_EP=450
VALID_EP=1

MODEL_DIR_PRE=$INI_PRE"."$DATE

echo "COMMIT: $COMMIT"
echo "HOST: $HOST"
echo "DATE: $DATE"
echo "APP: $APP"
echo "INI: $INI"
echo "ARGS: $@"

for i in {1..8}
do
echo ">>>> Run $i:"

$LAUNCH python $APP -c $INI --tag $COMMIT"."$HOST"_$i" --max_ep $MAX_EP --validate_ep $VALID_EP --model_dir "$MODEL_DIR_PRE" $TRACE $@
#killall python
sleep 15

echo "<<<<"
done

#$LAUNCH python $APP -c $INI --max_ep $MAX_EP --validate_ep $VALID_EP --model_dir $MODEL_DIR_PRE"" --conv_decay 0.0001 --fc_decay 0.0001
#killall python; sleep 15

#$LAUNCH python $APP -c $INI --max_ep $MAX_EP --validate_ep $VALID_EP --model_dir $MODEL_DIR_PRE".var_ne_val" --conv_decay 0.0001 --fc_decay 0.0001
#killall python; sleep 15

#$LAUNCH python $APP -c $INI --max_ep $MAX_EP --validate_ep $VALID_EP --model_dir $MODEL_DIR_PRE".var" --conv_decay 0.0001 --fc_decay 0.0001
#killall python; sleep 15

#$LAUNCH python $APP -c $INI --max_ep $MAX_EP --validate_ep $VALID_EP --model_dir $MODEL_DIR_PRE".match" --conv_decay 0.0001 --fc_decay 0.0001
#killall python; sleep 15

#$LAUNCH python $APP -c $INI --max_ep $MAX_EP --validate_ep $VALID_EP --model_dir $MODEL_DIR_PRE".match_v1" --model_type v1 --shortcut conv --conv_decay 0.0001 --fc_decay 0.0001
#killall python; sleep 15

#$LAUNCH python $APP -c $INI --max_ep $MAX_EP --validate_ep $VALID_EP --model_dir $MODEL_DIR_PRE".match_v1_nozero" --model_type v1 --shortcut conv --conv_decay 0.0001 --fc_decay 0.0001
#killall python; sleep 15

#$LAUNCH python $APP -c $INI --max_ep $MAX_EP --validate_ep $VALID_EP --model_dir $MODEL_DIR_PRE".match_v2_nozero" --model_type v2 --shortcut conv --conv_decay 0.0001 --fc_decay 0.0001
#killall python; sleep 15

#$LAUNCH python $APP -c $INI --tag $COMMIT"."$HOST --max_ep $MAX_EP --validate_ep $VALID_EP --model_dir $MODEL_DIR_PRE $@
#killall python; sleep 15

