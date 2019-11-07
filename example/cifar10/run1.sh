#!/bin/bash

CUR_DIR=$(pwd)

COMMIT=$(git rev-parse --short HEAD)
HOST=$(hostname | tr '-' '\n' | head -n 3 | tail -1)
DATE=$(date +%Y%m%d)

LAUNCH=../launch/launch_gpu_1.sh
#LAUNCH=../launch/launch_multi.sh
APP=cifar10_conv.py
TRACE=--trace

INI_PRE=cifar10
INI=config/$INI_PRE".ini"

MAX_EP=221
VALID_EP=1

MODEL_DIR_PRE=$INI_PRE"."$DATE

echo "COMMIT: $COMMIT"
echo "HOST: $HOST"
echo "DATE: $DATE"
echo "APP: $APP"
echo "INI: $INI"
echo "ARGS: $@"

for i in {1..1}
do
   echo ">>>> Run $i:"

    $LAUNCH python $APP -c $INI --tag $COMMIT"."$HOST"_$i" --max_ep $MAX_EP --validate_ep $VALID_EP --model_dir "$MODEL_DIR_PRE" $TRACE $@
    #killall python
    sleep 15

   echo "<<<<"
done
