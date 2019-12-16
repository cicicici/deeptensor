#!/bin/bash

CUR_DIR=$(pwd)

COMMIT=$(git rev-parse --short HEAD)
HOST=$(hostname | tr '-' '\n' | head -n 3 | tail -1)
DATE=$(date +%Y%m%d)

#LAUNCH=../launch/launch_gpu_4.sh
LAUNCH=../launch/launch_multi.sh
APP=imagenet_conv.py
TRACE=--trace

INI_PRE=$1
if [[ -z "$INI_PRE" || "$INI_PRE" == "-" ]]; then
    echo "$0 <config prefix> [-n <name>]"
    exit
fi
INI=config/$INI_PRE".ini"
set -- "${@: 2: $#}"

if [[ "$1" == "-n" ]]; then
    MODEL_DIR_PRE=$INI_PRE"."$2
    set -- "${@: 3: $#}"
else
    MODEL_DIR_PRE=$INI_PRE"."$DATE
fi
echo "Args: $@"

MAX_EP=450
VALID_EP=1

echo "COMMIT: $COMMIT"
echo "HOST: $HOST"
echo "DATE: $DATE"
echo "APP: $APP"
echo "INI: $INI"
echo "ARGS: $@"

for i in {1..4}
do
echo ">>>> Run $i:"

echo $LAUNCH python $APP -c $INI --tag $COMMIT"."$HOST"_$i" --max_ep $MAX_EP --validate_ep $VALID_EP --model_dir "$MODEL_DIR_PRE" $TRACE $@
$LAUNCH python $APP -c $INI --tag $COMMIT"."$HOST"_$i" --max_ep $MAX_EP --validate_ep $VALID_EP --model_dir "$MODEL_DIR_PRE" $TRACE $@

sleep 5

echo "<<<<"
done

