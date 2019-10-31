#!/bin/bash

CUR_DIR=`pwd`

PROCESS_COUNT=2
GPUS_PER_NODE=2
NODE_MAX_COUNT=1

NODE_LIST_NAME=( \
    "127.0.0.1" \
    )
NODE_LIST_GPUS=( \
    "$GPUS_PER_NODE" \
    )
NODE_LIST_COUNT=${#NODE_LIST_NAME[*]}

a_dump_list()
{
    echo "Available $NODE_LIST_COUNT nodes:"
    index=0
    while [ $index -lt $NODE_LIST_COUNT ]
    do
        echo ${NODE_LIST_NAME[$index]}":"${NODE_LIST_GPUS[$index]}
        let "index = $index + 1"
    done
}

a_run()
{
    HOST_STR=""
    index=0
    while [ $index -lt $NODE_MAX_COUNT ]
    do
        if [[ !  -z  $HOST_STR  ]]; then
            HOST_STR="$HOST_STR,"
        fi
        HOST_STR="$HOST_STR${NODE_LIST_NAME[$index]}:${NODE_LIST_GPUS[$index]}"
        let "index = $index + 1"
    done
    echo $HOST_STR
    echo mpirun -np $PROCESS_COUNT -H $HOST_STR -mca plm_rsh_args "-p 10022" -x LD_LIBRARY_PATH -x PATH -x TERM $@
    mpirun -np $PROCESS_COUNT -H $HOST_STR -mca plm_rsh_args "-p 10022" -x LD_LIBRARY_PATH -x PATH -x TERM $@
}

a_dump_list

a_run $@

