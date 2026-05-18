#!/bin/bash

# redirect the session output
exec > ./script_logs/out_train_workstation.log 2> ./script_logs/error_train_workstation.log

# running environment
export REMOTE_NODE=false
export WORKSTATION=true

# tqdm does not work well in log files
export TQDM_DISABLE=1

# print out the current utilization and info to the logs being saved
nvidia-smi

# run the Python interpreter for the main file
uv sync
#export CUDA_VISIBLE_DEVICES=2,3
# exec here takes over shell, python is now main process
exec nohup uv run -m cs_5960_ast train &
