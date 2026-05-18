#!/bin/bash

# redirect the session output
exec > ./script_logs/out_infer_workstation.log 2> ./script_logs/error_infer_workstation.log

# running environment
export DISTRIBUTED=true
export REMOTE_NODE=false
export WORKSTATION=true

export DEBUG=true

# override for job ID, this is the highest priority ID
export AST_JOB_ID="inference_3468780_smudgy_GAN"

# tqdm does not work well in log files
export TQDM_DISABLE=1

# set env variables for distributed setup
export DIST_BACKEND="nccl"
export DIST_URL="env://"
export MASTER_PORT=11037

# set how many nodes to use (easier to change here)
export NNODES=1
export NPROCPERNODE="gpu"
export MAXRESTARTS=0
export OMP_NUM_THREADS=4

# setting for uv, just suppresses a warning
export UV_LINK_MODE=copy

# try to reduce CUDA memory fragmentation
export PYTORCH_ALLOC_CONF="expandable_segments:True"

# print out the current utilization and info to the logs being saved
nvidia-smi

# run the Python interpreter for the main file
uv sync
# exec here takes over shell, python is now main process
exec nohup uv run torchrun --standalone --nnodes=$NNODES --nproc-per-node=$NPROCPERNODE --max-restarts=$MAXRESTARTS -m cs_5960_ast --load-path="/mn/stornext/d5/data/ericlu/CS5960AST/models/runs/node/3468780/model_checkpoints/model_state_161.pt" infer &
