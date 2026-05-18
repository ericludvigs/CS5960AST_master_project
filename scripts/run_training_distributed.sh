#!/bin/bash

# redirect the session output
exec > ./script_logs/out_dist.log 2> ./script_logs/error_dist.log

# error file for torch elastic
TORCHELASTIC_ERROR_FILE=./script_logs/torchelastic_error.log

# running environment
export DISTRIBUTED=true
export REMOTE_NODE=false
export WORKSTATION=true

export DEBUG=true

# override for job ID, this is the highest priority ID
#export AST_JOB_ID="c379351b-d346-49dc-a4d6-c3b1b36640b8"

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

# try to reduce CUDA memory fragmentation
export PYTORCH_ALLOC_CONF="expandable_segments:True"

# print out the current utilization and info to the logs being saved
nvidia-smi

# for fox system
#module load uv/0.7.13-GCCcore-14.2.0

# Run the Python interpreter for the main file
uv sync
# exec here takes over shell, python is now main process
exec nohup uv run torchrun --standalone --nnodes=$NNODES --nproc-per-node=$NPROCPERNODE --max-restarts=$MAXRESTARTS -m cs_5960_ast train &
