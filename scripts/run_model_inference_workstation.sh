#!/bin/bash

# redirect the session output
#exec > ./script_logs/out_infer_workstation.log 2> ./script_logs/error_infer_workstation.log

# running environment
export DISTRIBUTED=false
export REMOTE_NODE=false
export WORKSTATION=true

export DEBUG=true

# override for job ID, this is the highest priority ID
export AST_JOB_ID=testing_inference

# tqdm does not work well in log files
#export TQDM_DISABLE=1

# setting for uv, just suppresses a warning
export UV_LINK_MODE=copy

# try to reduce CUDA memory fragmentation
export PYTORCH_ALLOC_CONF="expandable_segments:True"

# run the Python interpreter for the main file
uv sync
# exec here takes over shell, python is now main process
exec uv run -m cs_5960_ast --load-path="/mn/stornext/d5/data/ericlu/CS5960AST/models/runs/node/3133149/model_checkpoints/model_state_101.pt" infer

exit
