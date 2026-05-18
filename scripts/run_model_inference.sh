#!/bin/bash

# redirect the session output
#exec > ./script_logs/out_infer.log 2> ./script_logs/error_infer.log

# running environment
export DISTRIBUTED=false
export REMOTE_NODE=false
export WORKSTATION=false

export DEBUG=true

# tqdm does not work well in log files
#export TQDM_DISABLE=1

# try to reduce CUDA memory fragmentation
export PYTORCH_ALLOC_CONF="expandable_segments:True"

# run the Python interpreter for the main file
uv sync
# exec here takes over shell, python is now main process
exec uv run -m cs_5960_ast --run-id="testing_inference" --load-path="~/Documents/Work/CS5960AST/CS5960AST_repo/models/runs/node/3307283/model_checkpoints/model_state_21.pt" infer
