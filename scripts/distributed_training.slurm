#!/bin/bash
## slurm settings
#SBATCH --partition=accel
#SBATCH --job-name=cs_5960_ast
#SBATCH --account=ec12
#SBATCH --time=14:00:00
#SBATCH --qos=normal
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=6
#SBATCH --cpus-per-task=16
#SBATCH --gpus-per-task=nvidia_h200_nvl:1
## to specify the 6 GPU node with the newest H200 (could try 6 tasks for this node)
## #SBATCH --gpus-per-task=nvidia_h200_nvl:1
## otherwise GPUs are "a100", "rtx30", "a100_80", "a40", "h100nv", "l40s"
#SBATCH --gpu-bind=none
#SBATCH --mem-per-cpu=1G
#SBATCH --output=/fp/projects01/ec12/ec-ericlu/CS5960AST/script_logs/slurm_out.log
#SBATCH --error=/fp/projects01/ec12/ec-ericlu/CS5960AST/script_logs/slurm_error.log

## redirect the session output - this will redirect python and bash output but not slurm metadata about job
cd /fp/projects01/ec12/ec-ericlu/CS5960AST/
exec > ./script_logs/out_node.log 2> ./script_logs/error_node.log

## should logs be in a folder with $SLURM_JOB_ID ?

set -o errexit  # Exit the script on any error
set -o nounset  # Treat any unset variables as an error

## error file for torch elastic, just in case
TORCHELASTIC_ERROR_FILE=./script_logs/torchelastic_error.log

## running environment, this is for custom python code
export DISTRIBUTED=true
export REMOTE_NODE=true
export WORKSTATION=false

export DEBUG=false

## override for job ID to resume from this checkpoint
#export AST_JOB_ID=3382452

## tqdm does not work well in log files
export TQDM_DISABLE=1

## set env variables for distributed setup - this is required by pytorch
export DIST_BACKEND="nccl"
export DIST_URL="env://"
export MASTER_PORT=11037

## get the first node name as master address (and print the list)
echo "NODELIST="${SLURM_NODELIST}
master_addr=$(scontrol show hostnames "$SLURM_JOB_NODELIST" | head -n 1)
export MASTER_ADDR=$master_addr
echo "MASTER_ADDR="$MASTER_ADDR

## misc multiprocessing settings, required by pytorch and OMP (slurm might use openMP settings too?)
export MAXRESTARTS=0
export OMP_NUM_THREADS=4

## setting for uv, it falls back on the fox system so this just suppresses a warning
export UV_LINK_MODE=copy

## for fox system
module --quiet purge
module load uv/0.7.13-GCCcore-14.2.0

## print out the current gpu utilization and info to the logs being saved
nvidia-smi

# Setup monitoring, from https://www.uio.no/english/services/it/research/platforms/edu-research/help/fox/GPUs/using-gpus.md
#nvidia-smi --query-gpu=timestamp,utilization.gpu,utilization.memory \
#	--format=csv --loop=60 > "models/runs/node/$SLURM_JOB_ID/gpu_util-$SLURM_JOB_ID.csv" &
#NVIDIA_MONITOR_PID=$!  # capture PID of monitoring process

## run the Python interpreter for the main file, via uv
uv sync
srun uv run -m cs_5960_ast train

## script based on:
## https://gist.github.com/TengdaHan/1dd10d335c7ca6f13810fff41e809904
## https://documentation.hpc.uio.no/jobs/job_scripts.html

# clean up monitoring
#kill -SIGINT "$NVIDIA_MONITOR_PID"

## - useful commands: -

## sbatch distributed_training.slurm
## queue a job with this script

## squeue --me
## print users jobs, tells status of currently queued jobs for yourself

## scancel JOBID
## get jobid from squeue --me, then cancel if needed

## squeue --start --me
## squeue --start -j JOBID
## get jobid as above, this is estimate of when the job will start running

## squeue --partition=accel
## the list of queued jobs on the accel (GPU) partition

## see: https://slurm.schedmd.com/sinfo.html

## sinfo -N -o "%N %G %P"
## print a list of NodeName GPU_name:count Partition

## sinfo -N -r -l
## print statistics per node with -l=Long format with extra info and -r=reasons for node being down or similar
## (no status = none)

## sinfo -N -o "%.11P %.11N %.22G %.11l %.5a %.11T %.16F"
## print per node, the node name, partition it belongs to, GPUs it has, max time limit, status (up, down, etc.), %F, state of nodes
## %F: Number of nodes by state in the format "allocated/idle/other/total"

## - slurm run notes -
## When you queue up a run, slurm(?) copies the .sh script you gave it into memory.
## Even if you make changes to distributed_training.slurm, and say commit/push these with git,
## the slurm run that happens will not reflect these changes.
## HOWEVER, when you tell it to go and run python with some repository as a target, no copy is made of that repo.
## Hence, your working copy (HEAD, I guess) at the time when THE JOB STARTS will be the one that is used.
## If you queue a job, then make some changes to Python, and then queue up a second job, if the first did not start yet,
## it will also use the changes made.
## HOWEVER - because of how Python works, the interpreter copies all the active files into memory. So if the Python interpreter started,
## you can make any changes you want and they will not be reflected until the next run starts.

## - slurm allocation notes -
## always select 4 GPUs, should be equivalent to --gpus-per-task=1 with None bind and 4 tasks
## #SBATCH --gres=gpu:4
## 1 GPU per process spawned by slurm, which is nodes * tasks
## #SBATCH --gpus-per-task=1

## ask for 4 gpus on the h200 node: get 6 (0-5) visible in nvidia-smi, get 3 visible to pytorch (fail)
## ask for 5 gpus on the h200 node: get 7 (0-6) visible in nvidia-smi, get 4 visible to pytorch (fail)
## ask for 6 gpus on the h200 node (note, it only HAS 6), get 8 (0-7) visible in nvidia-smi(?), get 6 visible in pytorch (success??)
## ???

## for i in range(device_count):
##     current_accelerator_name = torch.cuda.get_device_name(i)
## this code reports error, missing a GPU and can't get name for it
## fails on getting the properties for the process itself too, print silently fails and only prints default device
## the default device is always GPU:0, the GPU for process 0. Not helpful.

## bind None usually works, fails on h200 node if all 6 GPUs were not requested
## #SBATCH --gpu-bind=none
## Does not restrict visible devices, acts like torchrun

## #SBATCH --gpu-bind=verbose,map:0,1,2,3
## bind GPU 0 to task 0, GPU 1 to task 1, etc. - still gives invalid device ID error when ran

## #SBATCH --gpu-bind=per_task:4
## 4 gpus per task is not what we are doing, it can't find GPU 1-3 and just fails

## #SBATCH --gpu-bind=per_task:1
## problem: this presumably binds each GPU to be GPU:0 for each process which is incompatible with torchrun way of doing things
