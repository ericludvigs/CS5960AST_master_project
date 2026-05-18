"""
Main entry point for training and testing the AST model based on EMBER-2 framework

This is the file to adapt for customizing usage
"""

import os  # need os for setting env variable

# error message on laptop:
# NotImplementedError: The operator 'aten::upsample_trilinear3d.out' is not currently implemented for the MPS device.
# If you want this op to be considered for addition please comment on https://github.com/pytorch/pytorch/issues/141287 and
# mention use-case, that resulted in missing op as well as commit hash e2d141dbde55c2a4370fac5165b0561b6af4798b.
# As a temporary fix, you can set the environment variable `PYTORCH_ENABLE_MPS_FALLBACK=1`
# to use the CPU as a fallback for this op.
# WARNING: this will be slower than running natively on MPS.
#
# so to run on laptop, make sure fallback is on
os.environ["PYTORCH_ENABLE_MPS_FALLBACK"] = "1"
# and this option prevents fork poisoning with accelerator checks (apparently)
os.environ["PYTORCH_NVML_BASED_CUDA_CHECK"] = "1"
# recommended when running out of memory, see
# https://docs.pytorch.org/docs/stable/notes/cuda.html#optimizing-memory-usage-with-pytorch-alloc-conf
os.environ["PYTORCH_ALLOC_CONF"] = "expandable_segments:True"
# these env variables must be set before import torch

## actual import block
import argparse
import json
import logging
import random
import sys
import time
from datetime import timedelta
from pathlib import Path
from pprint import pformat

import numpy as np
import torch
import torch.distributed as dist
from torch.distributed.elastic.multiprocessing.errors import record
from torch.nn.parallel import DistributedDataParallel  # noqa: F401 early importing this for distributed training

from cs_5960_ast import default_configs
from cs_5960_ast.default_configs import AstConfigDict
from cs_5960_ast.methods.utils import pretty_timedelta, str_to_bool
from cs_5960_ast.model_code.model_main import ModelWrangler

### Wishlist of features
## Results
# TODO: make a corner plot (like a scatter plot) of the parameters from the style files, to show the sampling is evenly distributed
# TODO: a verification script that saves normalization info and plots to verify the norming process
# TODO: save .npy files from verification such that plots and analysis can be rerun later
## Misc.
# TODO: split out the convolution and loss operations that are used in actual model from utils to somewhere more obvious
# TODO: the normalization is buried with map2map code, when it should be in a higher level norms folder and easily accessible
#       unfortunately, this is a non-trivial change due to callback system used for imports and file discovery
# TODO: increase priority of run id argument if passed, only used as default if not passed, similar to one of the other args
# TODO: add ETA to regular epoch print (just multiply current time by remaining epochs)?
# TODO: consider switching gas and internal energy so the order is switched in plots and mass is clustered together
#       needs changing in all the plots too...
# TODO: potentially, copy redirected STDOUT log files over to the results directory when a run finishes
# TODO: make an environment verification script for main folder, run the info print method and a torch random tensor
#       that way can test whether a node/workstation is working without running training on it, and get the accelerator state
## Inference
# TODO: a prediction/inference method for the ModelWrangler wrapper class, with a switch between G and GE in the input arg.
#       Exposes contained EMBER2 model to an external script which can generate results from multiple EMBER2 models simultaneously
# TODO: a results script that can do a 2x2 grid of input/target and 2 different models for the outputs, rather than repeating
#       the input/target across multiple results.
## Verification, hyperparameters
# TODO: ideally, find some metric that can rate models against one another, and then have proper automated hyperparameter
#       searching, which can probe a space and output the overall performance of each model versus each other.
# TODO: consider verification on checkpoints, verify loss, distribution matching, etc. is getting better
# TODO: parameters/hyperparameters to test for results
#       input filter setup, does [16, 32, 64, 128] give the best results?
#       is 128 or 16 better for fixed channel downsampling? should it be 64?
#       which is better, simple convolutions or current styled modconvs in downsampling?
#       learning rates, do they need changing? (1e-4 seems to not work, 1e-5 needed)
#       optimizer, Adam or AdamW?
## Too complicated, out of scope
# NOTE: multiprocess logs via queue or socket or something
# NOTE: Would like to use power spectrum error as a training target - P(k) calculation is too slow for this currently
# NOTE: consider separating solve env to a completely different script, run on only one process, write config to disk
#       then load config from all other processes. Ensures prints etc. happen only once, and that config is the same everywhere
# NOTE: get CUDA errors via pytorch methods to try and catch/make visible some problems that can happen

## Extra notes
# NOTE: 'batch_size' in map2map does not work.
#       Tweak 'crop' instead to act like batches for sending more or less data to the GPU at once


@record
def main(override_config: AstConfigDict | None = None) -> None:
    """Main entry point for program

    Program sequence is:
    --------------------
    - Get program command line arguments

    - Figure out where program is running
    - Set up config (based on running environment and env variables)

    - Initialize distributed processing if on
    - Actual training
    - Verification, generate plots and initial results <- runs on inference mode too

    - Cleanup (with some summaries)
    """
    args: argparse.Namespace = get_args()

    # based on environment, setup config
    config: AstConfigDict = solve_env(override_config=override_config, args=args)
    logger: logging.Logger = config["logger"]

    logger.debug(f"Input {args=}")

    # set up distributed processing
    if config["distributed"] is True:
        current_device: torch.device = config["device"]
        dist.init_process_group(
            backend=config["dist_backend"],
            init_method=config["dist_url"],
            world_size=config["local_world_size"],
            rank=config["local_rank"],
            device_id=current_device,
        )
    local_rank: int = int(os.getenv("LOCAL_RANK", default="-10"))  # should always be set

    main_model = ModelWrangler(config=config, mode=args.mode)

    if args.mode == "train":
        # on the main process, count the time
        if local_rank == 0:
            start_training_time = time.monotonic()

        # NOTE: actual training is here
        main_model.train_model(epochs=config["num_epochs"])

        if local_rank == 0:
            end_training_time = time.monotonic()

        # free up whatever memory can be freed up since training is done
        torch.cuda.empty_cache()
        # do not try `torch.accelerator.empty_cache()`, results in error on non cuda accelerators which defeats the point
        # RuntimeError: device_allocator INTERNAL ASSERT FAILED at
        # "/Users/runner/work/pytorch/pytorch/pytorch/c10/core/CachingDeviceAllocator.h":110,
        # please report a bug to PyTorch. Allocator for mps is not a DeviceAllocator.

        if local_rank == 0:
            total_training_time = timedelta(seconds=end_training_time - start_training_time)
            logger.info(f"Total training time: {pretty_timedelta(total_training_time)}")

    # this will run during inference too
    main_model.test_model_run()

    if local_rank == 0:
        log_program_end_memory_stats(local_rank, config, logger)

    if config["distributed"]:
        # all processes should call destroy
        dist.destroy_process_group()
    if local_rank == 0:
        logger.info("Run complete.")
    logging.shutdown()
    return None


## supporting methods
def solve_env(
    *,
    override_config: AstConfigDict | None,
    args: argparse.Namespace,
) -> AstConfigDict:
    """Adapted from EMBER-2 methods, work out active environment and settings needed

    Does basically all the program setup work needed
    """
    set_seed(args.rng_seed)

    ## Args and environment variables
    # -----------------------------------------------------------------------------------------------------------------
    # fetch args/settings for program, bools
    debug_prints: bool = args.debug or str_to_bool(os.getenv("DEBUG", default=None))
    distributed_env: bool = str_to_bool(os.getenv("DISTRIBUTED", default=None))
    distributed_status: bool = args.distributed or distributed_env

    # if set via command line, ensure env var is set too
    if distributed_status and not distributed_env:
        os.environ["DISTRIBUTED"] = "1"

    # prefer env variables if set for strings
    dist_backend: str = os.getenv("DIST_BACKEND", default=args.dist_backend)  # arg defaults to "nccl"
    dist_url: str = os.getenv("DIST_URL", default=args.dist_url)  # arg defaults to "env://"

    # set up distributed for both torchrun and slurm
    # always set the torchrun variable (the name is easier to use)
    slurm_local_rank: str | None = os.getenv("SLURM_PROCID", default=None)
    if slurm_local_rank:
        local_rank: int = int(slurm_local_rank)
    else:
        local_rank: int = int(os.getenv("LOCAL_RANK", default=args.local_rank))  # defaults to 0 if arg not passed
    os.environ["LOCAL_RANK"] = str(local_rank)  # always set this
    # on non-distributed it just gets set to 0 and generic code will work correctly

    slurm_world_size: str | None = os.getenv("SLURM_NTASKS", default=None)
    if slurm_world_size:
        total_world_size: int = int(slurm_world_size)
    else:
        total_world_size: int = int(os.getenv("WORLD_SIZE", default=args.world_size))  # arg defaults to -1
    os.environ["TOTAL_WORLD_SIZE"] = str(total_world_size)  # always set this

    slurm_local_world_size: str | None = os.getenv("SLURM_NTASKS_PER_NODE", default=None)
    if slurm_local_world_size:
        local_world_size: int = int(slurm_local_world_size)
    else:
        local_world_size: int = int(os.getenv("LOCAL_WORLD_SIZE", default=args.local_world_size))  # arg defaults to -1
    os.environ["LOCAL_WORLD_SIZE"] = str(local_world_size)  # always set this

    # for pytorch env variables, see: https://docs.pytorch.org/docs/stable/elastic/run.html#environment-variables
    # for slurm, see: https://slurm.schedmd.com/sbatch.html#SECTION_OUTPUT-ENVIRONMENT-VARIABLES

    if distributed_status:
        if local_rank == 0:
            # TODO: is a print because distributed logging is not enabled
            print(f"Distributed training enabled, local rank: {local_rank}, local world size: {local_world_size}", flush=True)
        else:
            print(f"Additional process, local rank: {local_rank}, local world size: {local_world_size}", flush=True)

        if not torch.distributed.is_available():
            msg = f"Error with torch.distributed package, `torch.distributed.is_available()={torch.distributed.is_available()}`"
            raise RuntimeError(msg)
    else:
        local_rank = 0  # ensure local rank is 0 for non-distributed runs, so prints work

    # can run remote with full multiprocessing and multiple gpus, (use all data)
    # on a workstation/desktop with one cpu and gpu, (use subset of data, TODO: consider parameter search here)
    # or on just a laptop (very little data, fast turnaround)
    remote_node_status = str_to_bool(os.getenv("REMOTE_NODE", None))
    workstation_status = str_to_bool(os.getenv("WORKSTATION", None))

    if not remote_node_status and not workstation_status:
        laptop_status = True
    else:
        laptop_status = None
    local_run: bool = not remote_node_status and not workstation_status  # NOTE: kept functionality simple but could be expanded

    # load default settings
    if override_config is not None:
        config: AstConfigDict = override_config
    elif remote_node_status is True:
        config: AstConfigDict = default_configs.remote_node_config
    elif workstation_status is True:
        config: AstConfigDict = default_configs.workstation_config
    elif laptop_status is True:
        config: AstConfigDict = default_configs.laptop_config
    else:
        msg = "Could not determine environment"
        raise RuntimeError(msg)

    # add env variables/input args to config
    config["local_run"] = local_run
    config["distributed"] = distributed_status
    config["local_rank"] = local_rank
    config["local_world_size"] = local_world_size
    config["total_world_size"] = total_world_size
    config["dist_backend"] = dist_backend
    config["dist_url"] = dist_url

    ### Main paths, id of current run
    # -----------------------------------------------------------------------------------------------------------------
    ## ID
    ast_job_id: str | None = os.getenv("AST_JOB_ID", default=None)  # this is "our" var, use over all others
    slurm_job_id: str | None = os.getenv("SLURM_JOB_ID", default=None)  # always check this, use later
    if ast_job_id is None:
        if slurm_job_id:
            run_name: str = slurm_job_id
        else:
            run_name: str = os.getenv("TORCHELASTIC_RUN_ID", default=args.run_id)  # final fallback to "testing_3"
        os.environ["AST_JOB_ID"] = run_name  # always set this
    else:
        run_name = ast_job_id
    config["run_name"] = run_name
    config["slurm_job_id"] = slurm_job_id

    # checkpoint handling
    # NOTE: passing input argument will overwrite existing config entry
    if args.load_state is not None:
        config["load_state"] = args.load_state
    else:
        try:
            config["load_state"]
        except KeyError as e:
            msg = "Could not find `load_state` key in config, with no override input arg provided"
            raise KeyError(msg) from e

    config["load_state_strict"] = args.load_state_strict

    if args.load_path is not None:
        load_path = Path(args.load_path).expanduser().resolve()
        if not load_path.exists():
            msg = f"Provided path for loading checkpoints does not exist: {args.load_path=}"
            raise FileNotFoundError(msg)
        config["checkpoint_load_path"] = load_path
    else:
        config.setdefault("checkpoint_load_path", None)

    ## find where to put the output folder, and make an object for the actual output position
    config["output_dir_path"] = Path(config["output_dir_path"]).expanduser().resolve()
    output_dir = config["output_dir_path"] / run_name
    config["output_dir"] = output_dir

    config["temp_files_path"] = Path(config["temp_files_path"]).expanduser().resolve()

    ## data directory
    config["data_dir_path"] = Path(config["data_dir_path"]).expanduser().resolve()
    data_dir = config["data_dir_path"]
    if not data_dir.exists():
        msg = f"Could not find data directory at path: `{data_dir}`"
        raise FileNotFoundError(msg)

    # add the data directory path to the patterns
    # NOTE: if val config gets added, needs to be added here
    for sub_config in config["train_dataset_config"], config["test_dataset_config"]:
        for key in "in_patterns", "tgt_patterns", "style_pattern":
            pattern_value: str | list[str] | None = sub_config[key]
            if pattern_value is None:
                continue
            if isinstance(pattern_value, str):
                # pattern value as str will never come from in or tgt patterns, so they never get a str assigned back
                # ty raises error on assigning str to those patterns, but that cannot happen
                sub_config[key] = data_dir.as_posix() + "/" + pattern_value  # ty:ignore[invalid-assignment]
            elif isinstance(pattern_value, list):
                sub_config[key] = [data_dir.as_posix() + "/" + p for p in pattern_value]  # ty:ignore[invalid-assignment]
            else:
                msg = f"Unrecognized type for data pattern config `{key}`: {type(pattern_value)}"
                raise TypeError(msg)
                # NOTE: could do Path support here but not a priority

    ## Plots
    figure_dir = output_dir / Path("output_figures/")
    config["figure_dir"] = figure_dir

    ## Make these directories
    # resolve this path just to be sure its going in the right place
    abs_output_path = output_dir.resolve()
    abs_output_path.mkdir(parents=True, exist_ok=True)  # NOTE: we CAN use an existing run folder to load their checkpoint

    # save the dir for later access
    os.environ["OUTPUT_DIR"] = abs_output_path.as_posix()

    # prepare a log file for later
    os.environ["DATAFILES_LOG_FILENAME"] = "run_datafiles_info.log"
    # clear previous contents
    Path.open(output_dir / "run_datafiles_info.log", "w").write("")

    abs_fig_path = figure_dir.resolve()
    abs_fig_path.mkdir(parents=True, exist_ok=True)
    (abs_fig_path / ".placeholder").touch(exist_ok=True)  # make a placeholder file so this folder is included in version control

    ### Setup logging
    # -----------------------------------------------------------------------------------------------------------------
    if local_rank == 0:
        logger: logging.Logger = setup_logging(debug_status=debug_prints, log_file_path=output_dir.resolve())
    else:
        # TODO: set up logging properly for the other processes
        logger: logging.Logger = logging.getLogger("blank")
        logger.addHandler(logging.NullHandler())

    config["logger"] = logger

    ## now that logger is up, write the logs from earlier code
    # we have to set up log late so it can write to the correct run_id folder
    if local_rank == 0:
        logger.debug("- Debug prints enabled -")  # only prints if debug is on

        if override_config is not None:
            logger.info("- Using custom config -")
        if remote_node_status is True:
            logger.info("- Remote node environment set, using remote node config -")
        elif workstation_status is True:
            logger.info("- Workstation/desktop environment set, using workstation config -")
        elif laptop_status is True:
            logger.info("- No environment set, using default laptop config -")
        logger.info(f"Using settings: {config['config_name']}")

        logger.debug(f"LOCAL_RANK var: {os.getenv('LOCAL_RANK')}, program argument: {args.local_rank}")
        logger.debug(f"SLURM_PROCID var: {slurm_local_rank}, program argument: {args.local_rank}")

        logger.info(f"AST job id: {os.getenv('AST_JOB_ID')}")
        logger.debug(f"Slurm job id: {os.getenv('SLURM_JOB_ID')}")
        logger.debug(f"Torch env id: {os.getenv('TORCHELASTIC_RUN_ID')}")

    ### Detect accelerator
    # -----------------------------------------------------------------------------------------------------------------
    # only local runs should fall back to CPU, explicit error for unattended runs
    if not local_run and not torch.accelerator.is_available():
        msg: str = "PyTorch accelerator (GPU/TPU) could not be detected/accessed during remote run, environment must be fixed"
        raise RuntimeError(msg)
    # on a laptop, use accelerator if available
    elif torch.accelerator.is_available():
        device_type: str = torch.accelerator.current_accelerator().type  # ty:ignore[unresolved-attribute]
        # current accelerator can not be None if accelerator is available, type checker struggles with that

        if slurm_job_id:  # slurm is configured to see all gpus but that is not always the case
            current_device: torch.device = torch.device(type=device_type, index=local_rank)
        else:
            current_device: torch.device = torch.device(type=device_type, index=local_rank)
        # this is a bit messy, but it is because current accelerator method does not return device index

        # NOTE: torch.accelerator.current_device_index() always seems to return 0, which is unhelpful

        if current_device is None:
            msg = "Accelerator available but device is None, which should not be possible"
            raise RuntimeError(msg)

        logger.debug(f"PyTorch device: {current_device}, on process {local_rank}")  # change to INFO with multiprocess logging
        print(f"PyTorch device: {current_device}, on process {local_rank}", flush=True)  # print on all processes
    else:
        current_device: torch.device = torch.device(device="cpu")
        logger.info("PyTorch device: CPU only")

    config["device"] = current_device

    # output info about GPUs
    print_accelerator_info(cfg=config, device=current_device, logger=logger)

    ### Parallellization settings
    # -----------------------------------------------------------------------------------------------------------------
    # find out how many cpu cores are available
    cpu_count = os.cpu_count()
    if cpu_count is None:
        # user needs to figure this one out
        msg = f"Could not determine number of CPU cores available, {os.cpu_count()=}"
        raise RuntimeError(msg)
    config["detected_cpu_threads"] = cpu_count  # in case info on number of cpu cores is interesting
    # check if this key exists
    try:
        config["dataloader_num_workers"]
    except KeyError as e:
        msg: str = "Program requires `dataloader_num_workers` (cpu threads to use for dataloader) to be set in config"
        raise RuntimeError(msg) from e
    # if running on a laptop, user sets manually
    # setting to all detected cores locks up weaker systems

    # LOCAL_WORLD_SIZE: The local world size (e.g. number of workers running locally);
    # equals to --nproc-per-node specified on torchrun
    # we want to distribute cpu threads based on this, but probably don't need more than 8 threads per GPU
    local_world_size = int(os.getenv("LOCAL_WORLD_SIZE", default="1"))

    # configurable cap on how many threads can be used
    max_usable_threads_per_gpu_cap = config["max_usable_threads_per_gpu_cap"]
    # LOCAL_WORLD_SIZE can be -1, if not configured
    # therefore, use a divisor that must be larger than 1
    max_usable_threads_per_process = min(max_usable_threads_per_gpu_cap, cpu_count // max(1, local_world_size))
    config["max_usable_threads_per_gpu"] = max_usable_threads_per_process
    os.environ["OMP_NUM_THREADS"] = str(max_usable_threads_per_process)  # and set this env var with info for OpenMP

    ### Misc settings
    # -----------------------------------------------------------------------------------------------------------------
    config["debug_prints"] = debug_prints
    config["pin_memory"] = True  # pretty much always want this, doesn't work on MPS but torch has its own fallback set up

    ## For laptop runs, override data settings
    # had problems with model no longer running well locally, hence quick fix
    if local_run:
        config["train_dataset_config"]["crop"] = 32
        # should result in one batch with 64 datapoints instead of the full field for testing
        config["test_dataset_config"]["crop"] = 64
        config["test_dataset_config"]["crop_start"] = 64
        config["test_dataset_config"]["crop_stop"] = 128
    # crashing with 16GB...
    # if config["config_name"] == "workstation_config":
    #    config["train_dataset_config"]["crop"] = 100
    #    config["test_dataset_config"]["crop"] = 100
    #    config["test_dataset_config"]["crop_start"] = 100
    #    config["test_dataset_config"]["crop_stop"] = 200
    # NOTE: override to crop settings for workstation inference here

    ### Explicitly write out the config setup being used for this run
    # -----------------------------------------------------------------------------------------------------------------
    config_file_dump_path = abs_output_path / "run_config.json"
    if local_rank == 0:
        with config_file_dump_path.open("w") as f:
            json.dump(obj=config, fp=f, indent=4, default=set_default)
        logger.debug(f"Wrote run config to: {config_file_dump_path}")

    ### Output the full list of environment variables for debug purposes
    # -----------------------------------------------------------------------------------------------------------------
    if args.debug_env_var_dump and local_rank == 0:
        env_var_dump_path = abs_output_path / "debug_env_vars.txt"
        with env_var_dump_path.open("w") as f:
            for key, value in os.environ.items():
                f.write(f"{key}={value}\n")

    if local_rank == 0:
        logger.debug(f"Script running from: {Path.cwd().resolve()}")

    return config


def print_accelerator_info(cfg: AstConfigDict, device: torch.device, logger: logging.Logger) -> None:
    """Output info about the current accelerator (GPU) setup

    Main prints are done only on process 0, but the other processes will log accelerator info too
    """
    local_rank: int = cfg["local_rank"]

    if local_rank == 0:
        logger.info("")
        logger.info("------------------------------------------")
        logger.info(f"Torch version: {torch.version.__version__}")
    if torch.accelerator.is_available():
        device_type = device.type if device is not None else "unknown"  # device should not end up unknown but just in case
        # total number of accelerators we should have for bookkeeping
        device_count = cfg["local_world_size"] if cfg["slurm_job_id"] else torch.accelerator.device_count()

        if device_type == "cuda" and local_rank == 0:
            logger.info(f"CUDA version: {torch.version.cuda}")

        cfg["torch_version"] = torch.version.__version__
        cfg["cuda_version"] = torch.version.cuda if device_type == "cuda" else "N/A"
        cfg["accelerator_count"] = device_count

        if cfg["cuda_version"] is None:
            msg = f"CUDA version is None for some reason, {torch.version.cuda=}"
            raise RuntimeError(msg)

        visible_devices = torch.accelerator.device_count()
        if local_rank == 0:
            logger.info(f"Number of accelerators (GPUs) processes should have visible: {device_count}")
        print(f"Process {local_rank}, sees {visible_devices} GPUs, has set PyTorch device: {device}", flush=True)  # TODO: debugging

        # TODO: debug slurm allocation
        if device_type == "cuda":
            print(
                (
                    f"The properties of device {local_rank} for process {local_rank} is: "
                    f"\n    {torch.cuda.get_device_properties(local_rank)}"
                ),
                flush=True,
            )
        # figure out the accelerator for the active process and print info
        # build up a string for log gradually, to ensure it prints out all at once
        # done for all processes
        current_accelerator = torch.cuda.get_device_name(device) if device_type == "cuda" else device_type
        full_accelerator_info: list[str] = []
        full_accelerator_info.append(
            f"Accelerator info from process {local_rank}\n| Accelerator {local_rank}: {current_accelerator}"
        )

        # this unified version should work according to spec, but in practice is unimplemented and fails
        # so, manual if statements it is
        # print(f"Allocated: {torch.accelerator.memory.memory_allocated(i) / 1024**3:.2f} GB")
        # print(f"Cached:    {torch.accelerator.memory.memory_reserved(i) / 1024**3:.2f} GB")
        if device_type == "mps":
            alloc = torch.mps.current_allocated_memory()
            alloc_driver = torch.mps.driver_allocated_memory()
            # for total allocated memory, also include the fixed allocation for initializing the driver
            full_accelerator_info.append(f"| Recommended max memory: {torch.mps.recommended_max_memory() / 1024**3:.2f} GB")
            full_accelerator_info.append(f"| Reserved memory: {(alloc + alloc_driver) / 1024**3:.2f} GB")
        if device_type == "cuda":
            total_memory_bytes = torch.cuda.get_device_properties(device).total_memory
            total_memory_str = f"{total_memory_bytes / 1024**3:.2f} GB"

            full_accelerator_info.append(f"| Total memory: {total_memory_str}")
            full_accelerator_info.append(f"| Allocated: {torch.cuda.memory_allocated(device) / 1024**3:.2f} GB")
            full_accelerator_info.append(f"| Reserved: {torch.cuda.memory_reserved(device) / 1024**3:.2f} GB")
            free_mem, total_mem = torch.cuda.mem_get_info(device)
            full_accelerator_info.append(f"| Free/total memory: {free_mem / 1024**3:.2f} GB / {total_mem / 1024**3:.2f} GB")
            full_accelerator_info.append(f"| Memory Utilization: {1 - (free_mem / total_mem):.2%}")
            full_accelerator_info.append("")  # this will add a newline at the end

        full_accelerator_info_str = "\n".join(full_accelerator_info)
        logger.debug(full_accelerator_info_str)  # TODO: change to .info with distributed logging
        # TODO: is a print because no distributed logging
        print(full_accelerator_info_str, flush=True)

        # log to process 0 config the info from the extra accelerators
        if local_rank == 0 and device_type == "cuda":
            for i in range(device_count):
                current_accelerator_name = torch.cuda.get_device_name(i)

                total_memory_bytes: float = torch.cuda.get_device_properties(i).total_memory
                total_memory_str = f"{total_memory_bytes / 1024**3:.2f} GB"

                cfg[f"accelerator_{i}_name"] = current_accelerator_name  # ty:ignore[invalid-key] allow these extra keys
                cfg[f"accelerator_{i}_total_memory_GB"] = total_memory_str  # ty:ignore[invalid-key]
                logger.debug(f"| Full CUDA properties for accelerator {i}:\n|    {torch.cuda.get_device_properties(i)}")
        elif local_rank == 0 and device_type == "mps":
            for i in range(device_count):
                current_accelerator_name = device_type

                total_memory_bytes = torch.mps.recommended_max_memory()
                total_memory_str = f"{total_memory_bytes / 1024**3:.2f} GB"

                cfg[f"accelerator_{i}_name"] = current_accelerator_name  # ty:ignore[invalid-key] allow these extra keys
                cfg[f"accelerator_{i}_total_memory_GB"] = total_memory_str  # ty:ignore[invalid-key]

    elif local_rank == 0:
        # this only triggers for laptop fallback, remote run has errored already
        logger.info("No accelerator (GPU) detected, running on CPU only.")
    if local_rank == 0:
        logger.info("------------------------------------------\n")
    return None


def log_program_end_memory_stats(local_rank: int, cfg: AstConfigDict, logger: logging.Logger) -> None:
    """Log the peak memory usage and the torch.cuda.memory_summary()

    Intended to be used after program has completed
    """
    # early exit on non-main process
    if local_rank != 0:
        return None

    if cfg["device"].type == "cuda":
        logger.info("Summary of memory use in program:")
        # print some manually selected usage stats for each accelerator, aka each process
        for i in range(cfg["accelerator_count"]):
            mem_stats = torch.accelerator.memory_stats(i)

            name: str = cfg[f"accelerator_{i}_name"]  # ty:ignore[invalid-key]
            total_mem_bytes: float = torch.cuda.get_device_properties(i).total_memory
            total_mem_gb: float = total_mem_bytes / 1024**3
            total_mem_gb_str: str = cfg[f"accelerator_{i}_total_memory_GB"]  # ty:ignore[invalid-key]
            peak_reserved_gb = mem_stats["reserved_bytes.all.peak"] / 1024**3
            current_reserved_gb = mem_stats["reserved_bytes.all.current"] / 1024**3

            logger.info(f"Accelerator: {name}")
            logger.info(f"| Total memory: {total_mem_gb_str}")
            logger.info(f"| Peak allocated memory: {mem_stats['allocated_bytes.all.peak'] / 1024**3:.2f} GB")
            logger.info(f"| Peak reserved memory: {peak_reserved_gb:.2f} GB")
            logger.info(f"| Peak used/total memory: {peak_reserved_gb:.2f} GB / {total_mem_gb:.2f} GB")
            logger.info(f"| Peak memory utilization: {(peak_reserved_gb / total_mem_gb):.2%}")
            logger.info(f"| Current used/total memory: {current_reserved_gb:.2f} GB / {total_mem_gb:.2f} GB")
            logger.info(f"| Current memory utilization: {(current_reserved_gb / total_mem_gb):.2%}")
            # full debug memory stats
            logger.debug(f"\n{pformat(mem_stats)}\n")
        logger.debug(f"`torch.cuda.memory_summary`:\n{torch.cuda.memory_summary(device=None, abbreviated=False)}")

    # despite being a .accelerator method, memory_stats() fails on mps
    # logging peak memory usage is much harder outside of CUDA...
    # so for mps, print only current usage
    elif local_rank == 0 and cfg["device"].type == "mps":
        for i in range(cfg["accelerator_count"]):
            name = cfg[f"accelerator_{i}_name"]  # ty:ignore[invalid-key]
            total_mem_bytes = torch.mps.recommended_max_memory()
            total_mem_gb: float = total_mem_bytes / 1024**3
            total_mem_gb_str = cfg[f"accelerator_{i}_total_memory_GB"]  # ty:ignore[invalid-key]

            alloc = torch.mps.current_allocated_memory()
            alloc_driver = torch.mps.driver_allocated_memory()
            current_reserved_gb = (alloc + alloc_driver) / 1024**3

            logger.info(f"Accelerator: {name}")
            logger.info(f"| Total memory: {total_mem_gb_str}")
            logger.info(f"| Current used/total memory: {current_reserved_gb:.2f} GB / {total_mem_gb:.2f} GB")
            logger.info(f"| Current memory utilization: {(current_reserved_gb / total_mem_gb):.2%}")
    return None


def set_default(obj: torch.device | Path | logging.Logger) -> str:
    """Helper for serializing json, these classes have a str representation that works for written files"""
    if isinstance(obj, torch.device):
        return str(obj)
    elif isinstance(obj, Path):
        return obj.as_posix()
    elif isinstance(obj, logging.Logger):
        return obj.name
    # still error on unknowns
    msg = f"Unknown type for serialization encountered: {type(obj)=}"
    raise TypeError(msg)


def set_seed(seed: int | None = None) -> None:
    """Set random seed for more reproducible results

    seed=None is same as default behaviour without explicitly setting seeds, effectively skipping the method
    """
    random.seed(seed)
    np_rng: np.random.Generator = np.random.default_rng(seed)

    # torch does not support the standard pass None to get default behaviour...
    if seed is not None:
        torch_rng: torch.Generator = torch.manual_seed(seed)
    else:
        # might as well reuse the numpy generator to set a random seed in the allowed range
        torch_rng: torch.Generator = torch.manual_seed(np_rng.integers(0, 2**32 - 1))  # noqa: F841


def setup_logging(*, debug_status: bool = False, log_file_path: Path | None = None) -> logging.Logger:
    """All the logging configuration that's necessary"""
    logger = logging.getLogger("cs_5960_ast")
    logging.captureWarnings(capture=True)  # turn on warnings.warn messages being captured and logged also to our logfiles

    debug_filename = "debug_log.log"
    info_filename = "run_log.log"
    filename = debug_filename if debug_status else info_filename

    console_handler = logging.StreamHandler(sys.stdout)
    if log_file_path is not None:
        file_handler = logging.FileHandler(filename=log_file_path / filename, mode="w")
    else:
        file_handler = logging.NullHandler()
    console_handler.setLevel(logging.INFO)
    file_handler.setLevel(logging.DEBUG)

    console_formatter = logging.Formatter("%(message)s")
    file_formatter = logging.Formatter("%(asctime)-24s - %(name)s - %(levelname)-8s | %(message)s", datefmt="%Y-%m-%dT%H:%M:%S%z")

    # distributed runs should be ran via script and with redirected stdout, so write the full logging info to those runs
    # mostly interested in timestamps
    dist_status = str_to_bool(os.getenv("DISTRIBUTED", default=None))
    if dist_status:
        console_formatter = file_formatter

    console_handler.setFormatter(console_formatter)
    file_handler.setFormatter(file_formatter)

    logger.addHandler(console_handler)
    logger.addHandler(file_handler)

    # set root to debug to pass everything to a handler
    if debug_status:
        logger.setLevel(logging.DEBUG)
    else:
        logger.setLevel(logging.INFO)

    return logger


def get_args() -> argparse.Namespace:
    """Parse arguments and set runtime defaults."""
    parser = argparse.ArgumentParser(description="CS5960AST Model Trainer/Tester")

    parser_basics_group = parser.add_argument_group("basic flags")
    parser_basics_group.add_argument(
        "--run-id",
        "--run_id",
        type=str,
        default="testing_3",
        help=(
            "Identifier for this program run, used for output directory naming. "
            "Will be used for saving and loading checkpoints by default"
        ),
    )
    parser_basics_group.add_argument(
        "--rng-seed",
        "--rng_seed",
        "--seed",
        type=int,
        default=None,
        help="Optionally set random seed for reproducibility, defaults to None which gives random seed",
    )

    # whether to enable some extra logging
    parser_debug_group = parser.add_argument_group("debug flags")
    parser_debug_group.add_argument(
        "--debug", "--debug-prints", "--debug_prints", action="store_true", help="Enable extra debug logging"
    )
    parser_debug_group.add_argument(
        "--debug-env-var-dump",
        "--debug_env_var_dump",
        "--devd",
        action="store_true",
        help="If passed, dump active environment variables to file for debugging program state",
    )

    parser_loading_group = parser.add_argument_group("checkpoint loading flags")
    parser_loading_group.add_argument(
        "--load-state",
        action=argparse.BooleanOptionalAction,
        help=(
            "Whether to load a prior saved model checkpoint, if one exists. Enabled by default.\n"
            "Pass `--no-load-state` to disable, and start training from scratch"
        ),
    )
    # set to None such that bool result is only used if arg is present
    parser_loading_group.set_defaults(load_state=None)
    parser_loading_group.add_argument(
        "--load-state-non-strict",
        action="store_false",
        help="Allow incompatible keys when loading model states, defaults to strict loading",
        dest="load_state_strict",
    )
    parser_loading_group.set_defaults(load_state_strict=True)
    parser_loading_group.add_argument(
        "--load-path",
        type=str,
        help=(
            "Path to a model checkpoint to load, overrides loading from job ID directory. "
            "Can directly specify a checkpoint file, or a folder in which case latest file there is loaded"
        ),
    )

    # arguments for distributed training
    parser_distributed_group = parser.add_argument_group("distributed training flags")
    parser_distributed_group.add_argument("--distributed", "--dist", action="store_true", help="Enable distributed training")
    parser_distributed_group.add_argument(
        "--dist-backend",
        "--dist_backend",
        type=str,
        default="nccl",
        # in theory can use dist.get_default_backend_for_device(device=torch.accelerator.current_accelerator(check_available=True))
        # but is unreliable in practice
        help="`torch.distributed` backend, 'nccl' is recommended for CUDA. Set by default",
    )
    parser_distributed_group.add_argument(
        "--dist-url",
        "--dist_url",
        type=str,
        default="env://",
        help="`torch.distributed` argument: URL specifying how to initialize the process group. Defaults to `env://`",
    )
    parser_distributed_group.add_argument("--local-rank", "--local_rank", type=int, default=0, help="Rank of current process")
    parser_distributed_group.add_argument(
        "--local-world-size",
        "--local_world_size",
        type=int,
        default=-1,
        help=(
            "Number of processes participating in the local job (aka per node). "
            "If not set, rank can be inferred from environment variables"
        ),
    )
    parser_distributed_group.add_argument(
        "--world-size",
        "--world_size",
        type=int,
        default=-1,
        help="Total number of processes across all nodes. If not set, inferred by env variables",
    )

    subparsers = parser.add_subparsers(
        title="modes", dest="mode", required=True, description="Select whether to start training the model or run inference/testing"
    )
    train_parser = subparsers.add_parser(  # noqa: F841
        "train",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    test_parser = subparsers.add_parser(  # noqa: F841
        "infer",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    # if specific flags are needed for each mode at some point use these parsers

    args = parser.parse_args()
    return args


if __name__ == "__main__":
    # it should be possible to import the CS5960AST module, take the main() function and set your own config
    # then the whole setup can be used from a different (user) python script
    override_config = None
    main(override_config=override_config)
