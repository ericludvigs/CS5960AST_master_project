"""Config file for the default run configurations

Uses simple python dicts rather than json/toml/whatever for convenience and because of (later) path handling

Also makes overrides easy
"""

from pathlib import Path
from typing import TYPE_CHECKING, Literal, NotRequired, ReadOnly, TypedDict

if TYPE_CHECKING:
    import logging

    import torch


### Config definitions, necessary arguments ###


class AstDatasetConfigDict(TypedDict):
    """For annotating what a dataset config dict for this program should look like

    Most options correspond to map2map field dataset inputs
    """

    dataset_name: ReadOnly[str]

    ## map2map
    # glob patterns for finding files
    in_patterns: list[str]
    tgt_patterns: list[str]
    style_pattern: str | None

    # extra filtering via regex, more features than simple glob pattern
    # the filter will be applied in the same order as input patterns
    # a mismatch between the input and filter will lead to 0 files found
    in_filter_patterns: list[str] | None
    tgt_filter_patterns: list[str] | None
    style_filter_pattern: str | None

    # custom addition to map2map, each input field pattern will be matched with a specific norm function
    # allows different field types to each get a dedicated norm
    # this should be a list of norm functions that will be applied in that order
    in_norms: list[str] | None
    tgt_norms: list[str] | None

    # map2map args used in fields.py for the fielddataset setup
    callback_at: str | Path | None
    # directory of custom code defining callbacks for models, (models not relevant)
    # norms, criteria, and optimizers. Disabled if not set.
    # This is appended to the default locations,
    # thus has the lowest priority
    augment: bool | None
    # enable data augmentation of axis flipping and permutation
    aug_shift: int | tuple[int, ...] | None
    # data augmentation by shifting cropping by [0, aug_shift) pixels,
    # useful for models that treat neighboring pixels differently,
    # e.g. with strided convolutions.
    # Comma-sep. list of 1 or d integers
    aug_add: float | None
    # additive data augmentation, (normal) std, same factor for all fields
    aug_mul: float | None
    # multiplicative data augmentation, (log-normal) std, same factor for all fields

    crop: int | tuple[int, ...] | None
    # size to crop the input and target data. Default is the field size.
    # Comma-sep. list of 1 or d integers
    crop_start: int | tuple[int, ...] | None
    # starting point of the first crop. Default is the origin.
    # Comma-sep. list of 1 or d integers
    crop_stop: int | tuple[int, ...] | None
    # stopping point of the last crop. Default is the opposite corner to the origin.
    # Comma-sep. list of 1 or d integers
    crop_step: int | tuple[int, ...] | None
    # spacing between crops. Default is the crop size.
    # Comma-sep. list of 1 or d integers
    in_pad: int | tuple[int, ...]
    # size to pad the input data beyond the crop size, assuming periodic boundary condition.
    # Comma-sep. list of 1, d, or dx2
    # integers, to pad equally along all axes, symmetrically on each,
    # or by the specified size on every boundary, respectively
    tgt_pad: int | tuple[int, ...]
    # size to pad the target data beyond the crop size, assuming
    # periodic boundary condition, useful for super-resolution.
    # Comma-sep. list with the same format as --in-pad
    scale_factor: float
    # upsampling factor for super-resolution,
    # in which case crop and pad are sizes of the input resolution

    ## map2map sampler
    shuffle: bool
    # whether to randomize order of data samples
    div_data: bool
    # enable data division among GPUs for better page caching.
    # Data division is shuffled every epoch.
    # Only relevant if there are multiple crops in each field
    div_shuffle_dist: int
    # distance to further shuffle cropped samples relative to
    # their fields, to be used with --div-data.
    # Only relevant if there are multiple crops in each file.
    # The order of each sample is randomly displaced by this value.
    # Setting it to 0 turn off this randomization, and setting it to N
    # limits the shuffling within a distance of N files.
    # Change this to balance cache locality and stochasticity

    ## config for power spectrum calculation
    # not required for training data (does not calculate a power spectrum),
    # but is with test data (assuming you want a power spectrum)
    BoxSize: NotRequired[float]
    MAS: NotRequired[str]


class AstNetConfigDict(TypedDict):
    """For annotating the neural network config

    Options generally correspond to inputs in EMBER-2 model code
    """

    style_dim: int  # dimensions of style representation w to be used in network
    context_dim: int  # dimensions in the context vector passed to the network, would be 1 for passing just redshift z
    # see section 3.2.2 in ember2 paper, the mapping network specifically. Context is mapped to style, style passed in.
    style_depth: int  # layers in MappingNet for mapping style information to convolutions
    inp_channels: int  # number of channels for input
    out_channels: int  # number of channels to output
    filters: list[int]  # the filter setup for the layers
    # ember2 selects a start filter size (like 32), then 2**i from there up to maximum size of 128,
    # and repeats until num_layers is hit. Example, [32, 64, 128, 128, 128] with start 32 and num_layers = 5
    num_noise_ch: int  # number of channels of noise to generate and pass through mappingnet
    # after net will make a latent tensor of 2*noise channel size which is concatenated to field in upblocks
    # see map2map paper for more detail on this approach
    beta_ema: float  # input param for effective moving average
    lr_g: float  # learning rate for generator
    lr_d: float  # learning rate for discriminator
    betas_g: tuple[float, float]  # betas for Adam/AdamW optimizer for generator
    betas_d: tuple[float, float]  # betas for Adam/AdamW optimizer for discriminator
    use_noise: bool  # whether to inject noise during training
    # If True, will enable a separate noise generator network, if False disables all noise
    # there is functionality for passing a tensor of noise when calling G, this will bypass the noise generation network,
    # and simply concatenate the noise as is. But from the config level, only True or False is currently supported
    use_adversarial: bool  # whether to enable the discriminator part of the network


class AstConfigDict(TypedDict):
    """For annotating what a config dict for this program should look like

    NotRequired are filled in at runtime
    """

    ## Basics
    config_name: ReadOnly[str]
    local_run: NotRequired[bool]  # basically, whether hands-on or hands-off training
    debug_prints: NotRequired[bool]  # not a configurable setting, it's info on whether debug was enabled via program args

    ## Training basics
    batch_size: Literal[1]  # NOTE: batch size is broken, map2map problem. Just use 1.
    shuffle_dataloader: bool  # handled by map2map sampler
    num_epochs: int
    checkpoint_freq: int  # each X epochs, checkpoint and save model
    batch_break_point: int | None  # after X data samples, exit early
    # use for faster (but worse) training when testing model setup and results format

    ## CPU management
    detected_cpu_threads: NotRequired[int]
    max_usable_threads_per_gpu_cap: int  # will not use more than X threads per GPU
    max_usable_threads_per_gpu: NotRequired[int]  # calculated by threads // num of GPUs
    dataloader_num_workers: int  # how many threads to use for dataloader
    # NOTE: loads a batch into memory per thread, many threads will overload systems with little RAM (like a laptop)

    # logging container to pass through the program
    logger: NotRequired["logging.Logger"]

    ## Checkpoint settings
    load_state: NotRequired[bool]  # whether to load a prior saved model checkpoint, if one exists
    load_state_strict: NotRequired[bool]  # whether to allow incompatible keys when loading model states
    checkpoint_load_path: NotRequired[Path | None]  # path to load checkpoint from, if None is ignored

    ## Paths - job ID is used for output dir name
    run_name: NotRequired[str]
    slurm_job_id: NotRequired[str | None]
    data_dir_path: str | Path
    temp_files_path: str | Path
    output_dir_path: str | Path
    output_dir: NotRequired[Path]
    figure_dir: NotRequired[Path]

    ## Distributed training setup
    distributed: NotRequired[bool]
    local_rank: NotRequired[int]
    local_world_size: NotRequired[int]
    total_world_size: NotRequired[int]
    dist_backend: NotRequired[str]
    dist_url: NotRequired[str]

    ## Torch settings
    device: NotRequired["torch.device"]
    pin_memory: NotRequired[bool]
    torch_version: NotRequired[str]
    cuda_version: NotRequired[str | None]
    accelerator_count: NotRequired[int]

    # this will be subscripted with a set of entries for every accelerator
    accelerator_i_name: NotRequired[str]
    accelerator_i_total_memory_GB: NotRequired[float]

    ## Extra configs included in the top level one
    norm_config: dict[str, dict[str, float]]
    net_config: AstNetConfigDict
    train_dataset_config: AstDatasetConfigDict
    test_dataset_config: AstDatasetConfigDict


### Default configs ###

## norms are by default the same across setups, taken from here
# manually experimented to set these
# key must be of the form `filename.method` found inside the `norms` directory (inherited from map2map)
norm_config: dict[str, dict[str, float]] = {
    "ember2_norms.dm_in_norm": {
        "eps": 1e-8,
        "exponent": +0.7,
        "log_mult": 1.0 / 2.5,
    },
    "ember2_norms.dm_norm": {
        "eps": 1e-8,
        "exponent": +0.7,
        "log_mult": 1.0 / 2.5,
    },
    "ember2_norms.E_norm": {
        "exponent": -3,
        "m_t_mult": +3.193,
        "m_t_add": 5.346,
    },
    "ember2_norms.gas_norm": {
        "exponent": +1.3,
        "log_mult": +1.0 / 1.3,
    },
    "ember2_norms.T_norm": {
        "exponent": -0.5,
        "m_t_mult": -0.093,
        "m_t_add": 5.346,
    },
}

## net config is shared
net_config: AstNetConfigDict = {
    # see above for comments about setup
    "style_dim": 64,
    "context_dim": 7,
    "style_depth": 5,
    "inp_channels": 1,
    "out_channels": 4,
    "filters": [16, 32, 64, 128],
    "num_noise_ch": 4,
    "beta_ema": 0.995,
    "lr_g": 1e-5,
    "lr_d": 1e-5,
    "betas_g": (0.5, 0.9),
    "betas_d": (0.5, 0.9),
    "use_noise": True,
    "use_adversarial": True,
}

# randomly generated list of redshifts to use in range 0,33:
# [1, 5, 6, 10, 20, 23, 24, 27]
train_dataset_config: AstDatasetConfigDict = {
    "dataset_name": "Training",
    "in_patterns": ["train_smudgy/DMO-*-*.npy"],
    "tgt_patterns": ["train_smudgy/DM-*-*.npy", "train_smudgy/E-*-*.npy", "train_smudgy/gas-*-*.npy", "train_smudgy/T-*-*.npy"],
    "style_pattern": "train_smudgy/style-*-*.npy",
    "in_filter_patterns": [r".*DMO-(?:\d{3,}|[^1]\d*|1[^0]+)-(0|1|5|10|20|27|33)\.npy"],
    # this pattern says: something-{}-{}.npy, where the first group can be any of
    #   * number with 3 or more digits (100, 101, 102, etc. allowed)
    #   * a character that is NOT 1, followed by 0 or more digits (2, 3, 4,... allowed but NOT 10)
    #   * a 1 followed by any character except 0, disallows 10 but allows 11, 12, 13,...
    # so basically: take out sim 1 and sim 10, use the others. Sim 10 is our test data.
    "tgt_filter_patterns": [
        r".*DM-(?:\d{3,}|[^1]\d*|1[^0]+)-(0|1|5|10|20|27|33)\.npy",
        r".*E-(?:\d{3,}|[^1]\d*|1[^0]+)-(0|1|5|10|20|27|33)\.npy",
        r".*gas-(?:\d{3,}|[^1]\d*|1[^0]+)-(0|1|5|10|20|27|33)\.npy",
        r".*T-(?:\d{3,}|[^1]\d*|1[^0]+)-(0|1|5|10|20|27|33)\.npy",
    ],
    "style_filter_pattern": r".*style-(?:\d{3,}|[^1]\d*|1[^0]+)-(0|1|5|10|20|27|33)\.npy",
    "in_norms": ["ember2_norms.dm_in_norm"],
    "tgt_norms": ["ember2_norms.dm_norm", "ember2_norms.E_norm", "ember2_norms.gas_norm", "ember2_norms.T_norm"],
    "callback_at": None,
    "augment": None,  # augmentation
    "aug_shift": None,
    "aug_add": None,
    "aug_mul": None,
    "crop": 128,
    "crop_start": None,
    "crop_stop": None,
    "crop_step": None,
    "in_pad": 2,
    "tgt_pad": 0,
    "scale_factor": 1,
    # data sampler
    "shuffle": True,
    "div_data": False,
    "div_shuffle_dist": 1,
}

test_dataset_config: AstDatasetConfigDict = {
    "dataset_name": "Test",
    # include some data both from outside train set, and from inside train set
    # to see generalization performance
    "in_patterns": ["train_smudgy/DMO-*-*.npy"],
    "tgt_patterns": ["train_smudgy/DM-*-*.npy", "train_smudgy/E-*-*.npy", "train_smudgy/gas-*-*.npy", "train_smudgy/T-*-*.npy"],
    "style_pattern": "train_smudgy/style-*-*.npy",
    # intended to mainly target *-10-2, *-199-1, *-145-24
    "in_filter_patterns": [r".*DMO-(10|199|145)-(0|1|2|14|24|33)\.npy"],
    "tgt_filter_patterns": [
        r".*DM-(10|199|145)-(0|1|2|14|24|33)\.npy",
        r".*E-(10|199|145)-(0|1|2|14|24|33)\.npy",
        r".*gas-(10|199|145)-(0|1|2|14|24|33)\.npy",
        r".*T-(10|199|145)-(0|1|2|14|24|33)\.npy",
    ],
    "style_filter_pattern": r".*style-(10|199|145)-(0|1|2|14|24|33)\.npy",
    "in_norms": ["ember2_norms.dm_in_norm"],
    "tgt_norms": ["ember2_norms.dm_norm", "ember2_norms.E_norm", "ember2_norms.gas_norm", "ember2_norms.T_norm"],
    "callback_at": None,
    "augment": None,
    "aug_shift": None,
    "aug_add": None,
    "aug_mul": None,
    "crop": 128,
    "crop_start": 64,
    "crop_stop": 192,
    "crop_step": None,
    "in_pad": 2,
    "tgt_pad": 0,
    "scale_factor": 1,
    # data sampler
    "shuffle": True,
    "div_data": False,
    "div_shuffle_dist": 1,
    # Power spectrum calculation
    "BoxSize": 25.0,  # h^-1 Mpc
    "MAS": "None",  # deliberately set to None due to smudgy mass assignment,
    # which ends up different from classic schemes like CIC or TSC
}

## main config setups
remote_node_config: AstConfigDict = {
    "config_name": "remote_node_config",
    "batch_size": 1,  # NOTE: batch size is broken, map2map problem. Just use 1.
    "shuffle_dataloader": False,
    "num_epochs": 160,
    "batch_break_point": None,
    "max_usable_threads_per_gpu_cap": 64,
    "dataloader_num_workers": 8,
    "checkpoint_freq": 10,
    # load from checkpoint
    "load_state": True,
    ## paths
    "data_dir_path": "/fp/projects01/ec12/ec-ericlu/CS5960AST_data/",
    # might need method to copy data from somewhere else to a scratch directory when script starts
    "temp_files_path": "/fp/projects01/ec12/ec-ericlu/CS5960AST/",
    "output_dir_path": "/fp/projects01/ec12/ec-ericlu/CS5960AST/models/runs/node/",
    ## extra configs
    "norm_config": norm_config,
    "net_config": net_config,
    "train_dataset_config": train_dataset_config,
    "test_dataset_config": test_dataset_config,
}

workstation_config: AstConfigDict = {
    "config_name": "workstation_config",
    "batch_size": 1,  # NOTE: batch size is broken, map2map problem. Just use 1.
    "shuffle_dataloader": False,
    "num_epochs": 50,
    "batch_break_point": None,
    "max_usable_threads_per_gpu_cap": 32,
    "dataloader_num_workers": 4,
    "checkpoint_freq": 10,
    # load from checkpoint
    "load_state": True,
    ## paths
    "data_dir_path": "/mn/stornext/d13/euclid_nobackup/dennisfr/master-project/data/",
    "temp_files_path": "/mn/stornext/d5/data/ericlu/CS5960AST/",
    "output_dir_path": "/mn/stornext/d5/data/ericlu/CS5960AST/models/runs/workstation/",
    ## extra configs
    "norm_config": norm_config,
    "net_config": net_config,
    "train_dataset_config": train_dataset_config,
    "test_dataset_config": test_dataset_config,
}

laptop_config: AstConfigDict = {
    "config_name": "laptop_config",
    "batch_size": 1,  # NOTE: batch size is broken, map2map problem. Just use 1.
    "shuffle_dataloader": False,
    "num_epochs": 10,
    "max_usable_threads_per_gpu_cap": 4,
    "dataloader_num_workers": 0,  # turns off dataloader method parallellization, much less memory used
    "batch_break_point": 2,
    "checkpoint_freq": 5,
    # load from checkpoint
    "load_state": True,
    ## paths
    "data_dir_path": "~/Documents/Work/CS5960AST/CS5960AST_repo/data/",
    "temp_files_path": "~/Documents/Work/CS5960AST/CS5960AST_repo/",
    "output_dir_path": "~/Documents/Work/CS5960AST/CS5960AST_repo/models/runs/laptop/",
    ## extra configs
    "norm_config": norm_config,
    "net_config": net_config,
    "train_dataset_config": train_dataset_config,
    "test_dataset_config": test_dataset_config,
}
