# Master Project: Fast Baryon Field Generation from Dark Matter Seeds via a 3D Deep Learning Model
This project consists of a machine learning model capable of producing full-fledged 3D baryon fields corresponding to a hydrodynamical simulation based on a dark matter only input which can be produced from a simple N-body simulation.

Source data is taken from the CAMELS project: <https://www.camel-simulations.org/>

Code from map2map used as baseline for loading and managing data, available at: <https://github.com/sagasv5-xw/map2map/tree/styled_srsgan>

EMBER-2 used for baseline machine learning model, adapted from: <https://github.com/maurbe/ember-2/>, with webpage at <https://maurbe.github.io/ember-2/>

## Usage
Easiest, use package manager [`uv`](https://docs.astral.sh/uv/), and simply `uv sync` after cloning the repository.
This will install the project as an editable package along with all requirements in a new virtual environment.

Recommendation: on MacOS, use `brew install uv` for `uv` and not `curl` as written in link, it is easier to keep updated with a central package manager.

Requirements are in [`pyproject.toml`](pyproject.toml). Script entry point can be found in [`entry_point.py`](src/cs_5960_ast/entry_point.py).

Entry point has been configured for `uv` such that project can be ran with `uv run cs_5960_ast {train, infer}`.
Without `uv`, run the project as a python module, see below.

### Setup
`uv sync` will perform all setup.
Otherwise, the [`pyproject.toml`](pyproject.toml) file is the canonical source of dependencies, and should be directly usable for setup.
Create a virtual environment `.venv` for the project according to package manager instructions, and install the project package.
[Editable mode](https://setuptools.pypa.io/en/latest/userguide/development_mode.html) is recommended.

A [`requirements.txt`](requirements.txt) file is also provided, either dependency file can be used for setup with any standard python package manager.
This file is generated with `uv export --format requirements.txt --no-hashes >> requirements.txt`.

Setup example, [for `pip`](https://packaging.python.org/en/latest/guides/installing-using-pip-and-virtual-environments/):

```console
# (from the repository folder, wherever the project was cloned)
python3.13 -m venv .venv
source .venv/bin/activate
pip install --upgrade pip
pip install -e .
```

Note: use `pip install -e '.[dev]'` to also get development tools like linter and type checker.
`ruff` and `ty` should be fully configured for the project via the `pyproject.toml`.

Note: Project needs at least python 3.13 installed.
Make sure `pip` is being used from the `.venv`, check `which pip`.

### Run as module
After setup, trigger the python module with command:

```console
uv run -m cs_5960_ast {train, infer}
OR
python -m cs_5960_ast {train, infer}
```

The first command runs the cs_5960_ast python module with all dependencies handled by `uv`.
Will trigger `__main__.py` inside module source code, which leads to `entry_point.py`.

The second command is plain python but needs an activated virtual environment with dependencies installed, including the project module.

Either case must set mode argument according to use.

Other possible options for running which are not meaningfully different:

```console
uv run cs_5960_ast {train, infer} # (does not trigger __main__.py, proceeds directly to entry_point.py)
uv run python -m cs_5960_ast {train, infer} # (same as uv run -m)
```

### Arguments
The project must be ran in either training or inference mode, with the command line arguments `[...] cs_5960_ast train` or `[...] cs_5960_ast infer`, respectively.
Use `[...] cs_5960_ast -h` for a list of arguments + explanations.

#### Unattended runs
Multiple bash scripts are provided for running the project in various ways without needing a terminal connected. Very useful for overnight runs.
However, these scripts expect `uv` to be installed to simplify package management.
The options are:

- Training:
    - [`run_training.sh`](scripts/run_training.sh): Simple, non-distributed unattended training run intended for workstation/desktop pc.
    - [`run_training_distributed.sh`](scripts/run_training_distributed.sh): Distributed variant of the above script that uses `torchrun` for multi-GPU training.
    - [`distributed_training.slurm`](scripts/distributed_training.slurm): The "proper" model training script, intended for use with Slurm on a compute cluster. Set up for use with the Fox supercomputer. Note the `module load` to access `uv`.

- Inference:
    - [`run_model_inference.sh`](scripts/run_model_inference.sh): Loads model weights from the specified checkpoint and performs simple inference. Intended for laptops or similar for quick checking of model results. Does NOT run in the background and will output to terminal.
    - [`run_model_inference_workstation.sh`](scripts/run_model_inference_workstation.sh): Desktop version of above script. Also takes over console. Will use one GPU for model predictions, but is still fairly slow due to generating large numbers of plots without threading. Not recommended.
    - [`run_model_inference_workstation_distributed.sh`](scripts/run_model_inference_workstation_distributed.sh): Fully unattended and distributed inference script.
    Runs field generation on multiple GPUs and further multithreads saving of figures, giving much faster results for both more and larger fields.
    "Best" version to generate results with. Should still work with just one GPU, but it needs more than 8GB of VRAM.

Example of use for an overnight/long session on a 24/7 remote workstation or compute node with ssh:

```console
ssh user@remote_host # remember to set up .ssh/config with a host entry
cd /wherever/you/cloned/this/code/previously
./run_model_inference_workstation_distributed.sh
exit # check back in later
```

Check inside the scripts for details on what commands and environment variables they use.
Note the checkpoint path argument in inference scripts, this must be set to the model snapshot to be used.

### Training with Slurm on a compute node
Run `sbatch distributed_training.slurm` to queue a run via the workload manager.
Check the script status in queue with `squeue --me`, and use `squeue partition=accel` to see all current GPU jobs.
See the [`distributed_training.slurm`](distributed_training.slurm) script for configuration settings used.
By default, will request a single node with 4 GPUs to train on, and spawn one primary process per GPU with several threads - these are used for loading of data and power spectrum calculation.

## Data Files
Data is stored in the University of Oslo [Institute of Theoretical Astrophysics \(ITA\) Stornext storage server](https://www-int.mn.uio.no/astro/english/services/it/help/basic-services/computers/storage.html) at:

```console
/mn/stornext/d13/euclid_nobackup/dennisfr/master-project/data/
```

The field data we use is arranged with the file naming scheme: `{field type}-{simulation tag}-{redshift identifier}.npy`

The field type indicates which field is in the file, such as `E` for energy or `T` for temperature.
`DMO` are dark matter only fields and used for input, while `DM` are the dark matter component of fully simulated 4-channel fields.

The simulation tag refers to the simulation ID in the CAMELS LHS, e.g. files with different simulation tags refer to different simulations run with a different seed and cosmic parameters.
The parameters used are reflected in the context tensor with the same tag.

## Configuration
See the file [`default_configs.py`](src/cs_5960_ast/default_configs.py) for configuration needed for the project.
Notably, project runs in laptop mode by default, with config at the very bottom - a run will probably fail due to data path not pointing to the correct location.
Change this and any other settings needed before running.
If on the ITA server, use workstation mode (one of the provided scripts is recommended) and the model should run.
