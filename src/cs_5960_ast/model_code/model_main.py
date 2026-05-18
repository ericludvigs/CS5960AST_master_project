"""Primary model setup and training"""

import concurrent.futures
import logging  # noqa: TC003
import os
import re
import time
from datetime import timedelta
from pathlib import Path
from types import FunctionType
from typing import Any, Literal, cast

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.distributed as dist
import torch.multiprocessing as mp  # noqa: F401
from torch.nn.parallel import DistributedDataParallel
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from tqdm.auto import tqdm

import cs_5960_ast.map2map.src.utils.state as state_utils
from cs_5960_ast import AstConfigDict, AstDatasetConfigDict, AstNetConfigDict
from cs_5960_ast.map2map.src.data import DistFieldSampler, FieldDataOutputItem, FieldDataset
from cs_5960_ast.map2map.src.data.norms import ember2_norms
from cs_5960_ast.methods import narrow, utils
from cs_5960_ast.methods.Pk import Pk
from cs_5960_ast.model_code.model_ember2_code import Ember2NetWrapper


class ModelWrangler:
    """Wrapper class for handling ML model, training, etc.

    Init will always instatiate a working model. Run `train_model` to train.
    If distributed mode is on, will set up a DDP container. Expects a default process group to have been created already.
    With checkpoints available, model will be loaded to latest checkpoint state.

    Parameters
    ----------
    config : dict
        A configuration dictionary with key/values to use defined in `cs_5960_ast.AstConfigDict`.
    mode : one of "train" or "infer"
        Whether a model is being set up for training or inference.

    Attributes
    ----------
    ember2_instance : Ember2NetWrapper
        Instance of a class implementing a customized EMBER-2 machine learning model suitable for 3D field emulation.
        It is the model that this class exists to handle, performing checkpoint saving/loading, distributed set up, etc.
        Technically can accept other models with the same modules/attributes/general interface.
        The model is based on `nn.Module`.
    ember2_DDP : DistributedDataParallel
        If distributed mode is on, this class will set up a distributed ML model across multiple GPUs
        via the pytorch DistributedDataParallel container formalism. This attribute saves the relevant container.
    avg_mse_per_epoch : list[float]
        A list the length of the number of epochs the model was trained for, containing the average MSE against the target
        for each epoch as a single python float.
    train_dataset : FieldDataset
        Dataset used for training.
    train_sampler : DistFieldSampler
        The sampler attached to train set.
    test_dataset : FieldDataset
        Dataset used for test.
    test_sampler : DistFieldSampler
        Sampler attached to test set.
    NOTE: validation set not implemented
    current_epoch : int
        The latest epoch that has been trained for the currently loaded model.
        Updated dynamically during training. Set when loading a model checkpoint, for inference too.
    step : int
        Step is number of trained data batches * epochs,
        so specifically how many times the gradients have been updated and backprop run, etc.
        NOTE: if model had training resumed with different amounts of data available,
        the number of steps will be inconsistent between epochs.
    config : AstConfigDict
        A dictionary config object, containing all the necessary parameters for the program. See `default_configs.py`.
        There are a bunch of sub-configs extracted from this one and also saved. Just used to logically group settings.
    mode : "infer" or "train"
        Whether the model is to be used for field inference, or training.
    model_trained : bool
        Has the model been trained to completion (according to config)?
    ckpt_freq : int
        How often the model should make checkpoints during training.
    logger : logging.Logger
        An instanced logger object for outputting information to.
        The purpose for a logger (rather than just print()) is to save run logs to file in their associated model directory.
    tensorboard_writer : torch.utils.tensorboard.SummaryWriter
        An instanced tensorboard writer, used to log model metrics for visualization in tensorboard.
    local_rank : int
        The process rank for multiprocessing. Always 0 on non-distributed runs.
        Most logging and metrics writing is only performed on process 0 to avoid file conflicts.
    torch_device : torch.device
        The device that this process is running on - on distributed multi-gpu setups this tells you
        which specific GPU is active for this process. Also tells you which accelerator is being used (if any).
    """

    def __init__(self, config: AstConfigDict, mode: Literal["train", "infer"]) -> None:
        torch.backends.cudnn.benchmark = True
        # https://stackoverflow.com/questions/58961768/set-torch-backends-cudnn-benchmark-true-or-not
        # our input should be constant

        # on GPU node, address warning:
        # UserWarning: TensorFloat32 tensor cores for float32 matrix multiplication available but not enabled.
        if config["device"].type == "cuda" and torch.cuda.get_device_capability(config["device"]) >= (7, 0):
            torch.set_float32_matmul_precision("high")

        # it's fine to open as many figures as we have threads available
        # but also respect what the setting normally is
        default_matplotlib_figure_max = 20
        if config["max_usable_threads_per_gpu"] >= default_matplotlib_figure_max:
            plt.rcParams.update({"figure.max_open_warning": config["max_usable_threads_per_gpu"] + 1})

        # type hinting the attributes that should not be filled in yet
        self.ember2_instance: Ember2NetWrapper
        self.ember2_DDP: DistributedDataParallel[Ember2NetWrapper]
        self.avg_mse_per_epoch: list[float] = []
        self.train_dataset: FieldDataset
        self.train_sampler: DistFieldSampler
        self.train_loader: DataLoader[FieldDataOutputItem]
        self.test_dataset: FieldDataset
        self.test_sampler: DistFieldSampler
        self.test_loader: DataLoader[FieldDataOutputItem]

        # definitions
        self.current_epoch: int = 1  # gets overwritten if a checkpoint load happens
        self.step: int = 1  # same
        self.config: AstConfigDict = config
        self.mode = mode
        self.model_trained: bool = False
        self.net_config: AstNetConfigDict = config["net_config"]
        self.norm_config: dict = config["norm_config"]
        self.train_dataset_config: AstDatasetConfigDict = config["train_dataset_config"]
        self.test_dataset_config: AstDatasetConfigDict = config["test_dataset_config"]
        self.ckpt_freq = config["checkpoint_freq"]
        self.logger: logging.Logger = config["logger"]
        self.local_rank: int = int(os.getenv("LOCAL_RANK", "-1"))

        # self.check_normalization_trigger()
        torch_device = self.config["device"]
        self.torch_device = torch_device

        if self.local_rank == 0:
            output_dir = Path(os.getenv("OUTPUT_DIR", default=Path.cwd()))
            self.model_checkpoints_dir = output_dir / "model_checkpoints"  # NOTE: hardcoded
            self.model_checkpoints_dir.mkdir(exist_ok=True)
            (self.model_checkpoints_dir / ".placeholder").touch(
                exist_ok=True
            )  # make a placeholder file so this folder is included in version control
            # NOTE: by default ignoring the checkpoint files, they are too large for normal github version control
            # therefore, you have to (r)sync them manually. This is easier when you have the destination folder on all machines.
        else:
            # var should not end up used on other ranks, but will prevent type checking errors
            self.model_checkpoints_dir = Path.cwd()

        # make dataloaders
        if mode == "train":
            self.setup_train_dataloader()
        # always going to use the test dataloader, whether for inference or the end of a training run
        self.setup_test_dataloader()

        self.ember2_instance = Ember2NetWrapper(
            net_config=self.net_config,
            device=torch_device,
            figure_dir=config["figure_dir"],
            ckpt_freq=self.ckpt_freq,
            current_epoch=self.current_epoch,  # gets overwritten by checkpoint
            tensorboard_writer=None,  # set this later
        )
        self.ember2_instance = self.ember2_instance.to(torch_device)

        if config["distributed"] is True:
            self.ember2_DDP = DistributedDataParallel(
                module=self.ember2_instance, device_ids=[torch_device], output_device=torch_device, process_group=None
            )
            # torch_device should be set to the active process' designated GPU
            # process_group=None uses the default group created by init_process_group earlier
            self.ember2_instance: Ember2NetWrapper = self.ember2_DDP.module  # gets the module inside the DDP container
        # self.ember2_instance: Ember2NetWrapper = torch.compile(self.ember2_instance)

        # load checkpoint if one exists, after init
        self.load_checkpoint()

        # set up tensorboard, after checkpoint loading to use the purge feature
        # not used when re-running results to avoid making a million tensorboard entries
        if self.local_rank == 0 and self.mode != "infer":
            self.tensorboard_writer = SummaryWriter(
                log_dir=(output_dir / "tensorboard/").as_posix(),
                purge_step=self.current_epoch,  # loads up to current_epoch and removes data from epochs after that
                # filename_suffix=f".device_{config['device']}",  # not really necessary
            )
        else:
            self.tensorboard_writer = None
        # set in model
        self.ember2_instance.tensorboard_writer = self.tensorboard_writer

    def save_checkpoint(self) -> None:
        """Save a model checkpoint, for process 0 only

        Saves after ending epoch x, thus when model resumes it should not restart epoch x and train again,
        but proceed to epoch x+1. Hence saving state as current_epoch + 1. Technically this is consistent with training
        starting at epoch 1, it can be seen as model having "finished" the non-existent epoch 0 and proceeding from there.

        Also, consistent with map2map.

        Can be called from multiple processes, but will exit out and do nothing on any rank but 0.
        """
        # early return on processes other than main
        if self.local_rank != 0:
            return None

        if self.tensorboard_writer is not None:
            self.tensorboard_writer.flush()

        state = {
            "epoch": self.current_epoch + 1,
            "step": self.step,
            "model": self.ember2_instance.state_dict(),
            "gen_optimizer": self.ember2_instance.opt_g.state_dict(),
            # "scheduler": scheduler.state_dict(),
            "adv_optimizer": self.ember2_instance.opt_d.state_dict(),
            # "adv_scheduler": adv_scheduler.state_dict(),
            "rng": torch.get_rng_state(),
            "avg_mse_per_epoch": self.avg_mse_per_epoch,
        }
        # pytorch might automatically update containered optimizer but documentation is not 100% clear
        # so manually save and update optimizer parameters just in case

        state_filename = f"model_state_{self.current_epoch + 1}.pt"
        state_path = self.model_checkpoints_dir / state_filename
        torch.save(state, state_path)

        self.logger.debug(f"Checkpoint file written for epoch {self.current_epoch} at {state_path}")

        # cleanup
        del state
        return None

    def load_checkpoint(self) -> None:
        """Load and use a previous model checkpoint on the current process

        Searches for the checkpoint file with the highest saved epoch

        Checkpoint will be loaded to current accelerator device

        map2map sets up model, puts it on device, sets up DDP, then loads checkpoint

        Intended to be called by all processes

        Will set `self.current_epoch` and `self.step` according to how far the model got
        """
        # early exit from loading if checkpoint load is disabled
        if not self.config["load_state"]:
            if self.mode == "infer":
                msg = "Model set to inference mode with checkpoint loading disabled, so no model can be loaded or used"
                raise ValueError(msg)
            if self.local_rank == 0:
                self.logger.debug(f"Checkpoint loading disabled, starting training from epoch {self.current_epoch}")
            return None

        # config path if not None is a working path, it was checked in entry_point
        # but it can either point directly at a file or at a directory
        if (load_path := self.config["checkpoint_load_path"]) is not None and not load_path.is_file():
            ckpt_files = load_path.glob("*.pt")

            current_epoch = 0
            latest_ckpt_file = None

        elif load_path is not None and load_path.is_file():
            # this path will skip loop over files
            ckpt_files = []
            latest_ckpt_file = load_path

        elif load_path is None:
            ckpt_files = self.model_checkpoints_dir.glob("*.pt")
            # find .pt files with a number at the end, capture what that number is
            pattern = re.compile(r".*_(\d+)\.pt")

            current_epoch = 0
            latest_ckpt_file = None

        # if a file was directly specified above, this is an empty list, and skips loop
        for file in ckpt_files:
            matched = pattern.search(file.name)
            if matched is not None:
                found_epoch = int(matched.group(1))
                if found_epoch > current_epoch:
                    current_epoch = found_epoch
                    latest_ckpt_file = file

        if latest_ckpt_file is not None and latest_ckpt_file.is_file():
            if self.local_rank == 0:
                self.logger.debug(f"Found checkpoint file: {latest_ckpt_file}")
            # this will continue by not entering else block
        else:
            if self.mode == "infer":
                msg = (
                    f"Model set to inference mode but no checkpoint could be found at `{self.model_checkpoints_dir}`. Check job ID."
                )
                raise ValueError(msg)
            if self.local_rank == 0:
                self.logger.debug(f"No checkpoint file found, training starting from epoch {self.current_epoch}")
            # exit loading if no checkpoints
            return None

        # load the statefile
        state: dict[str, Any] = torch.load(latest_ckpt_file, map_location=self.torch_device, weights_only=False)

        statefile_epoch = state["epoch"]

        state_utils.load_model_state_dict(
            module=self.ember2_instance,
            state_dict=state["model"],
            strict=self.config["load_state_strict"],
        )

        if "optimizer" in state:
            self.ember2_instance.opt_g.load_state_dict(state["gen_optimizer"])
        # if "scheduler" in state:
        #    self.ember2_instance.scheduler.load_state_dict(state["scheduler"])
        if "adv_optimizer" in state:
            self.ember2_instance.opt_d.load_state_dict(state["adv_optimizer"])
        # if "adv_scheduler" in state:
        #    self.ember2_instance.adv_scheduler.load_state_dict(state["adv_scheduler"])

        torch.set_rng_state(state["rng"].cpu())  # move rng state back

        self.avg_mse_per_epoch = state.get("avg_mse_per_epoch", [])  # if not in state, keep empty

        # set the steps and epochs in both classes
        self.current_epoch: int = statefile_epoch
        self.step: int = state["step"]

        self.ember2_instance.current_epoch: int = statefile_epoch
        self.ember2_instance.step: int = state["step"]

        # model found and can be used for inference, let's just call it "trained"
        if self.mode == "infer":
            self.model_trained = True

        if self.local_rank == 0:
            self.logger.info(f"Model state at epoch {state['epoch']}, step {state['step']} loaded from file.")

        # cleanup
        del state
        return None

    def train_model(self, epochs: int) -> None:
        """Train model, handles epochs

        Model itself is expected to handle running through data batches for each epoch

        train model -> all (specified) epochs
        train epoch -> iterate through all training data batches
        train step -> process one data batch

        Checkpoint and model saving is handled in this class, outside of model

        Parameters
        ----------
        epochs : int
            Number of epochs to train for.
        """
        model: Ember2NetWrapper = self.ember2_instance

        if self.current_epoch >= epochs:
            msg = (
                f"Epoch loaded from checkpoint ({self.current_epoch}) is already greater than specified epochs to train for, "
                f"epochs={epochs}."
            )
            self.logger.error(msg)
            raise ValueError(msg)

        # activate training mode
        model.train(mode=True)

        # set up training loop
        error_ten_epochs_ago: float | None = None  # model "forgets" this if loading from checkpoint, not worth fixing

        # TODO: see if tqdm can be written to its own standalone file and not be a mess
        # set mininterval larger to avoid constant I/O
        for epoch_num in tqdm(  # progress bar for epoch
            range(self.current_epoch, epochs + 1),
            # from checkpoint epoch (default 1 with no checkpoint)
            # to epochs+1 because range only runs to n-1
            desc="Epochs",
            position=0,  # this bar is set to top level
            # the model class has already fetched tqdm settings when initialized, just use those
            disable=model.should_disable_tqdm,
            mininterval=model.tqdm_mininterval,
        ):
            self.current_epoch: int = epoch_num
            model.train_epoch(
                train_dataloader=self.train_loader,
                train_sampler=self.train_sampler,
                epoch_num=epoch_num,
                break_point=self.config["batch_break_point"],
            )
            self.step: int = model.step
            # NOTE: epochs are 1-indexed, list is 0-indexed
            self.avg_mse_per_epoch.append(np.mean(model.mse_error_during_epoch))

            # after model has run through data and applied training, every X epochs run a checkpoint
            ckpt_state = epoch_num % self.ckpt_freq == 0
            if epoch_num == epochs:  # always save at end of training, even if not on checkpoint epoch
                ckpt_state = True
            # ensure processing of epochs up until checkpoint is finished on all ranks
            if ckpt_state and self.config["distributed"] is True:
                dist.barrier()
            if ckpt_state and self.local_rank == 0:  # only save model on main process
                self.logger.info(f"Checkpoint at epoch: {epoch_num}")
                self.save_checkpoint()

                # relative difference in error since last checkpoint
                if error_ten_epochs_ago is not None:
                    error_diff: float = (self.avg_mse_per_epoch[epoch_num - 1] - error_ten_epochs_ago) / error_ten_epochs_ago
                    self.logger.info(f"MSE change last {self.ckpt_freq} epochs: {error_diff:+.4%}")
                # update with the current epoch error
                error_ten_epochs_ago = self.avg_mse_per_epoch[epoch_num - 1]
                # it is called "ten epochs ago" because of the default setting - technically if you change ckpt_freq,
                # the error is of ckpt_freq epochs ago

        # only on process 0, iff user defined a tensorboard setup
        # after the last epoch, after running through all the data,
        # log the state of the networks
        if self.local_rank == 0 and self.tensorboard_writer is not None:
            # if self.step % 50000 == 0:
            # original EMBER-2 code is every 50 000 steps, not sure if that makes multiple snapshots
            # or is tuned to only trigger at the end - went with just one snapshot for now
            model.log_network(model.G, "G", self.tensorboard_writer)
            model.log_network(model.GE, "GE", self.tensorboard_writer)
            model.log_network(model.D, "D", self.tensorboard_writer)

            # log shape of the network, using an example set of data the model saved
            # NOTE: .add_graph() causes CONSTANT problems, dropped for now
            # self.tensorboard_writer.add_graph(model, model.example_input, use_strict_trace=False)
            # del model.example_input  # clean up

            self.tensorboard_writer.add_hparams(
                run_name=".",
                hparam_dict={
                    "epochs": epochs,
                    "lr_g": self.net_config["lr_g"],
                    "betas_g_1": self.net_config["betas_g"][0],
                    "betas_g_2": self.net_config["betas_g"][1],
                    "lr_d": self.net_config["lr_d"],
                    "betas_d_1": self.net_config["betas_d"][0],
                    "betas_d_2": self.net_config["betas_d"][1],
                    "beta_ema": self.net_config["beta_ema"],
                    "style_dim": self.net_config["style_dim"],
                    "context_dim": self.net_config["context_dim"],
                    "style_depth": self.net_config["style_depth"],
                    "num_filters": len(self.net_config["filters"]),
                    "use_noise": self.net_config["use_noise"],
                },
                metric_dict={"hparam/final_mse": self.avg_mse_per_epoch[-1]},
            )
            # "Call this method to make sure that all pending events have been written to disk"
            self.tensorboard_writer.flush()

        # make the error plot since training is done
        fig, ax = plt.subplots()
        ax.plot(self.avg_mse_per_epoch)
        ax.set_title("Average error against target during training per epoch")
        ax.set_xlabel("Epoch")
        ax.set_ylabel("MSE")
        fig.savefig(self.config["figure_dir"] / "training_error.png")
        if self.tensorboard_writer is not None:
            self.tensorboard_writer.add_figure("Figures/training_error", fig, close=True, global_step=self.step)

        # training done, clean up
        model.train(mode=False)
        self.model_trained = True

    def test_model_run(self) -> None:
        """Run the model on test data and output comparison/result plots, also calculate power spectrums and compare"""
        if self.model_trained is False:
            msg = "Must train model before the run can be verified"
            raise AttributeError(msg)

        # standardize figure size
        self.figwidth = 6.5
        self.figheight = 5.8
        # multiply by number of columns and rows respectively

        self.colormap_to_use = "viridis"
        # bone, viridis?
        # map2map uses:
        # `inferno` if all values positive
        # `inferno_r` if all values negative
        # `RdBu_r` else

        # coordinates to insert a colorbar on the right of the figure
        self.axes_inset = (1.02, 0, 0.06, 1.0)

        # kwargs for tight layout
        self.tight_layout_pad = 1.50
        self.tight_layout_rect = (0, 0.08, 1, 1)

        self.subvolume_first_coord = 3

        self.input_color = "tab:green"
        self.target_color = "tab:blue"
        self.output_color = "tab:orange"
        self.max_k_color = "tab:purple"

        # CAMELS smallest simulation scale is 0.09765625 [h^-1 Mpc], about 100 kpc/h
        self.smallest_length_scale = 0.09765625
        self.max_resolution_k = utils.length_to_k(self.smallest_length_scale)
        # any data beyond this wave number is below the resolution of CAMELS and thus questionable
        self.largest_length_scale = 25.0  # max size of a box
        self.min_resolution_k = utils.length_to_k(self.largest_length_scale)

        # write out the number of parameters in the model
        if self.local_rank == 0:
            self.ember2_instance.num_parameters_summary()

        # TODO: might be able to change to handle multiple batches properly
        # fetch a single batch from the test dataloader
        # which would be the first one, which is random if shuffle is on
        # however, what you should actually do is use a training field where there is only one sample fetched
        # so that the entire field gets passed in to testing
        self.test_sampler.set_epoch(0)
        if self.local_rank == 0:
            self.logger.debug(f"Test loader batch count: {len(self.test_loader)}")

        if len(self.test_loader) == 0:
            msg = "Test dataloader has no data in it"
            raise ValueError(msg)

        if len(self.test_loader) > 1:
            self.logger.warning(f"""
            Warning: test dataloader has more than one entry ({len(self.test_loader)});
            this works with DIFFERENT datasets and only one crop per field;
            but is NOT supported with multiple crops per same field.

            Multiple crops per field will result in overwriting each others plots, and only the last will be kept.

            Field results from different simulations will be properly saved.

            For the test set, use `crop=None` to pass an entire field at once.

            Process: {self.local_rank}
            """)
        # TODO: fix it so fields can be stitched together from crops

        for data in self.test_loader:
            # ty doesn't properly use user annotations: https://github.com/astral-sh/ty/issues/136
            data = cast("FieldDataOutputItem", data)

            # unpack data
            input_vector: torch.Tensor = data["input"].to(self.config["device"])
            input_style: torch.Tensor = data["style"].to(self.config["device"])
            target_vector: torch.Tensor = data["target"]  # not assigned to gpu on purpose
            input_relpath: list[tuple[str]] = data["input_relpath"]  # ty:ignore[invalid-assignment]
            target_relpath: list[tuple[str]] = data["target_relpath"]  # ty:ignore[invalid-assignment]
            # relpath items are returned as tuples due to dataloader and collate_fn() (making ty incorrect here)
            # but for test data batch size should be 1 anyway so, unwrap:
            input_relpath: list[str] = [item[0] for item in input_relpath]
            target_relpath: list[str] = [item[0] for item in target_relpath]

            in_pad = self.config["test_dataset_config"]["in_pad"]
            tgt_pad = self.config["test_dataset_config"]["tgt_pad"]

            # this has to come after data to get the simulation name
            # filename is of form tag-x-z.npy
            simulation_tags = Path(input_relpath[0]).stem.split("-")
            # we want the sim number x and redshift indicator (and ignore tag)
            simulation_name = f"sim_{simulation_tags[1]}_{simulation_tags[2]}_epoch_{self.current_epoch}"

            # get the actual redshift from the style vector
            # style vector with dataloader is size([1,7])
            scale_factor = input_style[0, 0].item()
            z = 1 / scale_factor - 1

            # save the plots to a specific folder for which this field belongs
            figure_dir = self.config["figure_dir"] / simulation_name
            figure_dir.mkdir(exist_ok=True)

            # standardize the input parameters to write out the same text on the bottom right of the figure
            # via `fig.text`, tells you which simulation this figure is from and should be on all the figures
            figtext_kwargs = {
                "x": 0.835,
                "y": 0.060,
                "s": f"Epoch: {self.current_epoch}, z={z:.3f}\nSimulation: {simulation_name}",
                "ha": "left",
                "va": "top",
            }
            # note the use of simulation_name, these settings are per set of data

            # NOTE: matplotlib and colorbars is "fun"
            # see this answer for a bunch of ways to do things and caveats: https://stackoverflow.com/a/79031377

            # logging
            if self.local_rank == 0:
                self.logger.debug(f"Running model on test data from simulation: {simulation_name}")
                self.logger.debug(f"Input files: {input_relpath}, Target files: {target_relpath}")
                self.logger.debug(f"Input vector shape: {input_vector.shape}, with {in_pad}+{in_pad} padding")
                self.logger.debug(f"Target vector shape: {target_vector.shape}, with {tgt_pad}+{tgt_pad} padding")

            # prediction
            output_vector: torch.Tensor = self.ember2_instance.predict_ember2(input_vector, input_style, pad_size=in_pad)
            # TODO: test ember2_GE
            input_vector: torch.Tensor = narrow.narrow_like(input_vector, target_vector)  # remove padding from input

            # detach for plotting
            output_vector: torch.Tensor = output_vector.detach().cpu()
            input_vector: torch.Tensor = input_vector.detach().cpu()

            if self.local_rank == 0:
                self.logger.debug(f"Output vector shape: {output_vector.shape}, with {tgt_pad}+{tgt_pad} padding")

            # extract the correct tensors and associate with names
            input_labels_dict: dict[str, torch.Tensor] = {
                "Pure DM input": input_vector[0, 0, :, :, :],
            }
            target_labels_dict: dict[str, torch.Tensor] = {
                "Mixed DM target": target_vector[0, 0, :, :, :],
                "Internal energy target": target_vector[0, 1, :, :, :],
                "Gas density target": target_vector[0, 2, :, :, :],
                "Temperature T target": target_vector[0, 3, :, :, :],
            }
            output_labels_dict: dict[str, torch.Tensor] = {
                "Mixed DM output": output_vector[0, 0, :, :, :],
                "Internal energy output": output_vector[0, 1, :, :, :],
                "Gas density output": output_vector[0, 2, :, :, :],
                "Temperature T output": output_vector[0, 3, :, :, :],
            }
            power_spectrum_dict: dict[str, Pk] = {}  # filled in with power spectrums for each field type

            # colormaps to use per field
            self.field_colormaps = {
                "Pure DM input": "viridis",
                "Mixed DM target": "viridis",
                "Internal energy target": "plasma",
                "Gas density target": "bone",
                "Temperature T target": "RdBu_r",
                "Mixed DM output": "viridis",
                "Internal energy output": "plasma",
                "Gas density output": "bone",
                "Temperature T output": "RdBu_r",
            }

            # for later use, calculate the max/min values of the tensors
            # this is done only once in a dict to avoid recomputing
            input_minmax_dict: dict[str, float] = {}
            target_minmax_dict: dict[str, float] = {}
            output_minmax_dict: dict[str, float] = {}
            for key, tensor in input_labels_dict.items():
                input_minmax_dict[key + " min"] = torch.min(tensor).item()
                input_minmax_dict[key + " max"] = torch.max(tensor).item()
            for key, tensor in target_labels_dict.items():
                target_minmax_dict[key + " min"] = torch.min(tensor).item()
                target_minmax_dict[key + " max"] = torch.max(tensor).item()
            for key, tensor in output_labels_dict.items():
                output_minmax_dict[key + " min"] = torch.min(tensor).item()
                output_minmax_dict[key + " max"] = torch.max(tensor).item()

            ## Plot 2D slices of fields

            # DM plots
            fig, ax = plt.subplots(ncols=3, nrows=1, figsize=(self.figwidth * 3, self.figheight * 1.1))
            imgs_container: list[plt.AxesImage] = []
            for i, (key, container_dict) in enumerate(
                zip(
                    ["Pure DM input", "Mixed DM target", "Mixed DM output"],
                    [input_labels_dict, target_labels_dict, output_labels_dict],
                    strict=True,
                )
            ):
                colormap_to_use = self.field_colormaps[key]
                imgs_container.append(
                    ax[i].imshow(
                        container_dict[key][self.subvolume_first_coord, :, :],
                        cmap=colormap_to_use,
                        # vmin=input_minmax_dict["Pure DM input" + " min"],
                        # vmax=input_minmax_dict["Pure DM input" + " max"],
                    )
                )
                ax[i].set_title(key)
                cax = ax[i].inset_axes(self.axes_inset)
                fig.colorbar(imgs_container[i], cax=cax, orientation="vertical", extend="neither")
            # savefig
            fig.suptitle(
                f"2D slices of Dark Matter fields\nSlice at x={self.test_dataset_config['crop_start'] + self.subvolume_first_coord}"  # ty:ignore[unsupported-operator]
            )
            # TODO: tuples
            fig.text(**figtext_kwargs)  # ty:ignore[invalid-argument-type] could fix with typeddict but who has time for that
            fig.tight_layout(pad=self.tight_layout_pad, rect=self.tight_layout_rect)
            # NOTE: tight layout adjusted to give space for the infotext in the bottom right
            fig.savefig(figure_dir / "DM_comparison_slice.png")
            if self.tensorboard_writer is not None:
                self.tensorboard_writer.add_figure("Figures/DM_comparison_slice", fig, close=True, global_step=self.step)

            # Plots of all target channels vs output channels, cycles through first coord for slicing
            # set to use multiple processes to independently save the slices
            # simple loop for non-distributed run, otherwise set up processes
            field_slice_indexes = range(
                0,
                self.test_dataset_config["crop"],  # ty:ignore[invalid-argument-type]
                self.test_dataset_config["crop_step"] if self.test_dataset_config["crop_step"] is not None else 1,  # ty:ignore[invalid-argument-type]
            )  # TODO: why the tuples, anyway?
            comparison_slice_folder_name = "field_comparison_slices/"
            (figure_dir / comparison_slice_folder_name).mkdir(exist_ok=True)

            if self.config["distributed"] is False:
                for slice_coord in field_slice_indexes:
                    self.make_2d_field_comparison_slice(
                        slice_coord,
                        target_labels_dict,
                        output_labels_dict,
                        target_minmax_dict,
                        output_minmax_dict,
                        figure_dir,
                        figtext_kwargs,
                        comparison_slice_folder_name,
                    )

            elif self.config["distributed"] is True:
                # capped at workers per gpu since we do inference (predicting the field) per GPU, and distribute the work
                with concurrent.futures.ThreadPoolExecutor(max_workers=self.config["max_usable_threads_per_gpu"]) as executor:
                    # function just outputs None, but we want it to be queued and executed for all slices
                    future_2d_slices = [
                        executor.submit(
                            self.make_2d_field_comparison_slice,
                            j,
                            target_labels_dict,
                            output_labels_dict,
                            target_minmax_dict,
                            output_minmax_dict,
                            figure_dir,
                            figtext_kwargs,
                            comparison_slice_folder_name,
                        )
                        for j in field_slice_indexes
                    ]
                    for future in concurrent.futures.as_completed(future_2d_slices):
                        _ = future.result()

            # Histogram of output vs target
            nbins = 100
            fig, ax = plt.subplots(figsize=(self.figwidth * 1.3, self.figheight * 1))
            # looks better without ncols=1 and nrows=1, for some reason
            hist, bins = np.histogram(target_vector.flatten(), bins=nbins)
            ax.plot(bins[:-1], hist, label="target", color=self.target_color)
            hist, bins = np.histogram(output_vector.flatten(), bins=nbins)
            ax.plot(bins[:-1], hist, label="output", color=self.output_color)
            hist, bins = np.histogram(input_vector.flatten(), bins=nbins)
            ax.plot(bins[:-1], hist, label="input", color=self.input_color)
            # labeling
            ax.set_xlabel("Parameter magnitude")
            ax.set_ylabel("Count")
            ax.set_title("Histogram for comparing distributions of output and target fields")
            ax.legend()
            ax.grid()
            fig.text(**figtext_kwargs)  # ty:ignore[invalid-argument-type] fix with typeddict
            fig.tight_layout(pad=self.tight_layout_pad, rect=self.tight_layout_rect)
            fig.savefig(figure_dir / "output_distributions_comparison.png")
            plt.close(fig)

            # NOTE: consider saving separate plots for all field slices to be able to manually stitch together later?

            ## Power spectrum plotting
            if self.local_rank == 0:
                self.logger.info("Calculating power spectra...")
                start_Pk_calculation_time: int | float = time.monotonic()

            # on local systems, save time by only calculating spectra for a 2D slice of field
            if not self.config["local_run"]:
                power_spectrum_first_coord_slice = slice(None)  # full 3D calculation
            else:
                power_spectrum_first_coord_slice = 20

            fig, ax = plt.subplots(ncols=3, nrows=1, figsize=(self.figwidth * 3, self.figheight * 1))
            title_str = "Pure DM input"
            power_spectrum_dict[title_str] = self.calculate_power_spectrum(
                input_array=input_labels_dict[title_str].numpy()[power_spectrum_first_coord_slice, :, :]
            )
            self.plot_power_spectrum(ax=ax[0], Pk_inp=power_spectrum_dict[title_str], title_str=title_str, color=self.input_color)
            title_str = "Mixed DM target"
            power_spectrum_dict[title_str] = self.calculate_power_spectrum(
                input_array=target_labels_dict[title_str].numpy()[power_spectrum_first_coord_slice, :, :]
            )
            self.plot_power_spectrum(ax=ax[1], Pk_inp=power_spectrum_dict[title_str], title_str=title_str, color=self.target_color)
            title_str = "Mixed DM output"
            power_spectrum_dict[title_str] = self.calculate_power_spectrum(
                input_array=output_labels_dict[title_str].numpy()[power_spectrum_first_coord_slice, :, :]
            )
            self.plot_power_spectrum(ax=ax[2], Pk_inp=power_spectrum_dict[title_str], title_str=title_str, color=self.output_color)
            # common fig settings
            fig.suptitle("Power spectrum for Dark Matter fields")
            fig.text(**figtext_kwargs)  # ty:ignore[invalid-argument-type] fix with typeddict
            fig.tight_layout(pad=self.tight_layout_pad, rect=self.tight_layout_rect)
            fig.savefig(figure_dir / "DM_power_spectra.png")
            plt.close(fig)

            # power spectra for all fields
            fig, ax = plt.subplots(ncols=4, nrows=2, figsize=(self.figwidth * 4, self.figheight * 2))
            # target
            for i, (title_str, input_array) in enumerate(target_labels_dict.items()):
                try:
                    pk_calc = power_spectrum_dict[title_str]
                except KeyError:
                    pk_calc = self.calculate_power_spectrum(
                        input_array=input_array.numpy()[power_spectrum_first_coord_slice, :, :],
                    )
                    power_spectrum_dict[title_str] = pk_calc
                self.plot_power_spectrum(ax=ax[0, i], Pk_inp=pk_calc, title_str=title_str, color=self.target_color)
            # output
            for i, (title_str, input_array) in enumerate(output_labels_dict.items()):
                try:
                    pk_calc = power_spectrum_dict[title_str]
                except KeyError:
                    pk_calc = self.calculate_power_spectrum(
                        input_array=input_array.numpy()[power_spectrum_first_coord_slice, :, :],
                    )
                    power_spectrum_dict[title_str] = pk_calc
                self.plot_power_spectrum(ax=ax[1, i], Pk_inp=pk_calc, title_str=title_str, color=self.output_color)
            # common fig settings
            fig.suptitle("Power spectrum for all field components")
            fig.text(**figtext_kwargs)  # ty:ignore[invalid-argument-type] fix with typeddict
            fig.tight_layout(pad=self.tight_layout_pad, rect=self.tight_layout_rect)
            fig.savefig(figure_dir / "field_power_spectra.png")
            plt.close(fig)

            # power spectra overlaid and diffed
            fig, ax = plt.subplots(ncols=4, nrows=2, figsize=(self.figwidth * 4, self.figheight * 2))
            for i, title_str_1 in enumerate(target_labels_dict.keys()):
                title_str_2 = title_str_1.replace("target", "output")
                try:
                    pk_target_calc = power_spectrum_dict[title_str_1]
                    pk_output_calc = power_spectrum_dict[title_str_2]
                except KeyError:
                    pk_target_calc = self.calculate_power_spectrum(
                        input_array=target_labels_dict[title_str_1].numpy()[power_spectrum_first_coord_slice, :, :]
                    )
                    power_spectrum_dict[title_str_1] = pk_target_calc

                    pk_output_calc = self.calculate_power_spectrum(
                        input_array=output_labels_dict[title_str_2].numpy()[power_spectrum_first_coord_slice, :, :]
                    )
                    power_spectrum_dict[title_str_2] = pk_output_calc

                self.plot_power_spectrum_overlaid(
                    ax=ax[0, i],
                    Pk_1=pk_target_calc,
                    title_str_1=title_str_1,
                    color_1=self.target_color,
                    Pk_2=pk_output_calc,
                    title_str_2=title_str_2,
                    color_2=self.output_color,
                )
                self.plot_power_spectrum_difference(
                    ax=ax[1, i],
                    Pk_reference=pk_target_calc,
                    Pk_experiment=pk_output_calc,
                    title_str=f"{' '.join(title_str_1.split()[0:2])} P(k) difference",
                )
            # common fig settings
            fig.suptitle("Power spectrum comparisons for all field components")
            fig.text(**figtext_kwargs)  # ty:ignore[invalid-argument-type] fix with typeddict
            fig.tight_layout(pad=self.tight_layout_pad, rect=self.tight_layout_rect)
            fig.savefig(figure_dir / "field_power_spectra_comparison.png")
            plt.close(fig)

            # power spectra ratios plot
            fig, ax = plt.subplots(
                ncols=4,
                nrows=2,
                sharex=True,
                figsize=(self.figwidth * 4, self.figheight * 2),
                gridspec_kw={"height_ratios": [2, 1]},
            )
            for i, title_str_1 in enumerate(target_labels_dict.keys()):
                title_str_2 = title_str_1.replace("target", "output")
                title_str_3 = "Pure DM input"
                # power spectrum calculations, calculates if not found and sets dict
                pk_target_calc = power_spectrum_dict.setdefault(
                    title_str_1,
                    self.calculate_power_spectrum(
                        input_array=target_labels_dict[title_str_1].numpy()[power_spectrum_first_coord_slice, :, :]
                    ),
                )
                k_tgt: np.ndarray = pk_target_calc.k3D
                Pk_tgt: np.ndarray = pk_target_calc.Pk[:, 0]
                pk_output_calc = power_spectrum_dict.setdefault(
                    title_str_2,
                    self.calculate_power_spectrum(
                        input_array=output_labels_dict[title_str_2].numpy()[power_spectrum_first_coord_slice, :, :]
                    ),
                )
                k_out: np.ndarray = pk_output_calc.k3D
                Pk_out: np.ndarray = pk_output_calc.Pk[:, 0]
                max_k = np.max([k_tgt, k_out])
                # plot the input power spectrum too, but that only exists for dark matter
                if "DM" in title_str_1:
                    pk_input_calc = power_spectrum_dict.setdefault(
                        title_str_3,
                        self.calculate_power_spectrum(
                            input_array=input_labels_dict[title_str_3].numpy()[power_spectrum_first_coord_slice, :, :]
                        ),
                    )
                    k_inp: np.ndarray = pk_input_calc.k3D
                    Pk_inp: np.ndarray = pk_input_calc.Pk[:, 0]
                    ax[0, i].set_title("Dark Matter spectrum")
                else:
                    pk_input_calc = None
                    k_inp = None
                    Pk_inp = None
                    ax[0, i].set_title(title_str_1.replace("target", "spectrum"))

                if k_inp is not None and Pk_inp is not None:
                    ax[0, i].loglog(k_inp, Pk_inp, label="Input", color=self.input_color)
                ax[0, i].loglog(k_tgt, Pk_tgt, label="Target", color=self.target_color)
                ax[0, i].loglog(k_out, Pk_out, label="Output", color=self.output_color)
                ax[0, i].axvline(
                    max_k,
                    linestyle="--",
                    color=self.max_k_color,
                    label=f"Max $k={max_k:.2f}$ [$h$/Mpc]",
                )

                ax[0, i].grid()
                ax[0, i].legend()

                # ax[1, i].plot(k_out, np.ones_like(k_out), color="k", alpha=0.7)
                ax[1, i].axhline(1.0, color="k", alpha=0.7)
                if k_inp is not None and Pk_inp is not None:
                    ax[1, i].plot(k_inp, Pk_inp / Pk_tgt, label=r"P$_{\rm in}$ / P$_{\rm tgt}$", color=self.input_color)
                ax[1, i].plot(k_out, Pk_out / Pk_tgt, label=r"P$_{\rm out}$ / P$_{\rm tgt}$", color=self.output_color)

                ax[0, i].set_ylabel(r"Power Spectrum $\left[ \left( \text{Mpc} / h \right)^3 \right]$")
                ax[1, i].set_ylabel("Ratio")

                ax[1, i].set_ylim(0.5, 1.3)
                ax[1, i].set_xscale("log")
                ax[1, i].set_xlabel(r"$k$ $\left[ {h} / {\text{Mpc}} \right]$")

                ax[1, i].axhline(0.9, linestyle="--", color="k", alpha=0.5)
                ax[1, i].axhline(1.1, linestyle="--", color="k", label=r"$\pm$10%", alpha=0.5)
                ax[1, i].axvline(
                    max_k,
                    linestyle="--",
                    color=self.max_k_color,
                    label=f"Max $k={max_k:.2f}$ [$h$/Mpc]",
                )

                ax[1, i].grid()
                ax[1, i].legend()
            # common fig settings
            fig.suptitle("Power spectra + ratios for all field components")
            fig.text(**figtext_kwargs)  # ty:ignore[invalid-argument-type] fix with typeddict
            fig.tight_layout(pad=self.tight_layout_pad, rect=self.tight_layout_rect)
            fig.subplots_adjust(hspace=0.03)
            fig.savefig(figure_dir / "field_power_spectra_ratios.png")
            plt.close(fig)

            if self.local_rank == 0:
                end_Pk_calculation_time = time.monotonic()
                self.logger.info("Power spectrum figures saved.")
                total_Pk_calculation_time = timedelta(seconds=end_Pk_calculation_time - start_Pk_calculation_time)
                self.logger.info(f"Total power spectrum calculation time: {utils.pretty_timedelta(total_Pk_calculation_time)}")

        # finish and clean up resources
        del input_vector
        del input_style
        del target_vector
        del output_vector
        del data

    def calculate_power_spectrum(self, input_array: np.ndarray, **kwargs: str) -> Pk:
        """Calculate a power spectrum for the input array"""
        Pk_inp = Pk(
            delta=input_array,
            BoxSize=self.test_dataset_config["BoxSize"],
            axis=1,
            threads=self.config["max_usable_threads_per_gpu"],
            # difference in speed from threading is very small
            MAS=self.test_dataset_config["MAS"],
            verbose=False,
        )

        return Pk_inp

    def plot_power_spectrum(self, ax: plt.Axes, Pk_inp: Pk, title_str: str, **kwargs: str) -> None:  # noqa: N803
        """Plot a previously calculated power spectrum on the given axis"""
        k_inp: np.ndarray = Pk_inp.k3D
        Pk_inp: np.ndarray = Pk_inp.Pk[:, 0]

        ax.loglog(k_inp, Pk_inp, **kwargs)
        ax.set_title(title_str)
        ax.set_xlabel(r"$k$ $\left[ {h} / {\text{Mpc}} \right]$")
        ax.set_ylabel(r"P($k$) $\left[ \left( \text{Mpc} / h \right)^3 \right]$")

        ax.grid()
        return None

    def plot_power_spectrum_overlaid(
        self,
        ax: plt.Axes,
        Pk_1: Pk,  # noqa: N803
        title_str_1: str,
        color_1: str,
        Pk_2: Pk,  # noqa: N803
        title_str_2: str,
        color_2: str,
        **kwargs: str,
    ) -> None:
        """Plot two previously calculated power spectra on the given axis"""
        k_1: np.ndarray = Pk_1.k3D
        Pk_1: np.ndarray = Pk_1.Pk[:, 0]
        k_2: np.ndarray = Pk_2.k3D
        Pk_2: np.ndarray = Pk_2.Pk[:, 0]

        ax.loglog(k_1, Pk_1, label=title_str_1, color=color_1)
        ax.loglog(k_2, Pk_2, label=title_str_2, color=color_2)
        ax.set_title(f"{' '.join(title_str_1.split()[0:2])} P(k)")  # a bit sneaky, steal the component name from title 1
        ax.set_xlabel(r"$k$ $\left[ {h} / {\text{Mpc}} \right]$")
        ax.set_ylabel(r"$P(k)$ $\left[ \left( \text{Mpc} / h \right)^3 \right]$")

        ax.grid()
        ax.legend()
        return None

    def plot_power_spectrum_difference(
        self,
        ax: plt.Axes,
        Pk_reference: Pk,  # noqa: N803
        Pk_experiment: Pk,  # noqa: N803
        title_str: str,
        **kwargs: str,
    ) -> None:
        """Plot the relative difference between two power spectra on the given axis"""
        k_1: np.ndarray = Pk_reference.k3D
        k_2: np.ndarray = Pk_experiment.k3D  # noqa: F841
        Pk_ref: np.ndarray = Pk_reference.Pk[:, 0]
        Pk_exp: np.ndarray = Pk_experiment.Pk[:, 0]

        rel_diff = np.abs(Pk_ref - Pk_exp) / np.abs(Pk_ref)

        ax.plot(k_1, rel_diff * 100, color="red")
        ax.set_xscale("log")
        ax.set_yscale("log")
        ax.set_title(title_str)
        ax.set_xlabel(r"$k$ $\left[ {h} / {\text{Mpc}} \right]$")
        ax.set_ylabel("% relative error in $P(k)$")

        ax.grid()
        return None

    # TODO: make the spam of arguments class attributes probably
    def make_2d_field_comparison_slice(
        self,
        subvolume_first_coord: int,
        target_labels_dict: dict[str, torch.Tensor],
        output_labels_dict: dict[str, torch.Tensor],
        target_minmax_dict: dict[str, float],
        output_minmax_dict: dict[str, float],
        figure_dir: Path,
        figtext_kwargs: dict[str, Any],
        comparison_slice_folder_name: str,
    ) -> None:
        """Makes and saves figure for the slice j of a set of 3D fields"""
        fig, ax = plt.subplots(ncols=4, nrows=2, figsize=(self.figwidth * 4, self.figheight * 2))
        # target
        imgs_target: list[plt.AxesImage] = []
        for i, (key, vector) in enumerate(target_labels_dict.items()):
            colormap_to_use = self.field_colormaps[key]
            imgs_target.append(
                ax[0, i].imshow(
                    vector[subvolume_first_coord, :, :],
                    cmap=colormap_to_use,
                    norm=mpl.colors.Normalize(
                        # mpl.colors.SymLogNorm
                        # linthresh=0.01,
                        # linscale=1.0,
                        # base=10,
                        vmin=target_minmax_dict[key + " min"],
                        vmax=target_minmax_dict[key + " max"],
                    ),
                )
            )
            ax[0, i].set_title(key)
            cax = ax[0, i].inset_axes(self.axes_inset)
            fig.colorbar(imgs_target[i], cax=cax, orientation="vertical", extend="neither")
        # output
        imgs_output: list[plt.AxesImage] = []
        for i, (key, vector) in enumerate(output_labels_dict.items()):
            colormap_to_use = self.field_colormaps[key]
            imgs_output.append(
                ax[1, i].imshow(
                    vector[subvolume_first_coord, :, :],
                    cmap=colormap_to_use,
                    norm=mpl.colors.Normalize(
                        # mpl.colors.SymLogNorm
                        # linthresh=0.01,
                        # linscale=1.0,
                        # base=10,
                        vmin=output_minmax_dict[key + " min"],
                        vmax=output_minmax_dict[key + " max"],
                    ),
                )
            )
            ax[1, i].set_title(key)
            cax = ax[1, i].inset_axes(self.axes_inset)
            fig.colorbar(imgs_output[i], cax=cax, orientation="vertical", extend="neither")

        # NOTE: could set colorbar max and min values to be based on target, rather than output
        #       and then use extend="both", which would mark outlier values compared to target distribution
        #       not really "better", just different

        # final fig settings
        fig.suptitle(
            f"2D slices of all field components\nSlice at x={
                self.test_dataset_config['crop_start'] + subvolume_first_coord + 1  # ty:ignore[unsupported-operator]
            }"
        )
        fig.text(**figtext_kwargs)
        fig.tight_layout(pad=self.tight_layout_pad, rect=self.tight_layout_rect)
        fig.savefig(
            figure_dir
            / comparison_slice_folder_name
            / f"field_comparison_slice_{self.test_dataset_config['crop_start'] + subvolume_first_coord + 1}.png"  # ty:ignore[unsupported-operator]
        )
        plt.close(fig)
        return None

    def setup_train_dataloader(self) -> None:
        """Sets up a map2map dataloader"""
        # tied to config
        self.train_dataset: FieldDataset = FieldDataset(
            norm_config=self.norm_config,
            **self.train_dataset_config,
        )
        self.train_sampler: DistFieldSampler = DistFieldSampler(
            dataset=self.train_dataset,
            shuffle=self.train_dataset_config["shuffle"],
            div_data=self.train_dataset_config["div_data"],
            div_shuffle_dist=self.train_dataset_config["div_shuffle_dist"],
        )

        self.train_loader: DataLoader[FieldDataOutputItem] = DataLoader(
            dataset=self.train_dataset,
            batch_size=self.config["batch_size"],
            shuffle=self.config["shuffle_dataloader"],
            sampler=self.train_sampler,
            num_workers=self.config["dataloader_num_workers"],
            pin_memory=self.config["pin_memory"],
        )

        self.train_in_chan: list = self.train_dataset.in_chan
        self.train_out_chan: list = self.train_dataset.tgt_chan
        self.train_style_size: int = self.train_dataset.style_size

        self.train_dataset_size: np.ndarray = self.train_dataset.size
        self.train_dataset_ndim: int = self.train_dataset.ndim

        if self.local_rank == 0:
            self.logger.debug("Training dataloader info:")
            self.logger.debug(f"Input channels: {self.train_in_chan}")
            self.logger.debug(f"Output channels: {self.train_out_chan}")
            self.logger.debug(f"Style size: {self.train_style_size}")
            self.logger.debug(f"Dataset size: {self.train_dataset_size}")
            self.logger.debug(f"Dataset ndim: {self.train_dataset_ndim}")

    def setup_test_dataloader(self) -> None:
        """Sets up a map2map dataloader for test data"""
        self.test_dataset: FieldDataset = FieldDataset(
            norm_config=self.norm_config,
            **self.test_dataset_config,
        )
        self.test_sampler: DistFieldSampler = DistFieldSampler(
            dataset=self.test_dataset,
            shuffle=self.test_dataset_config["shuffle"],
            div_data=self.test_dataset_config["div_data"],
            div_shuffle_dist=self.test_dataset_config["div_shuffle_dist"],
        )

        self.test_loader: DataLoader[FieldDataOutputItem] = DataLoader(
            dataset=self.test_dataset,
            batch_size=self.config["batch_size"],
            shuffle=self.config["shuffle_dataloader"],
            sampler=self.test_sampler,
            num_workers=self.config["dataloader_num_workers"],
            pin_memory=self.config["pin_memory"],
        )

        self.test_in_chan: list = self.test_dataset.in_chan
        self.test_out_chan: list = self.test_dataset.tgt_chan
        self.test_style_size: int = self.test_dataset.style_size

        self.test_dataset_size: np.ndarray = self.test_dataset.size
        self.test_dataset_ndim: int = self.test_dataset.ndim

        if self.local_rank == 0:
            self.logger.debug("Testing dataloader info:")
            self.logger.debug(f"Input channels: {self.test_in_chan}")
            self.logger.debug(f"Output channels: {self.test_out_chan}")
            self.logger.debug(f"Style size: {self.test_style_size}")
            self.logger.debug(f"Dataset size: {self.test_dataset_size}")
            self.logger.debug(f"Dataset ndim: {self.test_dataset_ndim}")

    # TODO: test normalization + plots output and such somewhere else
    def check_normalization(
        self,
        input_array: np.ndarray,
        norm_func: FunctionType,
        *,
        plot_title: str = "unknown array",
        print_tensor: bool = False,
        **kwargs: float,
    ) -> None:
        """Check normalizing function by creating a histogram for value distribution"""
        array_tensor: torch.Tensor = torch.from_numpy(input_array.astype(np.float32))

        binning_values = [-10, -1, 0, 0.1, 1, 2, 10, 100, 1000, 10**4, 10**5]

        fig, ax = plt.subplots()
        fig.set_label(plot_title)
        # hist, bins = np.histogram(array_tensor.flatten(), bins=binning_values)
        # ax.plot(bins[:-1], hist, label="gas array in")
        ax.hist(array_tensor.flatten(), bins=binning_values, label="array in", log=True, edgecolor="black")

        print("input")
        print(f"max value: {array_tensor.max(): _.4f}")
        print(f"min value: {array_tensor.min(): _.4f}")
        print(f"average value: {array_tensor.mean(): _.4f}")
        print(f"median value:  {array_tensor.median(): _.4f}")
        print(
            f"zero values: {array_tensor.numel() - array_tensor.count_nonzero():_}; non-zero values: {array_tensor.count_nonzero():_}\
    ; total values: {array_tensor.numel():_}".replace("_", " ")
        )
        if print_tensor:
            print(array_tensor[0, 0, 0, :, :])
        norm_func(array_tensor, **kwargs)

        # hist, bins = np.histogram(array_tensor.flatten(), bins=binning_values)
        # ax.plot(bins[:-1], hist, label="gas array normalized")
        ax.hist(array_tensor.flatten(), bins=binning_values, label="array normalized", log=True, edgecolor="black")

        ax.set_xlabel("parameter magnitude")
        ax.set_ylabel("count")
        # ax.loglog()
        ax.set_xscale("symlog", linthresh=1)
        ax.set_title(plot_title)
        ax.legend()
        # ax.grid()

        print("norm")
        print(f"max value: {array_tensor.max(): _.4f}")
        print(f"min value: {array_tensor.min(): _.4f}")
        print(f"average value: {array_tensor.mean(): _.4f}")
        print(f"median value:  {array_tensor.median(): _.4f}")
        print(
            f"zero values: {array_tensor.numel() - array_tensor.count_nonzero():_}; \
non-zero values: {array_tensor.count_nonzero():_}; \
total values: {array_tensor.numel():_}".replace("_", " ")
        )
        if print_tensor:
            print(array_tensor[0, 0, 0, :, :])

        # plt.show()
        return None

    def run_normalization_check(self) -> None:
        T_file = Path(self.config["data_dir_path"]) / "T 1.npy"
        T_array = np.load(T_file)[None]
        # print(T_array)
        print(f"{T_array.size:_}".replace("_", " "))
        print(T_array.shape)

        # check normalizing function with sinh norm
        self.check_normalization(
            T_array,
            ember2_norms.T_norm,
            z=1,
            exponent=-0.5,
            m_t_mult=-0.193,
            m_t_add=5.346,
            print_tensor=False,
            plot_title="T array",
        )

        DM_only_file = Path(self.config["data_dir_path"]) / "DM_only 2.npy"
        DM_only_array = np.load(DM_only_file)[None]
        # print(DM_only_array)
        print(DM_only_array.size)
        print(DM_only_array.shape)

        # check normalizing function with sinh norm
        self.check_normalization(
            DM_only_array,
            ember2_norms.dm_in_norm,
            eps=1e-7,
            exponent=-3,
            log_mult=1.0 / 4.0,
            print_tensor=False,
            plot_title="DM only array",
        )

        DM_file = Path(self.config["data_dir_path"]) / "DM 1.npy"
        DM_array = np.load(DM_file)[None]
        # print(DM_array)
        print(DM_array.size)
        print(DM_array.shape)

        # check normalizing function with sinh norm
        self.check_normalization(
            DM_array,
            ember2_norms.dm_norm,
            exponent=-3,
            log_mult=1.0 / 4.0,
            print_tensor=False,
            plot_title="DM array",
        )

        E_file = Path(self.config["data_dir_path"]) / "E 1.npy"
        E_array = np.load(E_file)[None]
        # print(E_array)
        print(E_array.size)
        print(E_array.shape)

        # check normalizing function with sinh norm
        self.check_normalization(
            E_array,
            ember2_norms.E_norm,
            z=0,
            exponent=-3,
            m_t_mult=-0.193,
            m_t_add=5.346,
            print_tensor=False,
            plot_title="E array",
        )

        gas_file: Path = Path(self.config["data_dir_path"]) / "gas 1.npy"
        gas_array: np.ndarray = np.load(gas_file)[None]
        # print(gas_array)
        print(gas_array.size)
        print(gas_array.shape)

        # check normalizing function with generic norm
        self.check_normalization(
            gas_array,
            ember2_norms.generic_norm,
            z=1,
            k=1.0,
            x_0=2.0,
            q=1.6,
            plot_title="Gas array generic norm",
        )

        # check normalizing function with sinh norm
        self.check_normalization(
            gas_array,
            ember2_norms.gas_norm,
            z=1,
            print_tensor=True,
            plot_title="Gas array specialized norm",
        )

    def generate_random_style_vector(self) -> None:
        if self.local_rank == 0:
            style_file = Path(self.config["data_dir_path"]) / "random_style.npy"
            rng = np.random.default_rng()
            random_style = rng.normal(size=(6))
            self.logger.info(f"Style vector shape: {random_style.shape}")
            np.save(style_file, random_style)


## TODO: re-adapt method for checking plots before and after normalization (passing through the map2map dataloader)
