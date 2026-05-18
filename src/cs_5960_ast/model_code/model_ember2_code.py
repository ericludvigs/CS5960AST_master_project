import logging
import os
from pathlib import Path
from typing import Any, override

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.distributed as dist  # noqa: F401 make sure multiprocessing is enabled
import torch.multiprocessing as mp  # noqa: F401
from torch import nn
from torch.optim import AdamW
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from tqdm.auto import tqdm

from cs_5960_ast import AstNetConfigDict
from cs_5960_ast.ember2.modules import Discriminator, Generator
from cs_5960_ast.map2map.src.data import DistFieldSampler, FieldDataOutputItem
from cs_5960_ast.methods import narrow, utils
from cs_5960_ast.methods.utils import NSLoss, pair_base


class EMA:
    """EMBER-2 inherited method for setting a moving average, improves model performance"""

    def __init__(self, beta: float) -> None:
        super().__init__()
        self.beta = beta

    def update_average(self, old: torch.Tensor | None, new: torch.Tensor) -> torch.Tensor:
        if old is None:
            return new
        return old * self.beta + (1 - self.beta) * new


class Ember2NetWrapper(nn.Module):
    """Customized version of EMBER-2 model, adapted to 3D

    Parameters
    ----------
    net_config : dict
        A configuration dictionary with key/values to use defined in `cs_5960_ast.AstNetConfigDict`.
    device : torch.device
        The pytorch device computations are being performed on. Model should be moved to this device after init.
    ckpt_freq : int, default=10
        How often to save model checkpoints. Also calculates and prints relative difference in loss between checkpoints.
    current_epoch : int, default=1
        For restoring from a model checkpoint, set the checkpoint epoch here. Model will resume training from this point.
    logger : logging.Logger, optional
        An instantiated python logger to use for outputting printouts.
        If not specified (None), creates a new `cs_5960_ast.model` logger that inherits `cs_5960_ast` module settings.
    tensorboard_writer : torch.utils.tensorboard.SummaryWriter, optional
        A pytorch tensorboard writer, to be used for visualization of metrics via tensorboard. If not specified, nothing is logged.
    """

    def __init__(
        self,
        net_config: AstNetConfigDict,
        device: torch.device,
        figure_dir: Path,
        ckpt_freq: int = 10,
        current_epoch: int = 1,
        logger: logging.Logger | None = None,
        tensorboard_writer: SummaryWriter | None = None,
    ) -> None:
        # see class docstring
        super().__init__()

        # total number of training steps - both data batches and epochs
        self.step = 1

        # the number of data batches to train over, aka len(dataloader)
        # updated dynamically
        self.num_batches = 0

        self.tensorboard_writer = tensorboard_writer
        self.current_epoch: int = current_epoch
        # self.example_input = None

        # where to place the figures from training
        self.figure_dir = figure_dir
        self.training_fields_name = "training_fields"
        (self.figure_dir / self.training_fields_name).mkdir(parents=True, exist_ok=True)

        if logger is None:
            self.logger = logging.getLogger("cs_5960_ast.model")
        else:
            self.logger = logger

        mode = "base"  # zoom mode from ember2 is not supported, just use the "normal" base mode
        self.mode: str = mode
        self.net_config: AstNetConfigDict = net_config
        self.device: torch.device = device

        # do NOT use non_blocking with these modules, it breaks the entire model somehow
        self.G: Generator = Generator(mode, net_config).to(device)
        if device.type == "cuda" and torch.cuda.get_device_capability(device) >= (7, 0):
            self.G.compile()
        self.D: Discriminator = Discriminator(mode, **net_config).to(device)
        if device.type == "cuda" and torch.cuda.get_device_capability(device) >= (7, 0):
            self.D.compile()
        # ember2 did it that way with one method only unwrapping the config and the others using it as a dict object, dont know why
        self.GE: Generator = Generator(mode, net_config).to(device)
        if device.type == "cuda" and torch.cuda.get_device_capability(device) >= (7, 0):
            self.GE.compile()

        # this is not an optimizer, used to keep track of the error against target during training for user
        self.error: nn.MSELoss = nn.MSELoss()
        # self.error = nn.HuberLoss(delta=0.1)

        if device.type == "cuda":
            self.cuda(device=device)

        self.ema = EMA(self.net_config["beta_ema"])
        self.configure_optimizers()

        self.ckpt_freq: int = ckpt_freq
        self.dist_status = utils.str_to_bool(os.getenv("DISTRIBUTED", default=None))
        self.tqdm_disable: bool = os.getenv("TQDM_DISABLE", "0") == "1"
        self.local_rank: int = int(os.getenv("LOCAL_RANK", "-1"))
        # only the main process should use tqdm, if it is not turned off globally
        self.should_disable_tqdm: bool = (self.tqdm_disable) or (self.local_rank != 0)
        # we only need infrequent status checks when distributed, since the script is intended
        # to be ran as a background job without a console connected
        if self.dist_status:
            self.tqdm_mininterval = 60.0  # seconds
        else:
            self.tqdm_mininterval = 0.1  # seconds (default)

        self.compile()
        return None

    # when inheriting nn.Module
    # type check on torchs setattr definition is too narrow, it supports normal parameter saving in an else statement
    # but type check only allows torch.tensors and torch.modules, which causes weird type errors
    @override
    def __setattr__(self, name: str, value: Any | torch.Tensor | nn.Module) -> None:
        """Set attribute for class, just calls torch.nn.Module's own __setattr__"""
        super().__setattr__(name, value)

    def forward(self, x: torch.Tensor, style: torch.Tensor, noise: torch.Tensor | None = None) -> tuple[torch.Tensor, torch.Tensor]:
        """Generate a tensor with the .G and .GE generator belonging to the ember2 network

        Not sure if pytorch uses this forward

        EMBER-2 simply output result from GE in the end, with a code comment "# G or GE?"
        G is the normal model, GE has been through moving average.
        Which model is better depends on training and hyperparameters.

        In order to register model shape with pytorch, G, GE, and D networks are all called even if only G is used

        See: https://github.com/lanpa/tensorboardX/issues/319
        """
        out_G = self.G(x, style, noise=noise)
        out_GE = self.GE(x, style, noise=noise)
        _f: torch.Tensor = self.D(pair_base(x=x, y=out_G, p=out_GE, context=style))

        return out_G, out_GE

    def configure_optimizers(self) -> None:
        """Set up optimizer functions for the generator and discriminator networks

        Uses AdamW for optimizing function. Parameters set by config system, see `default_configs.py`.

        EMBER-2 used lr=1e-4 and betas=(0.5, 0.9) for both optimizers.

        Default betas from pytorch are (0.0, 0.99).
        """
        # NOTE: can use Adam or AdamW, difference seems minor
        # AdamW is supposed to be the best general case optimizer
        lr_g = self.net_config["lr_g"]
        betas_g = self.net_config["betas_g"]
        lr_d = self.net_config["lr_d"]
        betas_d = self.net_config["betas_d"]
        self.opt_g = AdamW(self.G.parameters(), lr=lr_g, betas=betas_g)
        self.opt_d = AdamW(self.D.parameters(), lr=lr_d, betas=betas_d)

        return None

    def call_ema(self) -> None:
        def update_moving_average(current_model: nn.Module, ma_model: nn.Module) -> None:
            for current_params, ma_params in zip(current_model.parameters(), ma_model.parameters(), strict=True):
                old_weight, up_weight = ma_params.data, current_params.data
                ma_params.data = self.ema.update_average(old_weight, up_weight)

        update_moving_average(self.G, self.GE)

    def train_epoch(
        self,
        train_dataloader: DataLoader[FieldDataOutputItem],
        train_sampler: DistFieldSampler,
        # validation_dataloader: DataLoader,
        epoch_num: int,
        break_point: int | None = None,
    ) -> None:
        """Train the custom EMBER-2 model for one epoch

        Expects a standard train, ~~validation~~, test split for input data.
        Model should not receive "test" data during training.

        Contains progress bar setup for iterating through the data batches.

        Parameters
        ----------
        train_dataloader : torch.utils.data.DataLoader
            An iterable `Pytorch` DataLoader object with training data.
        validation_dataloader : torch.utils.data.DataLoader
            An iterable `Pytorch` DataLoader object with validation data for verification of progress during training.
            Do not use "test" data for this, split data in independent "train-validation-test" sets.
        epoch_num : int
            The current epoch that is being trained.
        break_point: int | None, default=None
            After this many batches, breaks out of dataloader loop. Allows for cut-off, faster training runs.
            Intended for testing and iteration, not for training "real" models.

        Returns
        -------
        `None`
            Modifies active class instance, thus returning nothing. Use `predict_ember2()` to run inference for model.
        """
        # this assumes that number of data samples cannot change during an epoch, which should be fine
        self.num_batches = len(train_dataloader)
        self.mse_error_during_epoch: np.ndarray = np.zeros(self.num_batches, dtype=np.float64)

        train_sampler.set_epoch(epoch_num)

        # NOTE: breakpoint logic to not have to run through all the training data
        # None as input means do not break, hence setting it to an unreachable number
        if break_point is None:
            break_point = len(train_dataloader) + 1
        self.break_point = break_point

        # fallback when running without tqdm, for example for logfiles
        if self.tqdm_disable and self.local_rank == 0:
            self.logger.info(f"Starting epoch: {epoch_num}")

        # data batches loop
        # tqdm creates a progress bar for stepping through the training data
        for batch_num, data in enumerate(
            tqdm(  # progress bar for dataloader
                train_dataloader,
                desc="Training data progress",
                position=1,
                disable=self.should_disable_tqdm,
                mininterval=self.tqdm_mininterval,
            )
        ):
            data: FieldDataOutputItem

            self.train_step(data, batch_num, epoch_num)

            if (epoch_num <= 2 and batch_num <= 1) and self.local_rank == 0:  # noqa: PLR2004
                # only prints along with other information for some early epochs and batches
                # see train_step for rest of logging
                self.logger.debug(f"Finished training step {self.step}.")

            # NOTE: breakpoint ends loop early
            # use on laptop to not sit there for 20 minutes just to see if the pipeline works
            # we want *a* trained model to check results print and plots are created
            # it doesn't need to be a *good* model
            if batch_num >= break_point:
                break
            else:
                continue

        # log memory stats at the end of epoch
        if self.local_rank == 0 and self.tensorboard_writer is not None:
            # mem_get_info only works with cuda
            if self.device.type == "cuda":
                free_mem, total_mem = torch.cuda.mem_get_info(self.device)
                mem_used_GB = round((total_mem - free_mem) / 1024**3, 2)
                mem_utilization_percent = round((1 - (free_mem / total_mem)) * 100, 2)

                self.tensorboard_writer.add_scalar(tag="Memory/GB_used", scalar_value=mem_used_GB, global_step=self.step)
                self.tensorboard_writer.add_scalar(
                    tag="Memory/percent_used", scalar_value=mem_utilization_percent, global_step=self.step
                )
            elif self.device.type == "mps":
                alloc = torch.mps.current_allocated_memory()
                alloc_driver = torch.mps.driver_allocated_memory()
                max_usable = torch.mps.recommended_max_memory()
                mem_used_GB = round((alloc + alloc_driver) / 1024**3, 2)
                mem_utilization_percent = round(((alloc + alloc_driver) / max_usable) * 100, 2)

                self.tensorboard_writer.add_scalar(tag="Memory/GB_used", scalar_value=mem_used_GB, global_step=self.step)
                self.tensorboard_writer.add_scalar(
                    tag="Memory/percent_used", scalar_value=mem_utilization_percent, global_step=self.step
                )

        del data  # clean up data after finishing epoch
        return None

    def train_step(self, data: FieldDataOutputItem, batch_num: int, epoch_num: int) -> None:
        """Run a single data batch through training process

        Should be called during an active epoch of training for all batches.
        """
        use_adversarial = self.net_config["use_adversarial"]
        # still inits D, too annoying to track down all places to fix, just let it do nothing

        # unpack data
        input_vector: torch.Tensor = data["input"].to(self.device, non_blocking=True)
        input_style: torch.Tensor = data["style"].to(self.device, non_blocking=True)
        target_vector: torch.Tensor = data["target"].to(self.device, non_blocking=False)
        # these tensors can be loaded simultaneously, but the last one should sync

        # log for the first 2 epoch, the first 2 batches - for debugging purposes
        if (epoch_num <= 1 and batch_num <= 1) and self.local_rank == 0:
            self.logger.debug(f"Training data shapes at epoch {epoch_num}, batch {batch_num}/{self.num_batches}:")
            self.logger.debug(f" input_vector: {input_vector.shape}")
            self.logger.debug(f" input_style: {input_style.shape}")
            self.logger.debug(f" target_vector: {target_vector.shape}")
            self.logger.debug(f" relative paths of input data: \n{data['input_relpath']}")
            self.logger.debug(f" relative paths of target data: \n{data['target_relpath']}")

        # training process
        self.opt_g.zero_grad(set_to_none=True)
        utils.toggle_grad(self.G, requires_grad=True)
        # self.G.requires_grad_(requires_grad=True)
        utils.toggle_grad(self.D, requires_grad=False)
        # self.D.requires_grad_(requires_grad=False)

        # create two sets of independent noise, the way the ember2 paper describes
        # 8 channels of gaussian (normal) noise
        if self.net_config["use_noise"] is True:
            # if bool is passed, new mapping net parameter-tuned noise injection is used
            # with tensor, that tensor is concatenated directly (fixed noise)
            noise1 = True
            noise2 = True
            # noise1: torch.Tensor = utils.make_noise(input_vector, c=self.net_config["num_noise_ch"])
            # noise2: torch.Tensor = utils.make_noise(input_vector, c=self.net_config["num_noise_ch"])
        else:
            noise1 = None
            noise2 = None

        # generate output, remove padding from output vectors
        p1: torch.Tensor = self.G(input_vector, input_style, noise=noise1)
        p1: torch.Tensor = narrow.narrow_like(p1, target_vector)
        p2: torch.Tensor = self.G(input_vector, input_style, noise=noise2)
        p2: torch.Tensor = narrow.narrow_like(p2, target_vector)

        # more debugging output
        # incidentally difference in timestamp between initial log and this one tells you how long generation takes
        if (epoch_num <= 2 and batch_num <= 1) and self.local_rank == 0:  # noqa: PLR2004
            self.logger.debug(f" generated p1: {p1.shape}")
            self.logger.debug(f" generated p2: {p2.shape}")

        # dropped because add_graph() was not working well anyway
        # if self.example_input is None and self.local_rank == 0:
        #    # save an example set of input for the model for later logging purposes
        #    self.example_input = (input_vector, input_style, noise1)
        #    # saves only once

        # calculate MSE as an error metric and for a small, baseline error
        mse_error_1: torch.Tensor = self.error(p1, target_vector)
        mse_error_2: torch.Tensor = self.error(p2, target_vector)

        # balance MSE between the 2 generated outputs - this could be a silly thing to do
        mse_error: torch.Tensor = mse_error_1 * 0.5 + mse_error_2 * 0.5

        # update G with adversarial loss if enabled
        if use_adversarial:
            # remove padding from input before use in discriminator
            input_vector = narrow.narrow_like(input_vector, target_vector)

            # discriminator + loss
            f: torch.Tensor = self.D(pair_base(x=input_vector, y=p1, p=p2, context=input_style))
            g_loss = NSLoss(r=None, f=f, mode="G")

            # add mse error as a very small baseline for adversarial loss, to get generator to try and optimize out the error
            # but mse should be less emphasized than the main adversarial loss
            g_loss = g_loss + (mse_error * 10 ** (-3))

        # if no adversarial, just use MSE
        else:
            g_loss = mse_error

        # unified loss update
        g_loss.backward()
        self.opt_g.step()

        # train D
        if use_adversarial:
            self.opt_d.zero_grad(set_to_none=True)
            utils.toggle_grad(self.G, requires_grad=False)
            utils.toggle_grad(self.D, requires_grad=True)

            p1: torch.Tensor = p1.detach()
            p2: torch.Tensor = p2.detach()

            r: torch.Tensor = self.D(pair_base(x=input_vector, y=target_vector, p=target_vector, context=input_style))
            f: torch.Tensor = self.D(pair_base(x=input_vector, y=p1, p=p2, context=input_style))

            # update D
            d_loss = NSLoss(r, f, mode="D")
            d_loss.backward()
            self.opt_d.step()

            # debugging, but timestamp also tells you how long discrimination takes
            if (epoch_num <= 2 and batch_num <= 1) and self.local_rank == 0:  # noqa: PLR2004
                self.logger.debug(f"Discriminator outputs at epoch {epoch_num}, batch {batch_num + 1}/{self.num_batches}:")
                self.logger.debug(f" r (real): {r.shape}")
                self.logger.debug(f" f (fake): {f.shape}")
        # if no D, just copy these tensors
        else:
            p1: torch.Tensor = p1.detach()
            p2: torch.Tensor = p2.detach()

        # add error to metric tracking
        self.mse_error_during_epoch[batch_num] = mse_error.detach().cpu()

        # do effective moving average every 10 steps after warming up model
        warmup_steps = 50000
        if self.step % 10 == 0 and self.step > warmup_steps:
            self.call_ema()

        # log losses into tensorboard
        if self.local_rank == 0 and self.tensorboard_writer is not None:
            # dict here is just a way to make the add_scalar code below loopable
            values_to_log: dict[str, torch.Tensor] = {
                "Loss/mse": mse_error,
                "Loss/g_adv_loss": g_loss,
            }
            if use_adversarial:
                values_to_log.update(
                    {
                        "Loss/d_adv_loss": d_loss,
                        "Score/dfake_score": f,
                        "Score/dtrue_score": r,
                    }
                )
            for key, value in values_to_log.items():
                scalar: float = value.mean().detach().item()
                self.tensorboard_writer.add_scalar(tag=key, scalar_value=scalar, global_step=self.step)

        # log a picture of generation vs input/target every checkpoint into tensorboard
        # should only happen for the last batch of data, including if breaking early on data loop
        if batch_num + 1 in (self.num_batches, self.break_point) and epoch_num % self.ckpt_freq == 0 and self.local_rank == 0:
            self.logger.debug(
                f"Finished epoch {epoch_num}, final batch {batch_num + 1}/{self.num_batches}, mse error: {mse_error.item()}"
            )
            self.log_checkpoint_image(epoch_num, input_vector, target_vector, p1, p2)

        self.step += 1
        return None

    def predict_ember2(
        self, input_vector: torch.Tensor, input_style: torch.Tensor, pad_size: int | tuple[int, ...], *, noise: bool = False
    ) -> torch.Tensor:
        """Run a prediction through the EMBER-2 model

        Input and style tensors should be on correct device (like GPU) already before calling predict.

        NOTE: This will remove padding from output vector before returning according to pad_size. Pass 0 to disable.

        NOTE: Remember to remove padding from input vector after using it!

        NOTE: If target is padded, add padding to output vector to match!
        """
        self.eval()
        with torch.no_grad():
            output_vector: torch.Tensor = self.G(input_vector, input_style, noise=noise)
            output_vector = narrow.narrow_by(output_vector, pad_size)
            # TODO: fix tuple stuff
        return output_vector

    def predict_ember2_ge(
        self, input_vector: torch.Tensor, input_style: torch.Tensor, pad_size: int | tuple[int, ...], *, noise: bool = False
    ) -> torch.Tensor:
        """For attempting predictions with the alternate GE generator module

        GE seems to be the G model after running through moving average on its weights

        Input and style tensors should be on correct device (like GPU) already before calling predict.

        NOTE: This will remove padding from output vector before returning according to pad_size. Pass 0 to disable.

        NOTE: Remember to remove padding from input vector after using it!

        NOTE: If target is padded, add padding to output vector to match!
        """
        self.eval()
        with torch.no_grad():
            output_vector: torch.Tensor = self.GE(input_vector, input_style, noise=noise)
            output_vector = narrow.narrow_by(output_vector, pad_size)
        return output_vector

    def log_network(self, model: nn.Module, net_key: str, tensorboard_writer: SummaryWriter) -> None:
        """Logs a whole network module state into tensorboard via histogram

        Inherited from EMBER-2
        """
        for name, value in model.named_parameters():
            if value.grad is not None:
                tensorboard_writer.add_histogram(
                    tag=f"Networks/{net_key}_grad/{name}", values=value.grad.cpu(), global_step=self.step
                )
            tensorboard_writer.add_histogram(tag=f"Networks/{net_key}_params/{name}", values=value.cpu(), global_step=self.step)

    def num_parameters_summary(self) -> None:
        """Write out the estimated number of parameters in each sub-model within the overall EMBER-2 network"""
        self.params_G = utils.summary(self.G)
        self.params_GE = utils.summary(self.GE)
        self.params_D = utils.summary(self.D)
        self.logger.info(f"Estimate of parameters in Generator G: {self.params_G}")
        self.logger.info(f"Estimate of parameters in Generator GE: {self.params_GE}")
        self.logger.info(f"Estimate of parameters in Discriminator: {self.params_D}")

    def log_checkpoint_image(
        self,
        epoch_num: int,
        input_tsr: torch.Tensor,
        target_tsr: torch.Tensor,
        generator_tsr_1: torch.Tensor,
        generator_tsr_2: torch.Tensor,
    ) -> None:
        """Make (and save) a checkpoint bookkeeping plot"""
        # NOTE: could look at gas density for target/output, using DM since its directly comparable to input
        # fig settings
        figwidth = 6.4
        figheight = 5.4
        colormap_to_use = "viridis"
        # coordinates to insert a colorbar on the right of the figure
        axes_inset = (1.02, 0, 0.06, 1.0)
        # kwargs for tight layout
        tight_layout_pad = 1.50
        tight_layout_rect = (0, 0.08, 1, 1)

        subvolume_first_coord = generator_tsr_1.shape[2] // 2
        # pick a slice halfway through the x axis depending on the length of the tensor actually being used

        # copy the tensors to cpu for plotting
        input_tsr = input_tsr.cpu()
        target_tsr = target_tsr.cpu()
        generator_tsr_1 = generator_tsr_1.cpu()
        generator_tsr_2 = generator_tsr_2.cpu()

        target_labels_dict: dict[str, torch.Tensor] = {
            "Pure DM input": input_tsr[0, 0, :, :, :],
            "Mixed DM target": target_tsr[0, 0, :, :, :],
        }
        output_labels_dict: dict[str, torch.Tensor] = {
            "Mixed DM Generated 1": generator_tsr_1[0, 0, :, :, :],
            "Mixed DM Generated 2": generator_tsr_2[0, 0, :, :, :],
        }

        target_minmax_dict: dict[str, float] = {}
        output_minmax_dict: dict[str, float] = {}
        for key, tensor in target_labels_dict.items():
            target_minmax_dict[key + " min"] = torch.min(tensor).item()
            target_minmax_dict[key + " max"] = torch.max(tensor).item()
        for key, tensor in output_labels_dict.items():
            output_minmax_dict[key + " min"] = torch.min(tensor).item()
            output_minmax_dict[key + " max"] = torch.max(tensor).item()

        fig, ax = plt.subplots(ncols=2, nrows=2, figsize=(figwidth * 2, figheight * 2))
        # input + target
        imgs_target: list[plt.AxesImage] = []
        for i, (key, vector) in enumerate(target_labels_dict.items()):
            imgs_target.append(
                ax[0, i].imshow(
                    vector[subvolume_first_coord, :, :],
                    cmap=colormap_to_use,
                    norm=mpl.colors.Normalize(
                        vmin=target_minmax_dict[key + " min"],
                        vmax=target_minmax_dict[key + " max"],
                    ),
                )
            )
            ax[0, i].set_title(key)
            cax = ax[0, i].inset_axes(axes_inset)
            fig.colorbar(imgs_target[i], cax=cax, orientation="vertical", extend="neither")
        # output
        imgs_output: list[plt.AxesImage] = []
        for i, (key, vector) in enumerate(output_labels_dict.items()):
            imgs_output.append(
                ax[1, i].imshow(
                    vector[subvolume_first_coord, :, :],
                    cmap=colormap_to_use,
                    norm=mpl.colors.Normalize(
                        vmin=output_minmax_dict[key + " min"],
                        vmax=output_minmax_dict[key + " max"],
                    ),
                )
            )
            ax[1, i].set_title(key)
            cax = ax[1, i].inset_axes(axes_inset)
            fig.colorbar(imgs_output[i], cax=cax, orientation="vertical", extend="neither")

        # final fig settings
        fig.suptitle(f"2D slices of DM fields during training\nEpoch={epoch_num}, slice at x={subvolume_first_coord}")
        fig.tight_layout(pad=tight_layout_pad, rect=tight_layout_rect)

        fig.savefig(self.figure_dir / self.training_fields_name / f"training_fields_epoch_{epoch_num}.png")
        if self.tensorboard_writer is not None:
            self.tensorboard_writer.add_figure("Figures/training_fields", fig, close=True, global_step=epoch_num)
        else:
            plt.close(fig)

        self.logger.debug(f"Saved checkpoint training image for epoch {epoch_num}.")

        return None
