# ---
# jupyter:
#   jupytext:
#     formats: ipynb,py:percent
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.18.1
#   kernelspec:
#     display_name: .venv
#     language: python
#     name: python3
# ---

# %%
"""Test notebook to make training code run"""

# %%
import os

"""
NotImplementedError: The operator 'aten::upsample_trilinear3d.out' is not currently implemented for the MPS device.
If you want this op to be considered for addition please comment on https://github.com/pytorch/pytorch/issues/141287 and mention use-case,
that resulted in missing op as well as commit hash e2d141dbde55c2a4370fac5165b0561b6af4798b.
As a temporary fix, you can set the environment variable `PYTORCH_ENABLE_MPS_FALLBACK=1` to use the CPU as a fallback for this op.
WARNING: this will be slower than running natively on MPS.
"""
# we need this so...
os.environ["PYTORCH_ENABLE_MPS_FALLBACK"] = "1"
# must be set before import torch

# %%
from pathlib import Path
from types import FunctionType

import matplotlib.pyplot as plt
import numpy as np
import torch
from torch import nn
from torch.optim import Adam, AdamW
from torch.utils.data import DataLoader
from tqdm.auto import tqdm, trange

# %%
print(plt.get_backend())

# %%
# from ember2.modules import Discriminator, Generator
from cs_5960_ast.ember2.modules import Discriminator, Generator

# from .data import DistFieldSampler, FieldDataset
from cs_5960_ast.map2map.src.data import DistFieldSampler, FieldDataset
from cs_5960_ast.map2map.src.data.norms import ember2_norms
from cs_5960_ast.methods.Pk import Pk

# from ember2.net import Ember2Net
from cs_5960_ast.methods.utils import NSLoss, pair_base

# %%
print(plt.get_backend())

# %%
# Source - https://stackoverflow.com/a
# Posted by Karol Zlot, modified by community. See post 'Timeline' for change history
# Retrieved 2025-12-03, License - CC BY-SA 4.0

from time import sleep

import psutil
from tqdm.notebook import tqdm as notebook_tqdm

i = 0
with notebook_tqdm(total=100, desc="cpu%", position=1) as cpubar, notebook_tqdm(total=100, desc="ram%", position=0) as rambar:
    while True:
        rambar.n = psutil.virtual_memory().percent
        cpubar.n = psutil.cpu_percent()

        cpubar.refresh()
        rambar.refresh()

        sleep(0.5)
        i += 1
        if i >= 10:
            break


# %%
def check_normalization(
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
        f"zero values: {array_tensor.numel() - array_tensor.count_nonzero():_}; non-zero values: {array_tensor.count_nonzero():_}\
; total values: {array_tensor.numel():_}".replace("_", " ")
    )
    if print_tensor:
        print(array_tensor[0, 0, 0, :, :])

    plt.show()
    return None


# %%
input_dm_target_numpy = np.load(
    Path("/mn/stornext/d13/euclid_nobackup/dennisfr/master-project/data/train_smudgy/DMO-1-1.npy").resolve(), mmap_mode="r"
)[None]
print(f"Input DM target numpy shape: {input_dm_target_numpy.shape}")

subvolume_first_coord = 10
input_dm_target_field = torch.from_numpy(
    input_dm_target_numpy[:, :, subvolume_first_coord : subvolume_first_coord + 1, :, :].astype(np.float32)
)
print(f"Input DM target tensor shape: {input_dm_target_field.shape}")

# %%
T_file = Path("/mn/stornext/d13/euclid_nobackup/dennisfr/master-project/data/train_smudgy/T-1-33.npy")
T_array = np.load(T_file)[None]
style = np.load("/mn/stornext/d13/euclid_nobackup/dennisfr/master-project/data/train_smudgy/style-1-33.npy")
print(style)
# print(T_array)
print(f"{T_array.size:_}".replace("_", " "))
print(T_array.shape)

# %%
# check normalizing function with sinh norm
check_normalization(
    T_array,
    ember2_norms.T_norm,
    a=0.14285714,
    exponent=-0.5,
    m_t_mult=-0.093,
    m_t_add=5.346,
    print_tensor=False,
    plot_title="T array",
)

# %%
DM_only_file = Path("/mn/stornext/d13/euclid_nobackup/dennisfr/master-project/data/train_smudgy/DMO-100-33.npy")
DM_only_array = np.load(DM_only_file)[None]
# print(DM_only_array)
print(DM_only_array.size)
print(DM_only_array.shape)

# %%
check_normalization(
    DM_only_array,
    ember2_norms.log_norm,
    print_tensor=False,
    plot_title="DM only array log norm",
)

# %%
# check normalizing function with sinh norm
check_normalization(
    DM_only_array,
    ember2_norms.dm_in_norm,
    eps=1e-7,
    exponent=+0.7,
    log_mult=1.0 / 2.5,
    print_tensor=False,
    plot_title="DM only array",
)

# %%
DM_file = Path("/mn/stornext/d13/euclid_nobackup/dennisfr/master-project/data/train_smudgy/DM-1-33.npy")
DM_array = np.load(DM_file)[None]
# print(DM_array)
print(DM_array.size)
print(DM_array.shape)

# %%
# check normalizing function with sinh norm
check_normalization(
    DM_array,
    ember2_norms.dm_norm,
    exponent=+0.7,
    log_mult=1.0 / 2.5,
    print_tensor=False,
    plot_title="DM array",
)

# %%
E_file = Path("/mn/stornext/d13/euclid_nobackup/dennisfr/master-project/data/train_smudgy/E-1-33.npy")
E_array = np.load(E_file)[None]
# print(E_array)
print(E_array.size)
print(E_array.shape)

# %%
# check normalizing function with sinh norm
check_normalization(
    E_array,
    ember2_norms.E_norm,
    a=1,
    exponent=-3,
    m_t_mult=+3.193,
    m_t_add=5.346,
    print_tensor=False,
    plot_title="E array",
)

# %%
gas_file: Path = Path("/mn/stornext/d13/euclid_nobackup/dennisfr/master-project/data/train_smudgy/gas-1-33.npy")
gas_array: np.ndarray = np.load(gas_file)[None]
# print(gas_array)
print(gas_array.size)
print(gas_array.shape)

# %%
# check normalizing function with generic norm
check_normalization(
    gas_array,
    ember2_norms.generic_norm,
    a=1,
    k=1.0,
    x_0=2.0,
    q=1.6,
    plot_title="Gas array generic norm",
)

# %%
# check normalizing function with sinh norm
check_normalization(
    gas_array,
    ember2_norms.gas_norm,
    exponent=+1.3,
    log_mult=+1.0 / 1.3,
    print_tensor=False,
    plot_title="Gas array specialized norm",
)

# %%
target_file = Path("../../../data/target_array.npy")
target_data = np.concat((DM_array, T_array, E_array, gas_array), axis=1)
print(target_data.shape)
# np.save(target_file, target_data)

# %%
train_in_file = Path("../../../data/train_in_array.npy")
train_in_array = np.reshape(DM_only_array, (1, 1, 256, 256, 256))
print(train_in_array.shape)
# np.save(train_in_file, train_in_array)

# %%
style_file = Path("/mn/stornext/d13/euclid_nobackup/dennisfr/master-project/data/train_smudgy/style-1-1.npy")
rng = np.random.default_rng()
random_style = rng.normal(size=(7))
print(f"Random style vector shape: {random_style.shape}")
# np.save(style_file, random_style)

check = np.load(style_file)
print(f"Loaded style vector shape: {check.shape}")

# %%
norm_config: dict[str, dict[str, float]] = {
    "ember2_norms.dm_in_norm": {
        "eps": 1e-8,
        "exponent": -0.1,
        "log_mult": 1.0 / 1.5,
    },
    "ember2_norms.dm_norm": {
        "eps": 1e-8,
        "exponent": -0.1,
        "log_mult": 1.0 / 1.5,
    },
    "ember2_norms.E_norm": {
        "exponent": -2,
        "m_t_mult": -0.193,
        "m_t_add": 5.346,
    },
    "ember2_norms.gas_norm": {
        "exponent": -0.1,
        "log_mult": 1.0 / 1.5,
    },
    "ember2_norms.T_norm": {
        "exponent": -0.5,
        "m_t_mult": -0.193,
        "m_t_add": 5.346,
    },
}

# %%
use_cuda = torch.cuda.is_available()
use_mps = torch.backends.mps.is_available()
if use_cuda:
    device = torch.device("cuda")
elif use_mps:
    device = torch.device("mps")
else:
    device = torch.device("cpu")
print(device)

# %%
my_device = torch.accelerator.current_accelerator() if torch.accelerator.is_available() else torch.device("cpu")
print(f"Device: {my_device}")
if torch.accelerator.is_available():
    print(torch.accelerator.current_accelerator())
else:
    print("CPU only")


# %%
random_style = torch.randn(1, 6)
print(f"Style vector: {random_style}")
print(f"Style vector shape: {random_style.shape}")
# random_style = random_style.to(device)

# input_vector = torch.randn(2, 1, 256, 256, 256)
input_vector = torch.tensor(train_in_array)
print(f"Input shape: {input_vector.shape}")
# input_vector = input_vector.to(device)

# target_vector = torch.randn(2, 4, 256, 256, 256)
target_vector = torch.tensor(target_data)
print(f"Target shape: {target_vector.shape}")

# %%
# minimalist dataset for the random inputs
random_train_data = torch.utils.data.TensorDataset(input_vector, random_style, target_vector)

# dataloader with input vector and style vector
random_dataloader = DataLoader(dataset=random_train_data, batch_size=1, shuffle=True, num_workers=os.process_cpu_count())

for data in random_dataloader:
    input_vector = data[0]
    print(input_vector.shape)
    random_style = data[1]
    print(random_style.shape)
    random_target = data[2]
    print(random_target.shape)

# %% [markdown]
# ## define map2map dataloader
#

# %%
data_location: Path = Path("/mn/stornext/d13/euclid_nobackup/dennisfr/master-project/data/train_smudgy/").resolve()
# input_train_file: Path = data_location / "train_in_array.npy"
# target_train_file: Path = data_location / "target_array.npy"
style_train_file: Path = Path("/mn/stornext/d13/euclid_nobackup/dennisfr/master-project/data/train_smudgy/") / "style-1-1.npy"

# input_train_patterns = [input_train_file.as_posix()]
input_train_patterns = [(data_location / "DMO-1-1.npy").as_posix()]
target_train_patterns = [
    (data_location / "DM-1-1.npy").as_posix(),
    (data_location / "E-1-1.npy").as_posix(),
    (data_location / "gas-1-1.npy").as_posix(),
    (data_location / "T-1-1.npy").as_posix(),
]
style_train_pattern = style_train_file.as_posix()
# print(input_train_patterns)
train_dataset = FieldDataset(
    in_patterns=input_train_patterns,
    tgt_patterns=target_train_patterns,
    style_pattern=style_train_pattern,
    in_norms=["ember2_norms.dm_in_norm"],
    tgt_norms=["ember2_norms.dm_norm", "ember2_norms.E_norm", "ember2_norms.gas_norm", "ember2_norms.T_norm"],
    norm_config=norm_config,
    callback_at=None,
    augment=None,
    aug_shift=None,
    aug_add=None,
    aug_mul=None,
    crop=64,
    crop_start=100,
    crop_stop=164,
    crop_step=None,
    in_pad=2,
    tgt_pad=2,
    scale_factor=1,
    # **args.misc_kwargs,
)
train_sampler = DistFieldSampler(train_dataset, shuffle=True, div_data=False, div_shuffle_dist=1)

train_loader = DataLoader(
    train_dataset,
    batch_size=1,
    shuffle=False,
    sampler=train_sampler,
    # num_workers=os.process_cpu_count(),
    num_workers=1,  # paralell processing creates a ton of extra tensors, bad for laptop
    pin_memory=True,
)

in_chan = train_dataset.in_chan
out_chan = train_dataset.tgt_chan
style_size = train_dataset.style_size

print(f"Input channels: {in_chan}")
print(f"Output channels: {out_chan}")
print(f"Style size: {style_size}")

# %%
# before norm and before dataloader
fig, ax = plt.subplots(ncols=5, figsize=(16, 4.5))
subvolume_first_coord = 0
# in
img0 = ax[0].imshow(DM_only_array[0, 0, subvolume_first_coord, :, :])
ax[0].set_title("DM in")
fig.colorbar(img0, ax=ax[0], location="bottom")
# tgt
img1 = ax[1].imshow(DM_array[0, 0, subvolume_first_coord, :, :])
ax[1].set_title("DM out")
fig.colorbar(img1, ax=ax[1], location="bottom")
img2 = ax[2].imshow(E_array[0, 0, subvolume_first_coord, :, :])
ax[2].set_title("E")
fig.colorbar(img2, ax=ax[2], location="bottom")
img3 = ax[3].imshow(gas_array[0, 0, subvolume_first_coord, :, :])
ax[3].set_title("gas")
fig.colorbar(img3, ax=ax[3], location="bottom")
img4 = ax[4].imshow(T_array[0, 0, subvolume_first_coord, :, :])
ax[4].set_title("T")
fig.colorbar(img4, ax=ax[4], location="bottom")
plt.show()

# %%
# check what map2map dataloader returns
train_sampler.set_epoch(0)
print(f"Number of batches: {len(train_loader)}")
for i, data in enumerate(train_loader):
    print(f"batch {i}")

    print(data.keys())

    input_vector = data["input"]
    print(f"input vector: {input_vector.shape}")
    random_style = data["style"]
    print(f"style: {random_style.shape}")
    print(f"s={random_style}")
    random_target = data["target"]
    print(f"target vector: {random_target.shape}")

    fig, ax = plt.subplots(ncols=5, figsize=(16, 4.5))
    subvolume_first_coord = 0
    # in
    img0 = ax[0].imshow(input_vector[0, 0, subvolume_first_coord, :, :])
    ax[0].set_title("DM in")
    fig.colorbar(img0, ax=ax[0], location="bottom")
    # out
    img1 = ax[1].imshow(random_target[0, 0, subvolume_first_coord, :, :])
    ax[1].set_title("DM out normed")
    fig.colorbar(img1, ax=ax[1], location="bottom")
    img2 = ax[2].imshow(random_target[0, 1, subvolume_first_coord, :, :])
    ax[2].set_title("E")
    fig.colorbar(img2, ax=ax[2], location="bottom")
    img3 = ax[3].imshow(random_target[0, 2, subvolume_first_coord, :, :])
    ax[3].set_title("gas")
    fig.colorbar(img3, ax=ax[3], location="bottom")
    img4 = ax[4].imshow(random_target[0, 3, subvolume_first_coord, :, :])
    ax[4].set_title("T")
    fig.colorbar(img4, ax=ax[4], location="bottom")

    plt.show()

    del input_vector
    del random_style
    del random_target

    break

# %%
BoxSize = 100.0
Pk_inp = Pk(DM_only_array[0, 0, :, :, :], BoxSize, axis=1, threads=4, MAS="None", verbose=True)
k_inp = Pk_inp.k3D
Pk_inp = Pk_inp.Pk[:, 0]

fig, ax = plt.subplots(ncols=1, figsize=(10, 5))
ax.loglog(k_inp, Pk_inp)

# %% [markdown]
# ## unet
#

# %%
import itertools
import math
from math import log2

import numpy as np
import torch
import torch.nn.functional as F
from scipy.special import hyp2f1


def pixel_shuffle_3d_inv(x, r):
    """Rearranges tensor x with shape ``[B,C,H,W,D]`` to a tensor of shape ``[B,C*r*r*r,H/r,W/r,D/r]``."""
    [B, C, H, W, D] = list(x.size())
    x = x.contiguous().view(B, C, H // r, r, W // r, r, D // r, r)
    x = x.permute(0, 1, 3, 5, 7, 2, 4, 6)
    x = x.contiguous().view(B, C * (r**3), H // r, W // r, D // r)
    return x


def lag2eul(
    dis,
    val=1.0,
    eul_scale_factor=2,
    eul_pad=0,
    rm_dis_mean=True,
    periodic=False,
    a=0.3333,
    dis_std=6.0,
    boxsize=100.0,
    meshsize=512,
    **kwargs,
):
    """Transform fields from Lagrangian description to Eulerian description

    Only works for 3d fields, output same mesh size as input.

    Use displacement fields `dis` to map the value fields `val` from Lagrangian
    to Eulerian positions and then "paint" with CIC (trilinear) scheme.
    Displacement and value fields are paired when are sequences of the same
    length. If the displacement (value) field has only one entry, it is shared
    by all the value (displacement) fields.

    The Eulerian size is scaled by the `eul_scale_factor` and then padded by
    the `eul_pad`.

    Common mean displacement of all inputs can be removed to bring more
    particles inside the box. Periodic boundary condition can be turned on.

    Note that the box and mesh sizes don't have to be that of the inputs, as
    long as their ratio gives the right resolution. One can therefore set them
    to the values of the whole Lagrangian fields, and use smaller inputs.

    Implementation follows pmesh/cic.py by Yu Feng.
    """
    # NOTE the following factor assumes the displacements have been normalized
    # by data.norms.cosmology.dis, and thus undoes it
    z = 1 / a - 1

    dis_norm = dis_std * D(z) * meshsize / boxsize  # to mesh unit
    dis_norm *= eul_scale_factor

    if isinstance(dis, torch.Tensor):
        dis = [dis]
    if isinstance(val, (float, torch.Tensor)):
        val = [val]
    if len(dis) != len(val) and len(dis) != 1 and len(val) != 1:
        msg = "dis-val field mismatch"
        raise ValueError(msg)

    if any(d.dim() != 5 for d in dis):
        msg = "only support 3d fields for now"
        raise NotImplementedError(msg)
    if any(d.shape[1] != 3 for d in dis):
        msg = "only support 3d displacement fields"
        raise ValueError(msg)

    # common mean displacement of all inputs
    # if removed, fewer particles go outside of the box
    # common for all inputs so outputs are comparable in the same coords
    d_mean = 0
    if rm_dis_mean:
        d_mean = sum(d.detach().mean((2, 3, 4), keepdim=True) for d in dis) / len(dis)

    out = []
    if len(dis) == 1 and len(val) != 1:
        dis = itertools.repeat(dis[0])
    elif len(dis) != 1 and len(val) == 1:
        val = itertools.repeat(val[0])
    for d, v in zip(dis, val):
        dtype, device = d.dtype, d.device

        N, DHW = d.shape[0], d.shape[2:]
        DHW = torch.Size([s * eul_scale_factor + 2 * eul_pad for s in DHW])

        if isinstance(v, float):
            C = 1
        else:
            C = v.shape[1]
            v = v.contiguous().flatten(start_dim=2).unsqueeze(-1)

        mesh = torch.zeros(N, C, *DHW, dtype=dtype, device=device)

        pos = (d - d_mean) * dis_norm
        del d

        pos[:, 0] += torch.arange(0.5, DHW[0] - 2 * eul_pad, eul_scale_factor, dtype=dtype, device=device)[:, None, None]
        pos[:, 1] += torch.arange(0.5, DHW[1] - 2 * eul_pad, eul_scale_factor, dtype=dtype, device=device)[:, None]
        pos[:, 2] += torch.arange(0.5, DHW[2] - 2 * eul_pad, eul_scale_factor, dtype=dtype, device=device)

        pos = pos.contiguous().view(N, 3, -1, 1)  # last axis for neighbors

        intpos = pos.floor().to(torch.int)
        neighbors = (torch.arange(8, device=device) >> torch.arange(3, device=device)[:, None, None]) & 1
        tgtpos = intpos + neighbors
        del intpos, neighbors

        # CIC
        kernel = (1.0 - torch.abs(pos - tgtpos)).prod(1, keepdim=True)
        del pos

        v = v * kernel
        del kernel

        tgtpos = tgtpos.view(N, 3, -1)  # fuse spatial and neighbor axes
        v = v.view(N, C, -1)

        for n in range(N):  # because ind has variable length
            bounds = torch.tensor(DHW, device=device)[:, None]

            if periodic:
                torch.remainder(tgtpos[n], bounds, out=tgtpos[n])

            ind = (tgtpos[n, 0] * DHW[1] + tgtpos[n, 1]) * DHW[2] + tgtpos[n, 2]
            src = v[n]

            if not periodic:
                mask = ((tgtpos[n] >= 0) & (tgtpos[n] < bounds)).all(0)
                ind = ind[mask]
                src = src[:, mask]

            mesh[n].view(C, -1).index_add_(1, ind, src)

        if eul_scale_factor > 1:
            # print(mesh.shape,'before shuffle')
            mesh = pixel_shuffle_3d_inv(mesh, eul_scale_factor)
            # print(mesh.shape,'after shuffle')

        out.append(mesh)

    return out


def cropfield(field, idx, reps, crop, pad):
    start = np.unravel_index(idx, reps) * crop
    x = field.copy()
    for d, (i, N, (p0, p1)) in enumerate(zip(start, crop, pad)):
        x = x.take(range(i - p0, i + N + p1), axis=1 + d, mode="wrap")
    return x


def narrow_like(SR_box, tgt_grid):
    width = np.shape(SR_box)[1] - tgt_grid
    half_width = width // 2
    begin, stop = half_width, tgt_grid + half_width
    return SR_box[:, begin:stop, begin:stop, begin:stop]


def D(z, Om=0.31):
    """linear growth function for flat LambdaCDM, normalized to 1 at redshift zero"""
    OL = 1 - Om
    a = 1 / (1 + z)
    return a * hyp2f1(1, 1 / 3, 11 / 6, -OL * a**3 / Om) / hyp2f1(1, 1 / 3, 11 / 6, -OL / Om)


def f(z, Om=0.31):
    """linear growth rate for flat LambdaCDM"""
    OL = 1 - Om
    a = 1 / (1 + z)
    aa3 = OL * a**3 / Om
    return 1 - 6 / 11 * aa3 * hyp2f1(2, 4 / 3, 17 / 6, -aa3) / hyp2f1(1, 1 / 3, 11 / 6, -aa3)


def H(z, Om=0.31):
    """Hubble in [h km/s/Mpc] for flat LambdaCDM"""
    OL = 1 - Om
    a = 1 / (1 + z)
    return 100 * np.sqrt(Om / a**3 + OL)


def dis(x, undo=False, a: float = 0.3333, dis_std=6.0, **kwargs):
    # WAS dis_std=6000.0
    z = 1 / a - 1
    dis_norm = dis_std * D(z)  # [Kpc/h]

    if not undo:
        dis_norm = 1 / dis_norm

    x *= dis_norm


def narrow_by(a, c):
    """Narrow a by size c symmetrically on all edges."""
    ind = (slice(None),) * 2 + (slice(c, -c),) * (a.dim() - 2)
    return a[ind]


class Resampler(nn.Module):
    """Resampling, upsampling or downsampling.

    By default discard the inaccurate edges when upsampling.
    """

    def __init__(self, ndim, scale_factor, narrow=True):
        super().__init__()

        modes = {1: "linear", 2: "bilinear", 3: "trilinear"}
        self.mode = modes[ndim]

        self.scale_factor = scale_factor
        self.narrow = narrow

    def forward(self, x):
        x = F.interpolate(x, scale_factor=self.scale_factor, mode=self.mode, align_corners=False)

        if self.scale_factor > 1 and self.narrow == True:
            edges = round(self.scale_factor) // 2
            edges = max(edges, 1)
            x = narrow_by(x, edges)

        return x


class LeakyReLUStyled(nn.LeakyReLU):
    def __init__(self, negative_slope=0.2, inplace=False):
        super().__init__(negative_slope, inplace)

    """ Trivially evaluates standard leaky ReLU, but accepts second argument

    for style array that is not used
    """

    def forward(self, x, style=None):
        return super().forward(x)


class ConvStyled3d(nn.Module):
    """Convolution layer with modulation and demodulation, from StyleGAN2.

    Weight and bias initialization from `torch.nn._ConvNd.reset_parameters()`.
    """

    def __init__(self, in_chan, out_chan, style_size, kernel_size=3, stride=1, bias=True, resample=None):
        super().__init__()

        # self.style_weight = nn.Parameter(torch.empty(in_chan, style_size))
        # nn.init.kaiming_uniform_(self.style_weight, a=math.sqrt(5),
        #                          mode='fan_in', nonlinearity='leaky_relu')
        # self.style_bias = nn.Parameter(torch.ones(in_chan))  # NOTE: init to 1

        if resample is None:
            K3 = (kernel_size,) * 3
            self.weight = nn.Parameter(torch.empty(out_chan, in_chan, *K3))
            self.stride = stride
            self.conv = F.conv3d
        elif resample == "U":
            K3 = (kernel_size,) * 3
            # NOTE not clear to me why convtranspose have channels swapped
            self.weight = nn.Parameter(torch.empty(in_chan, out_chan, *K3))
            self.stride = stride
            self.conv = F.conv_transpose3d
        elif resample == "D":
            K3 = (2,) * 3
            self.weight = nn.Parameter(torch.empty(out_chan, in_chan, *K3))
            self.stride = 2
            self.conv = F.conv3d
        else:
            msg = f"resample type {resample} not supported"
            raise ValueError(msg)
        self.resample = resample

        nn.init.kaiming_uniform_(
            self.weight,
            a=math.sqrt(5),
            mode="fan_in",  # effectively 'fan_out' for 'D'
            nonlinearity="leaky_relu",
        )

        if bias:
            self.bias = nn.Parameter(torch.zeros(out_chan))
            fan_in, _ = nn.init._calculate_fan_in_and_fan_out(self.weight)
            bound = 1 / math.sqrt(fan_in)
            nn.init.uniform_(self.bias, -bound, bound)
        else:
            self.register_parameter("bias", None)

        def init_weight(m):
            if type(m) is nn.Linear:
                torch.nn.init.kaiming_uniform_(m.weight, a=math.sqrt(5), mode="fan_in", nonlinearity="leaky_relu")
                if m.bias is not None:
                    torch.nn.init.ones_(m.bias)

        self.style_block = nn.Sequential(
            nn.Linear(in_features=style_size, out_features=in_chan),
        )
        self.style_block.apply(init_weight)

    def forward(self, inputs):
        x, s = inputs[0], inputs[1]
        eps = 1e-8

        N, Cin, *DHWin = x.shape

        C0, C1, *K3 = self.weight.shape

        if self.resample == "U":
            Cin, Cout = C0, C1
        else:
            Cout, Cin = C0, C1

        # s = F.linear(s, self.style_weight, bias=self.style_bias)
        s = self.style_block(s)
        # modulation
        if self.resample == "U":
            s = s.reshape(N, Cin, 1, 1, 1, 1)
        else:
            s = s.reshape(N, 1, Cin, 1, 1, 1)
        w = self.weight * s
        # print(Cin, 'Cin2')
        # demodulation
        if self.resample == "U":
            fan_in_dim = (1, 3, 4, 5)
        else:
            fan_in_dim = (2, 3, 4, 5)
        w = w * torch.rsqrt(w.pow(2).sum(dim=fan_in_dim, keepdim=True) + eps)

        w = w.reshape(N * C0, C1, *K3)
        # print(N, Cin, *DHWin)
        x = x.reshape(1, N * Cin, *DHWin)
        # END HERE
        x = self.conv(x, w, bias=self.bias, stride=self.stride, groups=N)
        _, _, *DHWout = x.shape
        # print('N', N, 'Cout', Cout, 'DHWout', *DHWout)
        # x = x.reshape(N, Cout, *DHWout)
        # print(x.shape)
        x = x.view(N, Cout, *DHWout)

        return x


class G(nn.Module):
    def __init__(
        self, in_chan, out_chan, style_size, scale_factor=16, chan_base=512, chan_min=64, chan_max=512, cat_noise=True, **kwargs
    ):
        super().__init__()

        self.style_size = style_size
        self.scale_factor = scale_factor
        num_blocks = round(log2(self.scale_factor))

        assert chan_min <= chan_max

        def chan(b):
            c = chan_base >> b
            c = max(c, chan_min)
            c = min(c, chan_max)
            return c

        self.block0 = nn.Sequential(
            ConvStyled3d(in_chan, chan(0), self.style_size, 1),
            LeakyReLUStyled(0.2, True),
        )

        self.blocks = nn.ModuleList()
        for b in range(num_blocks):
            prev_chan, next_chan = chan(b), chan(b + 1)
            self.blocks.append(HBlock(prev_chan, next_chan, out_chan, cat_noise, style_size))

    def forward(self, x, s):
        y = x  # direct upsampling from the input

        print(x.shape)
        x = self.block0((x, s))
        print(x.shape)
        # y = None  # no direct upsampling from the input

        for block in self.blocks:
            x, y, s = block(x, y, s)

        return y


class HBlock(nn.Module):
    """The "H" block of the StyleGAN2 generator.

        x_p                     y_p
         |                       |
    convolution           linear upsample
         |                       |
          >--- projection ------>+
         |                       |
         v                       v
        x_n                     y_n

    See Fig. 7 (b) upper in https://arxiv.org/abs/1912.04958
    Upsampling are all linear, not transposed convolution.

    Parameters
    ----------
    prev_chan : number of channels of x_p
    next_chan : number of channels of x_n
    out_chan : number of channels of y_p and y_n
    cat_noise: concatenate noise if True, otherwise add noise

    Notes
    -----
    next_size = 2 * prev_size - 6
    """

    def __init__(self, prev_chan, next_chan, out_chan, cat_noise, style_size):
        super().__init__()

        self.upsample = Resampler(3, 2)

        # isolate conv style part to make input right
        self.addnoise0 = AddNoise(cat_noise, chan=prev_chan, style_size=style_size)

        # self.noise_upsample = nn.Sequential(
        #    AddNoise(cat_noise, chan=prev_chan, style_size=style_size),
        #    self.upsample,
        # )

        self.conv = nn.Sequential(
            ConvStyled3d(prev_chan + int(cat_noise), next_chan, style_size, 3),
            LeakyReLUStyled(0.2, True),
        )
        self.addnoise = AddNoise(cat_noise, chan=next_chan, style_size=style_size)

        self.conv1 = nn.Sequential(
            ConvStyled3d(next_chan + int(cat_noise), next_chan, style_size, 3),
            LeakyReLUStyled(0.2, True),
        )

        self.proj = nn.Sequential(
            # ConvStyled3d(next_chan + int(cat_noise), out_chan, style_size, 1),
            ConvStyled3d(next_chan, out_chan, style_size, 1),
            LeakyReLUStyled(0.2, True),
        )

    def forward(self, x, y, s):
        # x = self.noise_upsample(x, s)
        x = self.addnoise0(x, s)
        print(x.shape)
        x = self.upsample(x)
        print(x.shape)
        x = self.conv((x, s))
        print(x.shape)
        x = self.addnoise(x, s)
        print(x.shape)
        x = self.conv1((x, s))
        print(x.shape)

        if y is None:
            y = self.proj((x, s))
        else:
            y = self.upsample(y)

            y = narrow_by(y, 2)
            y = y + self.proj((x, s))
        return x, y, s


class StyleNoise(nn.Module):
    def __init__(self, in_chan=1, style_size=1, chans=[32, 64, 32]):
        super().__init__()

        self.act = nn.Tanh()

        self.nn = nn.Sequential(nn.Linear(in_chan + style_size, chans[0]), self.act)

        for i in range(len(chans) - 1):
            self.nn.append(nn.Linear(chans[i], chans[i + 1]))
            self.nn.append(self.act)

        self.nn.append(nn.Linear(chans[-1], 1))

    def forward(self, x, s):
        shape = x.shape
        x = x.flatten()
        x = torch.cat([x.unsqueeze(1), s.repeat(x.shape[0], 1)], dim=1)
        y = self.nn(x)
        return y.reshape(shape)


class AddNoise(nn.Module):
    """Add or concatenate noise.

    Add noise if `cat=False`.
    The number of channels `chan` should be 1 (StyleGAN2)
    or that of the input (StyleGAN).
    """

    def __init__(self, cat, chan=1, style_size=1):
        super().__init__()

        self.cat = cat
        self.nn = StyleNoise(in_chan=1, style_size=style_size)

        if not self.cat:
            self.std = nn.Parameter(torch.zeros([chan]))

    def forward(self, x, s):
        noise = self.nn(torch.randn_like(x[:, :1]), s)

        if self.cat:
            x = torch.cat([x, noise], dim=1)
        else:
            std_shape = (-1,) + (1,) * (x.dim() - 2)
            noise = self.std.view(std_shape) * noise

            x = x + noise

        return x


# %%
import torch
import torch.nn as nn


def narrow_like_(a, b):
    """Narrow a to be like b.

    Try to be symmetric but cut more on the right for odd difference
    """
    for d in range(2, a.dim()):
        width = a.shape[d] - b.shape[d]
        half_width = width // 2
        a = a.narrow(d, half_width, a.shape[d] - width)
    return a


class AddNoise(nn.Module):
    """Add or concatenate noise.

    Add noise if `cat=False`.
    The number of channels `chan` should be 1 (StyleGAN2)
    or that of the input (StyleGAN).
    """

    def __init__(self, cat, chan=1, style_size=1):
        super().__init__()

        self.cat = cat
        # self.nn = StyleNoise(in_chan=1, style_size=style_size)

        if not self.cat:
            self.std = nn.Parameter(torch.zeros([chan]))

    def forward(self, x):
        noise = torch.randn_like(x[:, :1])  # self.nn(torch.randn_like(x[:, :1]), s)

        if self.cat:
            x = torch.cat([x, noise], dim=1)
        else:
            std_shape = (-1,) + (1,) * (x.dim() - 2)
            noise = self.std.view(std_shape) * noise

            x = x + noise

        return x


class ResNetBlock(nn.Module):
    def __init__(self, in_chan, mid_chan, out_chan, style_size, last_act=True, cat_noise=False):
        super().__init__()
        self.last_act = last_act
        self.style_size = style_size

        # ResBlock: addnoise, conv, act, addnoise, conv, act
        self.skip = ConvStyled3d(in_chan, out_chan, self.style_size, kernel_size=1)

        self.addnoise1 = AddNoise(cat_noise)
        self.conv1 = ConvStyled3d(in_chan + cat_noise, mid_chan, self.style_size, kernel_size=3)
        self.act1 = LeakyReLUStyled(0.2, True)
        self.addnoise2 = AddNoise(cat_noise)
        self.conv2 = ConvStyled3d(mid_chan + cat_noise, out_chan, self.style_size, kernel_size=3)
        self.act2 = LeakyReLUStyled(0.2, True)

        self.act3 = LeakyReLUStyled(0.2, True)

    def forward(self, x, s):
        y = self.skip((x, s))

        x = F.pad(
            x,
            (
                1,
                1,
                1,
                1,
                1,
                1,
            ),
            mode="reflect",
        )
        x = self.addnoise1(x)
        x = self.conv1((x, s))
        x = self.act1(x)

        x = F.pad(
            x,
            (
                1,
                1,
                1,
                1,
                1,
                1,
            ),
            mode="reflect",
        )
        x = self.addnoise2(x)
        x = self.conv2((x, s))
        x = self.act2(x)

        x = x + narrow_like_(y, x)
        if self.last_act:
            x = self.act3(x)
        return x


class UpBlock(nn.Module):
    def __init__(self, in_chan, style_size, cat_noise=False):
        super().__init__()
        self.addnoise1 = AddNoise(cat_noise)
        # self.up = ConvStyled3d(in_chan+cat_noise, in_chan, style_size=style_size, kernel_size=10, stride=2, resample='U')
        self.conv1 = ConvStyled3d(in_chan, in_chan, style_size, kernel_size=3)
        self.act1 = LeakyReLUStyled(0.2, True)
        self.addnoise2 = AddNoise(cat_noise)
        self.conv2 = ConvStyled3d(in_chan + cat_noise, in_chan, style_size, kernel_size=3)
        self.act2 = LeakyReLUStyled(0.2, True)

    def forward(self, x, s):
        x = F.pad(
            x,
            (
                1,
                1,
                1,
                1,
                1,
                1,
            ),
            mode="reflect",
        )
        x = self.addnoise1(x)
        # x = self.up((x,s))
        x = F.interpolate(x, scale_factor=2, mode="trilinear", align_corners=False)

        x = self.conv1((x, s))
        x = self.act1(x)
        x = self.addnoise2(x)
        x = self.conv2((x, s))
        x = self.act2(x)
        return x


class DownBlock(nn.Module):
    def __init__(self, in_chan, style_size, cat_noise=False):
        super().__init__()
        self.addnoise = AddNoise(cat_noise)
        self.conv = ConvStyled3d(in_chan + cat_noise, in_chan, style_size, kernel_size=2, stride=2)
        self.act = LeakyReLUStyled(0.2, True)

    def forward(self, x, s):
        x = self.addnoise(x)
        x = self.conv((x, s))
        x = self.act(x)
        return x


class UNetG(nn.Module):
    def __init__(
        self, in_chan, out_chan, style_size, scale_factor=16, chan_base=512, chan_min=16, chan_max=512, cat_noise=True, **kwargs
    ):
        super().__init__()
        # https://github.com/dschaurecker/dl_halo/blob/main/train_test/map2map/models/generator.py

        # self.up0 = ConvStyled3d(in_chan, in_chan, style_size=style_size, kernel_size=2, stride=2, resample='U')

        self.resblock1 = ResNetBlock(in_chan, 64, 64, style_size)  # +2
        self.down1 = DownBlock(64, style_size)

        self.resblock2 = ResNetBlock(64, 128, 128, style_size)  # +4
        self.down2 = DownBlock(128, style_size)

        self.bottleneck = ResNetBlock(128, 256, 128, style_size)  # +8

        self.up1 = UpBlock(128, style_size)
        self.resblock3 = ResNetBlock(128 + 128, 128, 64, style_size)

        self.up2 = UpBlock(64, style_size)
        self.resblock4 = ResNetBlock(64 + 64, 64, out_chan, style_size, last_act=False)

        """
        self.conv_l0 = ResBlock_up(in_chan, 64, seq='CANCA')
        self.down_l0 = ConvBlock_up(64, seq='NDA')
        self.conv_l1 = ResBlock_up(64, 128, seq='NCANCA')
        self.down_l1 = ConvBlock_up(128, seq='NDA')

        self.conv_c = ResBlock_up(128, seq='NCANCA')

        self.up_r1 = ConvBlock_up(128, seq='NUXANXA')
        self.conv_r1 = ResBlock_up(256, 64, seq='NCANCA')
        self.up_r0 = ConvBlock_up(64, seq='NUXANXA')
        self.conv_r0 = ConvBlock_up(128, out_chan, seq='CAC')
        """

    def forward(self, x, s):
        # x = self.up0((x,s))
        # x = F.interpolate(x, scale_factor=2, mode="trilinear", align_corners=False)

        x1 = self.resblock1(x, s)  # +2
        x = self.down1(x1, s)

        x2 = self.resblock2(x, s)  # +4
        x = self.down2(x2, s)

        x = self.bottleneck(x, s)  # +8

        x = self.up1(x, s)
        x2 = narrow_like_(x2, x)
        x = torch.cat([x2, x], dim=1)
        x = self.resblock3(x, s)

        x = self.up2(x, s)
        x1 = narrow_like_(x1, x)
        x = torch.cat([x1, x], dim=1)
        x = self.resblock4(x, s)

        return x


# %%
N = 16
p = 3
x = torch.randn(1, 1, N + 2 * p, N + 2 * p, N + 2 * p)
s = torch.randn(1, 6)
m = UNetG(in_chan=1, out_chan=4, style_size=6)

y = m(x, s).detach().numpy()

print(y.shape)

fig, ax = plt.subplots(ncols=2, figsize=(10, 5))
ax[0].imshow(x[0, 0, 0])
ax[1].imshow(y[0, 0, 0])
plt.show()

fig, ax = plt.subplots()
hist, bins = np.histogram(x.flatten(), bins=100)
ax.plot(bins[:-1], hist, label="x")
hist, bins = np.histogram(y.flatten(), bins=100)
ax.plot(bins[:-1], hist, label="y")
ax.legend()
ax.grid()

# %%
p = 3
model_test_unet = UNetG(in_chan=1, out_chan=4, style_size=6)
for data in random_dataloader:
    input_vector = data[0][:, :, 0 : 16 + 2 * p, 0 : 16 + 2 * p, 0 : 16 + 2 * p]
    print(f"input: {input_vector.shape}")
    random_style = data[1]
    print(f"style: {random_style.shape}")

    y = model_test_unet(input_vector, random_style).detach().numpy()

    print(f"y: {y.shape}")

    fig, ax = plt.subplots(ncols=2, figsize=(10, 5))
    ax[0].imshow(input_vector[0, 0, 0])
    ax[1].imshow(y[0, 0, 0])
    plt.show()

    fig, ax = plt.subplots()
    hist, bins = np.histogram(input_vector.flatten(), bins=100)
    ax.plot(bins[:-1], hist, label="x")
    hist, bins = np.histogram(y.flatten(), bins=100)
    ax.plot(bins[:-1], hist, label="y")
    ax.legend()
    ax.grid()

# %% [markdown]
# ## ember2
#

# %%
net_config = {
    "inputs": [1, 32, 32, 32],
    "outputs": [4, 32, 32, 32],
    "context_dim": 6,
    "style_dim": 6,
    "style_depth": 4,
    "inp_channels": 1,
    "out_channels": 4,
    "nl": 3,
    "nf": 32,
    # "filters": [1, 64, 128, 64, 2],
    "filters": [16, 32, 64, 128],
    "ublock_shapes": [0, 0, 0],
    # "ublock_shapes": [256, 128, 128, 64, 1],
    "lr_g": 1e-4,
    "lr_d": 1e-4,
    # "noise": True
}


# %%
def filter_setup(num_layers, nf, nf_max=256):
    return [min(nf * 2**i, nf_max) for i in range(num_layers)]


# %%
filters = filter_setup(net_config["nl"], net_config["nf"])
print(filters)
net_config["filters"] = filters

# %%
p = 2
for data in random_dataloader:
    input_vector = data[0][:, :, 0 : 32 + 2 * p, 0 : 32 + 2 * p, 0 : 32 + 2 * p]
    print(input_vector.shape)
    random_style = data[1]
    print(random_style.shape)
    target_vector = data[2][:, :, 0 : 32 + 2 * p, 0 : 32 + 2 * p, 0 : 32 + 2 * p]
    print(target_vector.shape)

# %%
model = Generator(mode="base", config=net_config)
model = model.to(device)
print(f"input vector: {input_vector.shape}")
model(input_vector.to(device), random_style.to(device), noise=None)
print(type(model))


# %%
# model = SynthesisNet(mode="base", style_dim=6, inp_channels=1, out_channels=4, filters=[1, 64, 128, 64, 4])
# model(input_vector, random_style, noise=None)


# %%
class EMA:
    def __init__(self, beta):
        super().__init__()
        self.beta = beta

    def update_average(self, old, new):
        if old is None:
            return new
        return old * self.beta + (1 - self.beta) * new


# %%
class Ember2NetHack(nn.Module):
    """Hacked together minimalist version of ember2"""

    def __init__(self, config, mode) -> None:
        super().__init__()

        self.solve_env(config)

        self.mode = mode
        self.config = config

        self.G = Generator(mode, config)
        self.D = Discriminator(mode, **config)
        self.GE = Generator(mode, config)

        self.criterion = nn.MSELoss()
        # self.criterion = nn.HuberLoss(delta=0.1)

        self.ema = EMA(0.995)
        self.configure_optimizers()

        self.ckpt_freq = 10

    def forward(self, x, c, noise=None):
        return self.G(x, c, noise)

    def configure_optimizers(self) -> None:
        lr_g = self.config["lr_g"]
        lr_d = self.config["lr_d"]
        # self.opt_g = Adam(self.G.parameters(), lr=lr_g, betas=(0.5, 0.9))  # base: (0.0, 0.99)
        self.opt_g = AdamW(self.G.parameters(), lr=lr_g, betas=(0.5, 0.9))
        # self.opt_d = Adam(self.D.parameters(), lr=lr_d, betas=(0.5, 0.9))  # base: (0.0, 0.99)
        self.opt_d = AdamW(self.D.parameters(), lr=lr_d, betas=(0.5, 0.9))

    def EMA(self) -> None:
        def update_moving_average(current_model, ma_model) -> None:
            for current_params, ma_params in zip(current_model.parameters(), ma_model.parameters(), strict=True):
                old_weight, up_weight = ma_params.data, current_params.data
                ma_params.data = self.ema.update_average(old_weight, up_weight)

        update_moving_average(self.G, self.GE)

    def solve_env(self, cfg: dict) -> None:
        self.root_dir = (Path.cwd() / Path("../..")).resolve()
        print(self.root_dir)
        self.run_name = "testing_1"

        self.config_dir = self.root_dir / "models/configs/" / self.run_name
        self.run_dir = self.root_dir / "models/runs/" / self.run_name
        print(self.run_dir)
        Path.mkdir(self.run_dir, parents=True, exist_ok=True)  # check this dir exists

        use_cuda = torch.cuda.is_available()
        use_mps = torch.backends.mps.is_available()
        if use_cuda:
            self.device = torch.device("cuda")
        elif use_mps:
            self.device = torch.device("mps")
        else:
            self.device = torch.device("cpu")
        print(f"Device: {self.device}")

    def train(
        self,
        train_dataloader: DataLoader,
        # validation_dataloader: DataLoader,
        epochs: int,
        learning_rate: float = 2e-5,
        eps: float = 1e-8,
    ) -> list:
        """
        Trains a torch type model. TODO docs

        Expects a standard train, validation, test split for input data.
        Model should not receive "test" data during training.

        Uses AdamW for optimizing function. # TODO Adam or AdamW?

        Prints epoch information and progress bar to console.

        Parameters
        ----------
        train_dataloader : `Torch` DataLoader
            An iterable `Pytorch` DataLoader object with training data.
        validation_dataloader : `Torch` DataLoader
            An iterable `Pytorch` DataLoader object with validation data for model learning.
            Do *not* use "test" data for this.
        epochs : int
            Number of training epochs to use.
        learning_rate : float, default = 2e-5
            Learning rate to use for optimizer.
        eps : float, default = 1e-8
            Term added to the denominator to improve
            numerical stability in the optimizer.

        Returns
        -------
        Trained model. Will modify the input `model` object and thus returns nothing.
        """
        # optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate, eps=eps)
        loss = []
        break_point = 2

        # overwritten during training loop
        input_vector = None
        input_style = None
        target_vector = None
        p1 = None
        p2 = None

        # training loop
        for epoch_num in trange(epochs, desc="Epochs"):
            total_loss_train = 0
            batch_num = 0

            train_sampler.set_epoch(epoch_num)

            # tqdm creates a progress bar for stepping through the training data
            for data in tqdm(train_dataloader, desc="Training data progress"):
                # unpack data
                input_vector = data["input"].to(self.device)
                # print(f"Input vector: {input_vector.shape}")
                input_style = data["style"].to(self.device)
                # print(f"Input style: {input_style.shape}")
                target_vector = data["target"].to(self.device)
                # print(f"Target vector: {target_vector.shape}")

                self.opt_g.zero_grad(set_to_none=True)
                # toggle_grad(self.G, True)
                self.G.requires_grad_(requires_grad=True)
                # toggle_grad(self.D, False)
                self.D.requires_grad_(requires_grad=False)

                p1 = self.G(input_vector, input_style)
                p2 = self.G(input_vector, input_style)

                # print(f"input_vector: {input_vector.shape}")
                # print(f"input_style: {input_style.shape}")
                # print(f"p1: {p1.shape}")
                # print(f"p2: {p2.shape}")

                mse_error = self.criterion(p1, target_vector)
                mse_error.backward()
                self.opt_g.step()

                total_loss_train += mse_error.detach()
                loss.append(mse_error.detach().cpu())

                # TODO: ends loop early
                if batch_num >= break_point:
                    batch_num += 1  # not that it really matters but whatever
                    break
                else:
                    batch_num += 1
                    continue

                f = self.D(pair_base(input_vector, p1, p2, input_style))
                g_loss = NSLoss(None, f, mode="G")
                g_loss.backward()
                self.opt_g.step()

                # train D
                self.opt_d.zero_grad(set_to_none=True)
                # toggle_grad(self.G, False)
                self.G.requires_grad_(requires_grad=False)
                # toggle_grad(self.D, True)
                self.D.requires_grad_(requires_grad=True)

                p1 = p1.detach()
                p2 = p2.detach()

                r = self.D(pair_base(input_vector, y, p1, input_style))
                f = self.D(pair_base(input_vector, p1, p2, input_style))

                d_loss = NSLoss(r, f, mode="D")
                d_loss.backward()
                self.opt_d.step()

                # self.log(f"g_adv {g_loss}")
                # self.log(f"dfake {f}")
                # self.log(f"dtrue {r}")
                # self.log(f"d_adv {d_loss}")

                ## old
                # not scaling before passing to loss improves stability
                # output = model(input_ids=input_id, attention_mask=mask)

                # batch_loss = criterion(output, train_label.long())
                # item() turns a torch tensor to a simple python number object
                # total_loss_train += batch_loss.item()

                # model.zero_grad()
                # batch_loss.backward()
                # optimizer.step()
                ## old

            ## validation not now
            # now evaluate against the validation data
            # total_acc_validation = 0
            # total_loss_validation = 0
            # with torch.no_grad():
            #     for data in validation_dataloader:
            #         validation_label = data["label"]
            #         validation_input = data["encoded_text_tensor"]
            #         validation_label = validation_label.to(self.device)
            #         mask = validation_input["attention_mask"].to(self.device)
            #         input_id = validation_input["input_ids"].squeeze(1).to(self.device)

            #         output = model(input_ids=input_id, attention_mask=mask)

            #         batch_loss = criterion(output, validation_label.long())
            #         total_loss_validation += batch_loss.item()

            #         acc = (output.argmax(dim=1) == validation_label).sum().item()
            #         total_acc_validation += acc

            if epoch_num % self.ckpt_freq == 0:
                # torch.save(nn, self.run_dir + f"/{mode}/set_{set_nr}.ckpt_{step}.pt")
                print("checkpoint")

            # if self.step % 10 == 0 and self.step > 50000:
            #    self.EMA()

            # epoch information
            print(
                f"Epochs: {epoch_num + 1} \
                    | Train Loss: {total_loss_train / len(train_dataloader): .3e}"  # \
                # | Validation Loss: {total_loss_validation / len(validation_dataloader.dataset): .3e}"
            )

        del input_vector
        del input_style
        del target_vector
        del p1
        del p2

        return loss


# %%
model = Ember2NetHack(mode="base", config=net_config)
model = model.to(device)
# model(input_vector, random_style, noise=None)
loss = model.train(train_dataloader=train_loader, epochs=20)
print(type(model))

# %%
plt.plot(loss)
plt.show()

# %%
print(input_vector.shape)
output_vector = model.G(input_vector.to(device), random_style.to(device))
output_vector = output_vector.detach().cpu()
subvolume_first_coord = 10
fig, ax = plt.subplots(ncols=3)
ax[0].imshow(input_vector[0, 0, subvolume_first_coord, :, :])
ax[1].imshow(output_vector[0, 0, subvolume_first_coord, :, :])
ax[2].imshow(target_vector[0, 0, subvolume_first_coord, :, :])
plt.show()

# %%
fig, ax = plt.subplots()
hist, bins = np.histogram(output_vector.flatten(), bins=100)
ax.plot(bins[:-1], hist, label="output")
hist, bins = np.histogram(target_vector.flatten(), bins=100)
ax.plot(bins[:-1], hist, label="target")
ax.legend()
ax.grid()
