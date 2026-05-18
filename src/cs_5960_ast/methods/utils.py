import importlib
import json
import math
import os
import random
import sys
from configparser import RawConfigParser
from datetime import timedelta
from functools import cache

import numpy as np
import torch
import torch.nn.functional as F
from torch import nn


## custom
def pretty_timedelta(td: timedelta) -> str:
    """From: https://docs.python.org/3/library/datetime.html#timedelta-objects"""
    if td.days >= 0:
        return str(td)
    return f"-({-td!s})"


def str_to_bool(s: str | None) -> bool:
    """Convert string to boolean, for env variables mostly"""
    if s is None:
        return False
    if s.lower() in ["true", "1", "yes", "y"]:
        return True
    elif s.lower() in ["false", "0", "no", "n"]:
        return False
    else:
        msg = f"Cannot convert string '{s}' to boolean."
        raise ValueError(msg)


def k_to_length(k: float) -> float:
    """Convert a wavenumber k to length scale assuming standard fourier l = 2pi/k"""
    # k [h Mpc^-1]
    length = (2 * np.pi) / k  # [h^-1 Mpc]
    return length


# CAMELS smallest simulation scale is 0.09765625 [h^-1 Mpc], about 100 kpc/h
def length_to_k(length: float) -> float:
    """Convert a length scale to wavenumber k assuming standard fourier l = 2pi/k"""
    # length [h^-1 Mpc]
    k = (2 * np.pi) / length  # [h Mpc^-1]
    return k


## phd model with attention
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


class LeakyReLUStyled(nn.Module):
    def __init__(self, negative_slope=0.2, inplace=False, style_size=5):
        super().__init__()
        self.inplace = inplace
        self.negative_slope = negative_slope

        self.nn = nn.Sequential(
            nn.Linear(style_size, 2 * style_size), nn.LeakyReLU(0.2), nn.Linear(2 * style_size, 1), nn.Sigmoid()
        )

    def forward(self, x, style=None):
        if style is not None:
            slope = self.nn(style)  # shape [b, 1]
            slope = slope.view(-1, 1, 1, 1, 1)  # match 5D dim
            # print(f"LReLUS slope {slope}")
            return torch.where(x >= 0, x, x * slope)
        else:
            return F.leaky_relu(x, self.negative_slope, self.inplace)


class ConvStyled3d(nn.Module):
    """Convolution layer with modulation and demodulation, from StyleGAN2.

    Weight and bias initialization from `torch.nn._ConvNd.reset_parameters()`.
    """

    def __init__(
        self,
        in_chan: int,
        out_chan: int,
        style_size: int,
        kernel_size: int = 3,
        stride: int = 1,
        padding: int = 0,
        *,
        bias: bool = True,
        resample: str | None = None,
    ) -> None:
        super().__init__()

        # self.style_weight = nn.Parameter(torch.empty(in_chan, style_size))
        # nn.init.kaiming_uniform_(self.style_weight, a=math.sqrt(5),
        #                          mode='fan_in', nonlinearity='leaky_relu')
        # self.style_bias = nn.Parameter(torch.ones(in_chan))  # NOTE: init to 1

        if resample is None:
            K3 = (kernel_size,) * 3
            self.weight: nn.Parameter = nn.Parameter(torch.empty(out_chan, in_chan, *K3))
            self.stride: int = stride
            self.conv = F.conv3d
        elif resample == "U":
            K3 = (kernel_size,) * 3
            # NOTE not clear to me why convtranspose have channels swapped
            self.weight: nn.Parameter = nn.Parameter(torch.empty(in_chan, out_chan, *K3))
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
        self.style_size = style_size
        self.padding = padding

        nn.init.kaiming_uniform_(
            self.weight,
            a=math.sqrt(5),
            mode="fan_in",  # effectively 'fan_out' for 'D'
            nonlinearity="leaky_relu",
        )

        if bias:
            self.bias = nn.Parameter(torch.zeros(out_chan))
            fan_in, _ = nn.init._calculate_fan_in_and_fan_out(self.weight)  # noqa: SLF001
            # adapted from `torch.nn._ConvNd.reset_parameters()`
            bound = 1 / math.sqrt(fan_in)
            nn.init.uniform_(self.bias, -bound, bound)
        else:
            self.register_parameter("bias", None)

        def init_weight(m: nn.Module) -> None:
            if type(m) is nn.Linear:
                torch.nn.init.kaiming_uniform_(m.weight, a=math.sqrt(5), mode="fan_in", nonlinearity="leaky_relu")
                if m.bias is not None:
                    torch.nn.init.ones_(m.bias)

        self.style_block = nn.Sequential(
            nn.Linear(in_features=style_size, out_features=in_chan),
        )
        self.style_block.apply(init_weight)

    def forward(self, inputs: tuple[torch.Tensor, torch.Tensor]) -> torch.Tensor:
        x, s = inputs[0], inputs[1]
        eps = 1e-8

        N, Cin, *DHWin = x.shape

        C0, C1, *K3 = self.weight.shape

        if self.resample == "U":
            Cin, Cout = C0, C1
        else:
            Cout, Cin = C0, C1

        # s = F.linear(s, self.style_weight, bias=self.style_bias)
        s: torch.Tensor = self.style_block(s)
        # modulation
        if self.resample == "U":
            s: torch.Tensor = s.reshape(N, Cin, 1, 1, 1, 1)
        else:
            s: torch.Tensor = s.reshape(N, 1, Cin, 1, 1, 1)
        w: torch.Tensor = self.weight * s
        # print(Cin, "Cin2")
        # demodulation
        if self.resample == "U":
            fan_in_dim = (1, 3, 4, 5)
        else:
            fan_in_dim = (2, 3, 4, 5)
        w: torch.Tensor = w * torch.rsqrt(w.pow(2).sum(dim=fan_in_dim, keepdim=True) + eps)

        w: torch.Tensor = w.reshape(N * C0, C1, *K3)
        # print(N, Cin, *DHWin)
        x: torch.Tensor = x.reshape(1, N * Cin, *DHWin)
        # END HERE
        # print(f"x before conv in convstyled3d: {x.shape}")
        # 6 size dimension of padding because to torch that means pad the last 3 dimensions in tensor on both sides
        x = F.pad(x, (self.padding,) * 6, mode="reflect")
        # print(f"x after padding in convstyled3d: {x.shape}")
        x: torch.Tensor = self.conv(x, w, bias=self.bias, stride=self.stride, padding=0, groups=N)
        _, _, *DHWout = x.shape
        # print("N", N, "Cout", Cout, "DHWout", *DHWout)
        # x = x.reshape(N, Cout, *DHWout)
        x: torch.Tensor = x.view(N, Cout, *DHWout)

        # print(f"x after conv in convstyled3d: {x.shape}")
        return x


""" Attention """


class MHSelfAttention(nn.Module):
    def __init__(self, in_chan, mid_chan, num_heads=1):
        super(MHSelfAttention, self).__init__()

        self.num_heads = num_heads
        self.mid_chan = mid_chan
        self.in_chan = in_chan
        self.scale = mid_chan**-0.5

        self.norm = nn.GroupNorm(num_groups=in_chan, num_channels=in_chan, eps=1e-6, affine=True)

        self.q_conv = nn.Sequential(
            nn.Conv3d(in_chan, in_chan * 2 * num_heads, kernel_size=1),
            nn.Conv3d(in_chan * 2 * num_heads, mid_chan * 2 * num_heads, kernel_size=1),
            nn.Conv3d(mid_chan * 2 * num_heads, mid_chan * num_heads, kernel_size=1),
        )

        self.k_conv = nn.Sequential(
            nn.Conv3d(in_chan, in_chan * 2 * num_heads, kernel_size=1),
            nn.Conv3d(in_chan * 2 * num_heads, mid_chan * 2 * num_heads, kernel_size=1),
            nn.Conv3d(mid_chan * 2 * num_heads, mid_chan * num_heads, kernel_size=1),
        )

        self.v_conv = nn.Sequential(
            nn.Conv3d(in_chan, in_chan * 2 * num_heads, kernel_size=1),
            nn.Conv3d(in_chan * 2 * num_heads, in_chan * 2 * num_heads, kernel_size=1),
            nn.Conv3d(in_chan * 2 * num_heads, in_chan * num_heads, kernel_size=1),
        )

        self.out_conv = nn.Conv3d(in_chan * num_heads, in_chan, kernel_size=1)

        self.gamma = nn.Parameter(torch.ones(1, in_chan, 1, 1, 1))
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, x):
        b, c, h, w, d = x.size()
        N = h * w * d

        x_in = x
        x = self.norm(x)

        q = self.q_conv(x).view(b, self.num_heads, self.mid_chan, N).permute(0, 1, 3, 2)  # to [b,c,h*w*d]
        k = self.k_conv(x).view(b, self.num_heads, self.mid_chan, N)
        v = self.v_conv(x).view(b, self.num_heads, self.in_chan, N)

        attn_scores = torch.matmul(q, k) * self.scale
        del q, k
        attn_weights = self.softmax(attn_scores)
        del attn_scores

        out = torch.matmul(v, attn_weights.transpose(-2, -1))
        del attn_weights
        out = out.view(b, c * self.num_heads, h, w, d)
        out = self.out_conv(out)

        out = out * self.gamma + x_in
        return out


class MHCrossAttention(nn.Module):
    def __init__(self, in_chan, mid_chan, style_size=1, num_heads=1):
        super(MHCrossAttention, self).__init__()

        self.num_heads = num_heads
        self.mid_chan = mid_chan
        self.in_chan = in_chan
        self.scale = mid_chan**-0.5

        self.norm = nn.GroupNorm(num_groups=in_chan, num_channels=in_chan, eps=1e-6, affine=True)

        self.q_conv = nn.Sequential(
            nn.Conv3d(
                in_chan, in_chan * 2 * num_heads, kernel_size=3, padding=1, padding_mode="reflect"
            ),  # Give Q vector some information about neighbouring pixels
            nn.Conv3d(in_chan * 2 * num_heads, mid_chan * 2 * num_heads, kernel_size=1),
            nn.Conv3d(mid_chan * 2 * num_heads, mid_chan * num_heads, kernel_size=1),
        )

        self.k_proj = nn.Sequential(
            nn.Linear(style_size, style_size * 2 * num_heads),
            nn.Linear(style_size * 2 * num_heads, mid_chan * 2 * num_heads),
            nn.Linear(mid_chan * 2 * num_heads, mid_chan * num_heads),
        )

        self.v_proj = nn.Sequential(
            nn.Linear(style_size, style_size * 2 * num_heads),
            nn.Tanh(),
            nn.Linear(style_size * 2 * num_heads, in_chan * 2 * num_heads),
            nn.Linear(in_chan * 2 * num_heads, in_chan * num_heads),
        )

        self.out_conv = nn.Conv3d(in_chan * num_heads, in_chan, kernel_size=1)

        self.gamma = nn.Parameter(torch.ones(1, in_chan, 1, 1, 1))
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, x, s):
        b, c, h, w, d = x.size()
        N = h * w * d

        x_in = x
        x = self.norm(x)

        q = self.q_conv(x).view(b, self.num_heads, self.mid_chan, N).permute(0, 1, 3, 2)
        k = self.k_proj(s).view(b, self.num_heads, self.mid_chan)
        k = k.unsqueeze(2)
        v = self.v_proj(s).view(b, self.num_heads, self.in_chan)
        v = v.unsqueeze(2)

        attn_scores = torch.matmul(q, k.transpose(-2, -1)) * self.scale
        del q, k
        attn_weights = self.softmax(attn_scores)
        del attn_scores
        out = torch.matmul(attn_weights, v)
        del attn_weights
        out = out.permute(0, 1, 3, 2).contiguous()
        out = out.view(b, c * self.num_heads, h, w, d)

        out = self.out_conv(out)

        out = out * self.gamma + x_in
        return out


## ember 2

# .......................................................................................


def load_config(cfg):
    parser = RawConfigParser()
    parser.read(cfg)

    h = {s: dict(parser.items(s)) for s in parser.sections()}
    d = {}
    for key in h["config"].keys():
        d[key] = json.loads(parser.get("config", key))
    return d


# .......................................................................................


def set_seed(seed: float) -> None:
    """Obsolete"""
    random.seed(seed)
    np.random.seed(seed)  # noqa: NPY002
    torch.manual_seed(seed)


def summary(net: nn.Module) -> str:
    """Return a string with a rounded estimate of the number of parameters in the passed module

    Inherited from EMBER-2

    `return rf"{round(sum(p.numel() for p in net.parameters()) / 1e6, 1)}e6"`
    """
    return rf"{round(sum(p.numel() for p in net.parameters()) / 1e6, 1)}e6"


def data_get(tensor: torch.Tensor) -> float:
    """Get the value of a torch tensor as a simple python float, without gradient

    Inherited from EMBER-2

    `return tensor.detach().item()`
    """
    return tensor.detach().item()


def to_numpy(x: torch.Tensor) -> np.ndarray:
    """Get tensor as a numpy array with no gradient attached, and moved to CPU

    Inherited from EMBER-2

    `return x.detach().cpu().numpy()`
    """
    return x.detach().cpu().numpy()


def make_noise(x: torch.Tensor, c: int = 8) -> torch.Tensor:
    """Makes a tensor of normal noise with shape like x

    The tensor args:

    n: batch number
    c: channel
    h: height
    w: width
    d: depth
    """
    n, _, h, w, d = x.shape  # promoted c to kwarg
    return torch.normal(0, 0.1, (n, c, h, w, d), device=x.device)


def toggle_grad(nets: nn.Module | list[nn.Module], *, requires_grad: bool = False) -> None:
    """Set requires_grad=False for all the networks to avoid unnecessary computations

    Parameters
    ----------
        nets (network list)   -- a list of networks
        requires_grad (bool)  -- whether the networks require gradients or not
    """
    if not isinstance(nets, list):
        nets: list[nn.Module] = [nets]
    for net in nets:
        if net is not None:
            for param in net.parameters():  # ty:ignore[unresolved-attribute] what does ~None mean and why is it here??
                param.requires_grad = requires_grad


def transfer_to(data: dict, device: torch.device) -> dict:
    """EMBER-2 method to recursively send torch tensors to a specified torch device"""
    for key in data.keys():  # noqa: PLC0206, SIM118
        if type(data[key]) is dict:  # this should be isinstance()
            for subkey in data[key].keys():  # noqa: SIM118
                data[key][subkey] = data[key][subkey].to(device)
        else:
            data[key] = data[key].to(device)
    return data


# .......................................................................................


def NSLoss(r: torch.Tensor | None, f: torch.Tensor, mode: str) -> torch.Tensor:  # noqa: N802 this is an established scientific function
    """See equations 5, 6, 7 in EMBER-2 paper: https://arxiv.org/abs/2502.15875

    This is the loss function we use based on their reasoning.

    r is not used in Discriminator mode, and so can be skipped by providing it as None

    r for real tensor, f for fake tensor, scores from discriminator setup
    """
    if mode == "D" and r is not None:
        return (F.softplus(-r) + F.softplus(f)).mean()
    elif mode == "G":
        return F.softplus(-f).mean()

    #  errors
    elif mode == "D" and r is None:
        msg = "r tensor must be provided in Discriminator mode"
        raise ValueError(msg)
    else:
        msg = "mode must be 'D' (discriminator) or 'G' (generator)"
        raise ValueError(msg)


# .......................................................................................


def add_context(x: torch.Tensor, context: torch.Tensor) -> torch.Tensor:
    """EMBER-2 method to support pair_base method, stacks style tensor with the tensors to be discriminated

    The tensor args:

    n: batch number
    _: channel, ignored
    h: height
    w: width
    d: depth
    """
    n, _, h, w, d = x.shape
    c = context.shape[-1]

    c_feat = context.view(n, c, 1, 1, 1).repeat(1, 1, h, w, d)
    return torch.cat([x, c_feat], dim=1)


def pair_base(x: torch.Tensor, y: torch.Tensor, p: torch.Tensor, context: torch.Tensor) -> torch.Tensor:
    """EMBER-2 method used to combine tensors for passing to discriminator, which will learn to discriminate between them

    x: input tensor
    y: first tensor for discrimination (should be generated by model)
    p: second tensor for discrimination (can be target data or another independently generated tensor)
    context: style tensor
    """
    x = torch.cat([x, y, p], dim=1)
    x = add_context(x, context)
    return x


# .......................................................................................


## map2map
@cache
def import_attr(name, *pkgs, callback_at=None):
    """Import attribute. Try package first and then callback directory.

    To use a callback, `name` must contain a module, formatted as 'mod.attr'.

    map2map inherited method

    Examples
    --------
    >>> import_attr('attr', pkg1.pkg2)

    tries to import attr from pkg1.pkg2.

    >>> import_attr('mod.attr', pkg1.pkg2, pkg3, callback_at='path/to/cb_dir')

    first tries to import attr from pkg1.pkg2.mod, then from pkg3.mod, finally
    from 'path/to/cb_dir/mod.py'.
    """
    if name.count(".") == 0:
        attr = name

        errors = []

        for pkg in pkgs:
            try:
                return getattr(importlib.import_module(pkg.__name__), attr)
            except (ModuleNotFoundError, AttributeError) as e:
                errors.append(e)

        raise Exception(errors)
    else:
        mod, attr = name.rsplit(".", 1)

        errors = []

        for pkg in pkgs:
            try:
                return getattr(importlib.import_module(pkg.__name__ + "." + mod), attr)
            except (ModuleNotFoundError, AttributeError) as e:
                errors.append(e)

        if callback_at is None:
            raise Exception(errors)

        callback_at = os.path.join(callback_at, mod + ".py")
        if not os.path.isfile(callback_at):
            raise FileNotFoundError("callback file not found")

        if mod in sys.modules:
            return getattr(sys.modules[mod], attr)

        spec = importlib.util.spec_from_file_location(mod, callback_at)
        module = importlib.util.module_from_spec(spec)
        sys.modules[mod] = module
        spec.loader.exec_module(module)

        return getattr(module, attr)
