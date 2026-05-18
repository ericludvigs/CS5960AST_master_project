from typing import cast

import torch
from torch import nn

# from lib.nn.layers import *
# from lib.nn.utils import *
from cs_5960_ast import AstNetConfigDict
from cs_5960_ast.ember2.layers import DiscriminatorBlock, FinalDiscriminatorBlock, SynthesisBlock, dBlock


class Generator(nn.Module):
    """The generator part of the net, see 3.2 in https://arxiv.org/abs/2502.15875"""

    def __init__(self, mode: str, config: AstNetConfigDict) -> None:
        super().__init__()

        self.mapping = MappingNet(**config)
        self.synthesis = SynthesisNet(mode, **config)

    def forward(self, x: torch.Tensor, context: torch.Tensor, *, noise: torch.Tensor | bool | None = None) -> torch.Tensor:
        style: torch.Tensor = self.mapping(context=context)
        out: torch.Tensor = self.synthesis(x=x, style=style, noise=noise)
        return out


class MappingNet(nn.Module):
    """From ember2 paper:

    `The Mapping network shown in purple is a simple 5-layer multi-layer perceptron that maps the global redshift information z
    to the style vector w.`
    """

    def __init__(self, style_depth: int, style_dim: int, context_dim: int, **kwargs: str | float | list[int] | tuple) -> None:
        super().__init__()

        layers = []
        for i in range(style_depth):
            layers.append(nn.Linear(in_features=context_dim if i == 0 else style_dim, out_features=style_dim))
            layers.append(nn.LeakyReLU(0.1))
        self.net: nn.Sequential = nn.Sequential(*layers)

    def forward(self, context: torch.Tensor) -> torch.Tensor:
        return self.net(context)


class SynthesisNet(nn.Module):
    """Main part of the generator, with both up blocks and down blocks. Convolution happens here."""

    def __init__(
        self,
        mode: str,
        style_dim: int,
        inp_channels: int,
        out_channels: int,
        filters: list[int],
        num_noise_ch: int,
        **kwargs: str | float | tuple,
    ) -> None:
        super().__init__()

        self.mode = mode
        self.filters = filters

        self.dblocks: nn.ModuleList[nn.Module] = nn.ModuleList()
        self.ublocks: nn.ModuleList[nn.Module] = nn.ModuleList()
        self.ublocks_index: list[int] = []

        fixed_channels = filters[-1]  # [0] TODO: experiment if 128 fixed for downsampling is better
        for i in range(len(filters)):
            is_first = i == 0
            is_last = i == len(filters) - 1

            dblock: dBlock = dBlock(
                inp_ch=fixed_channels if not is_first else inp_channels,
                out_ch=fixed_channels,
                is_last=is_last,
                style_dim=style_dim,
            )

            ublock: SynthesisBlock = SynthesisBlock(
                style_dim=style_dim,
                inp_ch=filters[i],
                out_ch=filters[max(i - 1, 0)],
                rgb_ch=out_channels,
                cond_ch=fixed_channels,
                num_noise_ch=num_noise_ch,
                is_last=is_last,
                is_first=is_first,
            )

            self.dblocks.append(module=dblock)
            self.ublocks.append(module=ublock)
            self.ublocks_index.append(i)

    def forward(self, x: torch.Tensor | None, style: torch.Tensor, *, noise: torch.Tensor | bool | None) -> torch.Tensor:
        _xres = x

        ys: list[torch.Tensor] = []
        for block in self.dblocks:
            block = cast("dBlock", block)
            x, y = block(x, style)
            ys.append(y)

        x = None
        for block, y in zip(self.ublocks[::-1], ys[::-1], strict=True):
            block = cast("SynthesisBlock", block)
            y = cast("torch.Tensor", y)
            x: torch.Tensor = block(x, y, style, noise=noise)

        if x is None:
            msg = (
                f"Tensor x being returned from SynthesisNet is None, should have passed through SynthesisBlocks in: {self.ublocks}"
            )
            raise ValueError(msg)

        return x


class Discriminator(nn.Module):
    """The discriminator part of the net, see 3.2 in https://arxiv.org/abs/2502.15875"""

    def __init__(
        self, mode: str, inp_channels: int, out_channels: int, context_dim: int, filters: list[int], **kwargs: str | float | tuple
    ) -> None:
        super().__init__()

        if mode == "base":
            in_channels = inp_channels + out_channels * 2 + context_dim  # out_channels * 2 for Adler approach
        elif mode == "zoom":
            # in_channels = inp_channels + out_channels * 2 + context_dim # +0 -> *2
            in_channels = (inp_channels + out_channels) * 2 + context_dim
        else:
            raise SystemExit

        filters = [in_channels, *filters]
        self.blocks = nn.ModuleList()
        for i in range(len(filters) - 1):
            is_last = i == len(filters) - 2
            if not is_last:
                block = DiscriminatorBlock(inp_ch=filters[i], out_ch=filters[i + 1], context_dim=context_dim)

            else:
                block = FinalDiscriminatorBlock(filters[i], context_dim=context_dim)
            self.blocks.append(block)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        for _i, block in enumerate(self.blocks):
            x = block(x)
        return x
