import torch
import torch as th
import torch.nn.functional as F
from torch import nn

# import cs_5960_ast.devel.AUNetG as AUNet
from cs_5960_ast.methods.utils import ConvStyled3d

# from lib.nn.torch_utils import misc, persistence
# from lib.nn.torch_utils.ops import bias_act, conv2d_resample, fma, upfirdn2d
# from utils import ConvStyled3d


class dBlock(nn.Module):  # noqa: N801 name inherited from EMBER-2
    """Down block for use in a U-Net, part of the encoder side"""

    def __init__(self, inp_ch: int, out_ch: int, *, is_last: bool, style_dim: int = 7) -> None:
        super().__init__()

        # self.resblock1 = AUNet.ResNetBlock(inp_ch, out_ch, out_ch, style_dim)

        self.conv1 = ConvStyled3d(
            in_chan=inp_ch, out_chan=out_ch, style_size=style_dim, kernel_size=3, stride=1, padding=1, bias=True
        )
        self.conv2 = ConvStyled3d(
            in_chan=out_ch, out_chan=out_ch, style_size=style_dim, kernel_size=3, stride=1, padding=1, bias=True
        )
        self.act = nn.LeakyReLU(0.2, inplace=True)

        # nn.Conv2d(in_channels=inp_ch, out_channels=out_ch, style_size=style_dim, kernel_size=3, stride=1, bias=True)
        # self.conv1 = nn.Conv2d(in_channels=inp_ch, out_channels=out_ch, kernel_size=3, stride=1, padding=1)
        # self.conv2 = nn.Conv2d(out_ch, out_ch, kernel_size=3, stride=1, padding=1)
        # self.pool = nn.Conv2d(out_ch, out_ch, kernel_size=3, stride=2, padding=1)

        self.is_last = is_last
        if not is_last:
            self.pool = ConvStyled3d(
                in_chan=out_ch, out_chan=out_ch, style_size=style_dim, kernel_size=2, stride=2, padding=0, bias=True
            )

    def forward(self, x: torch.Tensor, s: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        # print(f"x shape before down conv1: {x.shape}")
        x = self.act(self.conv1((x, s)))
        # print(f"x shape before down conv2: {x.shape}")
        x = self.act(self.conv2((x, s)))
        y = x

        if not self.is_last:
            x = self.pool((x, s))

        # print(f"x shape before returning after downblock: {x.shape}")
        # print(f"y shape before returning: {y.shape}")
        return x, y


class GenerateNoise(nn.Module):
    """Generate new noise to be concatenated to a neural network inside the network layers.

    The number of channels `chan` should be 1 for StyleGAN2
    or that of the input for plain StyleGAN.

    `style_size` is the size of the style vector to be used when generating noise later.

    EMBER-2 used 8 channels of plain Gaussian noise

    Intended for use in the upsampling blocks of a U-Net,
    specifically the decoder blocks of EMBER-2
    """

    def __init__(self, chan: int = 1, style_size: int = 1) -> None:
        super().__init__()
        self.chan = chan

        self.nn = nn.Sequential(
            nn.Linear(style_size, 8),
            nn.Tanh(),
            nn.Linear(8, 16),
            nn.Linear(16, 2 * chan),
        )
        return None

    def forward(self, x: torch.Tensor, s: torch.Tensor) -> torch.Tensor:
        """Indexes are:

        b: batch number
        _c: channel (not used)
        h: height
        w: width
        d: depth
        """
        b, _c, h, w, d = x.shape
        noise = torch.randn(b, self.chan, h, w, d, device=x.device)

        scale, bias = self.nn(s).chunk(2, dim=1)
        # print(f"Addnoise Scale {scale}, Bias {bias}")
        scale = scale.view(b, -1, 1, 1, 1)
        bias = bias.view(b, -1, 1, 1, 1)
        noise = noise * scale + bias

        return noise


class SynthesisBlock(nn.Module):
    """Up block for use in a U-Net, part of the decoder side

    Also called synthesis because it synthesizes new tensors via upscaling
    """

    def __init__(
        self,
        style_dim: int,
        inp_ch: int,
        out_ch: int,
        rgb_ch: int,
        cond_ch: int,
        num_noise_ch: int,
        *,
        is_last: bool,
        is_first: bool,
    ) -> None:
        super().__init__()

        # NOTE: channels of noise to be generated and passed through noise mapping network
        self.noise_ch = num_noise_ch
        total_ch = cond_ch if is_last else inp_ch + cond_ch

        self.conv1: ConvStyled3d = ConvStyled3d(in_chan=total_ch, out_chan=out_ch, style_size=style_dim, padding=1)
        self.conv2: ConvStyled3d = ConvStyled3d(in_chan=out_ch + self.noise_ch, out_chan=out_ch, style_size=style_dim, padding=1)

        self.is_first = is_first
        if self.is_first:
            self.iconv: ConvStyled3d = ConvStyled3d(in_chan=out_ch, out_chan=rgb_ch, style_size=style_dim, kernel_size=3, padding=1)

        # make a new network inside the block that can generate noise to concat to the output
        self.noise_generator = GenerateNoise(chan=self.noise_ch, style_size=style_dim)

    def cat_noise(self, x: torch.Tensor, noise: torch.Tensor | None = None) -> torch.Tensor:
        """Concatenates noise to an input tensor

        Indexes are:

        bn: batch number
        c: channel
        h: height
        w: width
        d: depth
        """
        _bn, _c, h, w, d = x.shape
        if noise is None:
            n = x.new_empty(*x.shape).normal_(std=0.1)[:, : self.noise_ch]
        else:
            n = noise[:, : self.noise_ch, :h, :w, :d]
        return th.cat([x, n], dim=1)

    def forward(
        self, x: torch.Tensor | None, y: torch.Tensor, style: torch.Tensor, *, noise: torch.Tensor | bool | None = None
    ) -> torch.Tensor:
        """See block diagram for EMBER-2

        interpolates tensor from earlier up in downblocks, then stacks with newly up-blocked tensor
        """
        if x is not None:
            try:
                x = F.interpolate(input=x, scale_factor=2, mode="trilinear")

                # in case of odd size, pad tensor by 1 to make even for concat
                def true_fn(x: torch.Tensor) -> torch.Tensor:
                    x = F.pad(input=x, pad=(0, 1, 0, 1, 0, 1), mode="reflect")
                    # pad dimensions from the left by (0, 1), (0, 1), and (0, 1)
                    # see https://docs.pytorch.org/docs/stable/generated/torch.nn.functional.pad.html
                    # the function signature is very confusing
                    return x

                # else: do nothing, just proceed to concatenation below
                def false_fn(x: torch.Tensor) -> torch.Tensor:
                    return x.clone()

                """.clone() because of warning:
                torch._dynamo.exc.UncapturedHigherOrderOpError:
                    Cond doesn't work unless it is captured
                    completely with torch.compile. Got Encountered aliasing during higher order op tracing
                Explanation: Higher order ops do not support aliasing. Found in cond
                Hint: Replace `return input` with `return input.clone()` to avoid aliasing.
                """

                # in y.shape[], 0 is batch, 1 is channel, 2, 3, 4 are 3D dimensions

                # torch's internal branching system that compiles properly for graph and better speed
                x: torch.Tensor = torch.cond(y.shape[2] % 2 != 0, true_fn, false_fn, (x,))

                x = th.cat(tensors=(x, y), dim=1)

            except RuntimeError as e:
                msg1 = "Attempted to concatenate tensors:"
                msg2 = f"x shape: {x.shape}, y shape: {y.shape}"
                msg = f"Error during upsample and concat in SynthesisBlock forward\n{msg1}\n{msg2}"
                raise RuntimeError(msg) from e
        else:
            x = y

        if isinstance(noise, bool):
            if noise is True:
                noise_tensor: torch.Tensor = self.noise_generator(x, style)
            else:
                noise_tensor = None
        elif noise is None:
            noise_tensor = None
        elif isinstance(noise, torch.Tensor):
            noise_tensor: torch.Tensor = noise

        x: torch.Tensor = self.conv1((x, style))
        x: torch.Tensor = self.cat_noise(x, noise_tensor)
        x: torch.Tensor = self.conv2((x, style))

        if self.is_first:
            x: torch.Tensor = self.iconv((x, style))
        return x


class DiscriminatorBlock(nn.Module):
    """Block for use in the EMBER-2 discriminator network"""

    def __init__(self, inp_ch: int, out_ch: int, context_dim: int) -> None:
        super().__init__()

        # left path
        # self.skip = nn.Conv2d(in_channels=inp_ch, out_channels=out_ch, kernel_size=3, stride=2, padding=1)
        self.skip = nn.Conv3d(in_channels=inp_ch, out_channels=out_ch, kernel_size=3, stride=2, padding=1, padding_mode="reflect")
        # self.skip = ConvStyled3d(in_chan=inp_ch, out_chan=out_ch, kernel_size=3, stride=2, padding=1, style_size=context_dim)
        # self.conv1 = nn.Conv2d(in_channels=inp_ch, out_channels=out_ch, kernel_size=3, stride=1, padding=1)
        self.conv1 = nn.Conv3d(in_channels=inp_ch, out_channels=out_ch, kernel_size=3, stride=1, padding=1, padding_mode="reflect")
        # self.conv1 = ConvStyled3d(in_chan=inp_ch, out_chan=out_ch, kernel_size=3, stride=1, padding=1, style_size=context_dim)
        # self.conv2 = nn.Conv2d(in_channels=out_ch, out_channels=out_ch, kernel_size=3, stride=1, padding=1)
        self.conv2 = nn.Conv3d(in_channels=out_ch, out_channels=out_ch, kernel_size=3, stride=1, padding=1, padding_mode="reflect")
        # self.conv2 = ConvStyled3d(in_chan=out_ch, out_chan=out_ch, kernel_size=3, stride=1, padding=1, style_size=context_dim)

        self.pool = nn.Conv3d(in_channels=out_ch, out_channels=out_ch, kernel_size=3, stride=2, padding=1, padding_mode="reflect")
        # self.pool = ConvStyled3d(in_chan=out_ch, out_chan=out_ch, kernel_size=3, stride=2, padding=1, style_size=context_dim)
        self.act = nn.LeakyReLU(negative_slope=0.2, inplace=True)

        # right path
        self.fftc1 = nn.Conv3d(
            in_channels=inp_ch * 2, out_channels=out_ch * 2, kernel_size=3, stride=1, padding=1, padding_mode="reflect"
        )
        # self.fftc1 = ConvStyled3d(
        #    in_chan=inp_ch * 2, out_chan=out_ch * 2, kernel_size=3, stride=1, padding=1, style_size=context_dim
        # )
        self.fftc2 = nn.Conv3d(
            in_channels=out_ch * 2, out_channels=out_ch * 2, kernel_size=3, stride=1, padding=1, padding_mode="reflect"
        )
        # self.fftc2 = ConvStyled3d(
        #    in_chan=out_ch * 2, out_chan=out_ch * 2, kernel_size=3, stride=1, padding=1, style_size=context_dim
        # )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x_ = x.clone()
        skip = self.skip(x)

        # left path
        x = self.act(self.conv1(x))
        x = self.act(self.conv2(x))

        # right path
        # fft on tensor with dim (0, 1, 2, 3, 4) = (bn, c, h, w, d)
        y = th.fft.fftn(input=x_, dim=(2, 3, 4))
        y = th.cat([y.real, y.imag], dim=1)
        y = self.act(self.fftc1(y))
        y = self.act(self.fftc2(y))

        yr, yi = torch.chunk(y, 2, dim=1)
        y = th.complex(yr, yi)
        y = th.fft.ifftn(y, dim=(2, 3, 4)).real

        # residual and pool
        x = x + y
        # TODO: potential improvement, group norm for normalizing concatenation if fourier is included
        x = self.pool(x)
        return x + skip


class FinalDiscriminatorBlock(nn.Module):
    """EMBER-2 architecture calls for a distinct final discriminator block that outputs a final discrimination score"""

    def __init__(self, inp_ch: int, context_dim: int) -> None:
        super().__init__()

        self.conv = nn.Conv3d(in_channels=inp_ch, out_channels=inp_ch, kernel_size=3, stride=1, padding=1, padding_mode="reflect")
        # self.conv = ConvStyled3d(in_chan=inp_ch, out_chan=inp_ch, kernel_size=3, stride=1, padding=1, style_size=context_dim)
        self.act = nn.LeakyReLU(negative_slope=0.2, inplace=True)
        self.fc = nn.Conv3d(in_channels=inp_ch, out_channels=1, kernel_size=4, padding_mode="reflect")
        # self.fc = ConvStyled3d(in_chan=inp_ch, out_chan=1, kernel_size=4, style_size=context_dim)

    def flatten(self, x: torch.Tensor) -> torch.Tensor:
        return x.reshape(x.shape[0], -1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x: torch.Tensor = self.act(self.conv(x))
        x = self.flatten(self.fc(x))
        return x.squeeze()
