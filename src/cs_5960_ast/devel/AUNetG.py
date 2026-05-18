import itertools
import math
from math import log2

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

""" Smaller modules """


def narrow_like_(a, b):
    """Narrow a to be like b.

    Try to be symmetric but cut more on the right for odd difference
    """
    for d in range(2, a.dim()):
        width = a.shape[d] - b.shape[d]
        half_width = width // 2
        a = a.narrow(d, half_width, a.shape[d] - width)
    return a


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


class GEGLU(nn.Module):
    r"""
    A variant of the gated linear unit activation function
    from https://arxiv.org/abs/2002.05202.
    """

    def __init__(self, dim_in: int, dim_out: int):
        super().__init__()
        self.proj = nn.Linear(dim_in, dim_out * 2)

    def gelu(self, gate):  # not support mps
        return F.gelu(gate.to(dtype=torch.float32)).to(dtype=gate.dtype)

    def forward(self, h):
        h, gate = self.proj(h).chunk(2, dim=-1)
        return h * self.gelu(gate)


class GEGLU3D(nn.Module):
    def __init__(self, in_chan, out_chan):
        super().__init__()
        self.proj = nn.Conv3d(in_chan, out_chan * 2, kernel_size=1)

    def gelu(self, gate):
        return F.gelu(gate.to(dtype=torch.float32)).to(dtype=gate.dtype)

    def forward(self, h):
        h, gate = self.proj(h).chunk(2, dim=1)
        return h * self.gelu(gate)


'''
class AddNoise(nn.Module):
    """Add or concatenate noise.

    Add noise if `cat=False`.
    The number of channels `chan` should be 1 (StyleGAN2)
    or that of the input (StyleGAN).
    """

    def __init__(self, cat, chan=1, style_size=1):
        super().__init__()

        self.cat = cat
        #self.nn = StyleNoise(in_chan=1, style_size=style_size)

        if not self.cat:
            self.std = nn.Parameter(torch.zeros([chan]))

    def forward(self, x):
        noise = torch.randn_like(x[:, :1]) #self.nn(torch.randn_like(x[:, :1]), s)

        if self.cat:
            x = torch.cat([x, noise], dim=1)
        else:
            std_shape = (-1,) + (1,) * (x.dim() - 2)
            noise = self.std.view(std_shape) * noise

            x = x + noise

        return x
'''


class AddNoise(nn.Module):
    """Add or concatenate noise.

    Add noise if `cat=False`.
    The number of channels `chan` should be 1 (StyleGAN2)
    or that of the input (StyleGAN).
    """

    def __init__(self, cat, chan=1, style_size=1):
        super().__init__()

        self.cat = cat
        self.chan = chan

        self.nn = nn.Sequential(nn.Linear(style_size, 8), nn.Tanh(), nn.Linear(8, 16), nn.Linear(16, 2 * chan))

    def forward(self, x, s):
        b, c, h, w, d = x.shape
        noise = torch.randn(b, self.chan, h, w, d, device=x.device)

        scale, bias = self.nn(s).chunk(2, dim=1)
        # print(f"Addnoise Scale {scale}, Bias {bias}")
        scale = scale.view(b, -1, 1, 1, 1)
        bias = bias.view(b, -1, 1, 1, 1)
        noise = noise * scale + bias

        if self.cat:
            x = torch.cat([x, noise], dim=1)
        else:
            x = x + noise

        return x


""" map2map styled modules """

'''
class LeakyReLUStyled(nn.LeakyReLU):
    def __init__(self, negative_slope=0.2, inplace=False):
        super().__init__(negative_slope, inplace)

    """ Trivially evaluates standard leaky ReLU, but accepts second argument

    for style array that is not used
    """

    def forward(self, x, style=None):
        return super().forward(x)
'''


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
            raise ValueError("resample type {} not supported".format(resample))
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


class TransformerBlock(nn.Module):
    def __init__(self, in_chan, mid_chan, style_size, self_num_heads=16, cross_num_heads=32):
        super().__init__()

        self.norm = nn.GroupNorm(num_groups=in_chan, num_channels=in_chan, eps=1e-6, affine=True)
        self.conv0 = nn.Conv3d(in_chan, mid_chan, kernel_size=1)

        self.self_attention = MHSelfAttention(mid_chan, mid_chan, self_num_heads)
        self.cross_attention = MHCrossAttention(mid_chan, mid_chan, style_size, cross_num_heads)

        self.norm1 = nn.GroupNorm(num_groups=mid_chan, num_channels=mid_chan, eps=1e-6, affine=True)
        self.act = GEGLU3D(mid_chan, mid_chan)
        self.conv1 = nn.Conv3d(mid_chan, mid_chan, kernel_size=1)

        self.conv_out = nn.Conv3d(mid_chan, in_chan, kernel_size=1)

    def forward(self, x, s):
        x_in = x

        x = self.norm(x)
        x = self.conv0(x)
        x = self.self_attention(x)
        x = self.cross_attention(x, s)

        x1 = x
        x = self.norm1(x)
        x = self.act(x)
        x = self.conv1(x)
        x = x + x1
        del x1

        x = self.conv_out(x)
        return x + x_in


""" Unet Modules """


class ResNetBlock(nn.Module):
    def __init__(self, in_chan, mid_chan, out_chan, style_size, last_act=True, cat_noise=False):
        super().__init__()
        self.last_act = last_act
        self.style_size = style_size

        # ResBlock: addnoise, conv, act, addnoise, conv, act
        self.skip = ConvStyled3d(in_chan, out_chan, self.style_size, kernel_size=1)

        self.addnoise1 = AddNoise(cat_noise, style_size=style_size)
        self.conv1 = ConvStyled3d(in_chan + cat_noise, mid_chan, self.style_size, kernel_size=3)
        self.act1 = LeakyReLUStyled(0.2, True)
        self.addnoise2 = AddNoise(cat_noise, style_size=style_size)
        self.conv2 = ConvStyled3d(mid_chan + cat_noise, out_chan, self.style_size, kernel_size=3)
        self.act2 = LeakyReLUStyled(0.2, True)

        if last_act:
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
        x = self.addnoise1(x, s)
        x = self.conv1((x, s))
        x = self.act1(x, s)

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
        x = self.addnoise2(x, s)
        x = self.conv2((x, s))
        x = self.act2(x, s)

        x = x + narrow_like_(y, x)

        if self.last_act:
            x = self.act3(x, s)

        return x


class UpBlock(nn.Module):
    def __init__(self, in_chan, style_size, cat_noise=False):
        super().__init__()
        self.addnoise1 = AddNoise(cat_noise, style_size=style_size)
        # self.up = ConvStyled3d(in_chan+cat_noise, in_chan, style_size=style_size, kernel_size=2, stride=2, resample='U')
        self.up = Resampler(2, 3)
        self.conv1 = ConvStyled3d(in_chan, in_chan, style_size, kernel_size=3)
        self.act1 = LeakyReLUStyled(0.2, True)
        self.addnoise2 = AddNoise(cat_noise, style_size=style_size)
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
        x = self.addnoise1(x, s)
        # x = self.up((x,s))
        x = F.interpolate(x, scale_factor=2, mode="trilinear", align_corners=False)

        x = self.conv1((x, s))
        x = self.act1(x, s)
        x = self.addnoise2(x, s)
        x = self.conv2((x, s))
        x = self.act2(x, s)
        return x


class DownBlock(nn.Module):
    def __init__(self, in_chan, style_size, cat_noise=False):
        super().__init__()
        self.addnoise = AddNoise(cat_noise, style_size=style_size)
        self.conv = ConvStyled3d(in_chan + cat_noise, in_chan, style_size, kernel_size=2, stride=2)
        self.act = LeakyReLUStyled(0.2, True)

    def forward(self, x, s):
        x = self.addnoise(x, s)
        x = self.conv((x, s))
        x = self.act(x, s)
        return x


class AUNetG(nn.Module):
    def __init__(
        self, in_chan, out_chan, style_size, scale_factor=16, chan_base=512, chan_min=16, chan_max=512, cat_noise=True, **kwargs
    ):
        super().__init__()
        # https://github.com/dschaurecker/dl_halo/blob/main/train_test/map2map/models/generator.py

        self.conv_in = ConvStyled3d(in_chan, 32, style_size, kernel_size=1, stride=1)

        self.resblock1 = ResNetBlock(32, 64, 64, style_size)  # +2
        self.down1 = DownBlock(64, style_size)

        self.resblock2 = ResNetBlock(64, 128, 128, style_size)  # +4
        self.down2 = DownBlock(128, style_size)

        self.bottleneck1 = ResNetBlock(128, 256, 512, style_size)  # +8
        self.bottleneck2 = ResNetBlock(512, 256, 128, style_size)

        self.up1 = UpBlock(128, style_size)
        self.resblock3 = ResNetBlock(128 + 128, 128, 64, style_size)

        self.up2 = UpBlock(64, style_size)
        self.resblock4 = ResNetBlock(64 + 64, 64, 32, style_size, last_act=False)

        self.conv_out = ConvStyled3d(32, out_chan, style_size, kernel_size=1, stride=1)

    def forward(self, x, s):
        x = F.interpolate(x, scale_factor=2, mode="trilinear", align_corners=False)

        y = x  # Predict the residual to y

        x = self.conv_in((x, s))

        x1 = self.resblock1(x, s)  # +2
        x = self.down1(x1, s)

        x2 = self.resblock2(x, s)  # +4
        x = self.down2(x2, s)

        x = self.bottleneck1(x, s)  # +8
        x = self.bottleneck2(x, s)

        x = self.up1(x, s)
        x2 = narrow_like_(x2, x)
        x = torch.cat([x2, x], dim=1)
        x = self.resblock3(x, s)

        x = self.up2(x, s)
        x1 = narrow_like_(x1, x)
        x = torch.cat([x1, x], dim=1)
        x = self.resblock4(x, s)

        x = self.conv_out((x, s))

        return y + x
