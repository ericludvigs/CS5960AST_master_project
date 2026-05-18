"""File kept for reference, but is not directly used and will not function"""

import warnings

import tqdm
from torch import nn
from torch.optim import Adam

from cs_5960_ast.methods.utils import NSLoss, data_get, pair_base, summary, toggle_grad

from .modules import Discriminator, Generator

# old imports:
# from lib.nn.modules import *
# from lib.nn.utils import *
# from sorcery import dict_of


warnings.filterwarnings("ignore")


# ........................................................................


class Ember2Net(nn.Module):
    def __init__(self, config, mode):
        """Main ember2 model class"""
        super().__init__()

        self.mode = mode
        self.config = config
        cfg = self.get_cfg(mode)

        self.G = Generator(mode, cfg)
        self.D = Discriminator(mode, **cfg)

        self.GE = Generator(mode, cfg)
        self.ema = EMA(0.995)
        self.configure_optimizers()

    def forward(self, x, c, noise=None):
        """Forward is never visibly used"""
        return self.GE(x, c, noise)  # G or GE? # <- ember2 shipped GE, but were clearly unsure

    def summary(self):
        """Passed to tensorboard"""
        print("Gen:", summary(self.G), flush=True)
        print("Dis:", summary(self.D), flush=True)

    def EMA(self):
        """EMA implementation, used during training"""

        def update_moving_average(current_model, ma_model):
            for current_params, ma_params in zip(current_model.parameters(), ma_model.parameters()):
                old_weight, up_weight = ma_params.data, current_params.data
                ma_params.data = self.ema.update_average(old_weight, up_weight)

        update_moving_average(self.G, self.GE)

    def configure_optimizers(self):
        """Used before training, preconfigured with values"""
        lr_g = self.config["lr_g"]
        lr_d = self.config["lr_d"]
        self.opt_g = Adam(self.G.parameters(), lr=lr_g, betas=(0.5, 0.9))  # base: (0.0, 0.99)
        self.opt_d = Adam(self.D.parameters(), lr=lr_d, betas=(0.5, 0.9))  # base: (0.0, 0.99)

    def qim(self, x, cmap="viridis"):
        """Never used"""
        import matplotlib.pyplot as plt

        plt.figure()
        plt.imshow(x[0].detach().cpu().numpy(), cmap=cmap)
        plt.show()

    def _step_base(self, x, y, c):
        """Important for training"""
        # train G
        self.opt_g.zero_grad(set_to_none=True)
        toggle_grad(self.G, requires_grad=True)
        toggle_grad(self.D, requires_grad=False)

        p1 = self.G(x, c)
        p2 = self.G(x, c)

        f = self.D(pair_base(x, p1, p2, c))
        g_loss = NSLoss(None, f, mode="G")
        g_loss.backward()
        self.opt_g.step()

        # train D
        self.opt_d.zero_grad(set_to_none=True)
        toggle_grad(self.D, requires_grad=True)
        toggle_grad(self.G, requires_grad=False)

        p1 = p1.detach()
        p2 = p2.detach()

        r = self.D(pair_base(x, y, p1, c))
        f = self.D(pair_base(x, p1, p2, c))

        d_loss = NSLoss(r, f, mode="D")
        d_loss.backward()
        self.opt_d.step()

        self.log(f"g_adv", g_loss)
        self.log(f"dfake", f)
        self.log(f"dtrue", r)
        self.log(f"d_adv", d_loss)

        return p1

    def train_step_base(self, batch):
        """Unwraps data during training"""
        x = batch["x"]
        y = batch["y"]  # [:, 0] # lvl 0
        c = batch["c"]  # [:, 0] # lvl 0
        p = self._step_base(x, y, c)

    def train_step(self, step, batch, logger):
        """Training here, most of the actual work is in _step_base"""
        self.step = step
        self.logger = logger

        self.train_step_base(batch)

        if self.step % 10 == 0 and self.step > 50000:
            self.EMA()

        if self.step % 50000 == 0:
            self.log_network(self.G, "G")
            self.log_network(self.D, "D")

        self.logger = None

    def predict_base(self, data):
        """Never used, unclear why different from forward and discriminate_base"""
        self.eval()

        for snap in tqdm.tqdm(data.keys()):
            x = data[snap]["x"]
            c = data[snap]["c"][0]  # lvl 0

            data[snap]["y"] = data[snap]["y"][0]
            data[snap]["p"] = self.GE(x, c, noise=None).detach()  # G or GE?
        return data

    def discriminate_base(self, data):
        """Function is never used anywhere, no idea what it is actually for"""
        self.eval()

        for snap in tqdm.tqdm(data.keys()):
            x = data[snap]["x"]
            y = data[snap]["y"][0]
            c = data[snap]["c"][0]

            p1 = self.GE(x, c, noise=None).detach()
            p2 = self.GE(x, c, noise=None).detach()

            data[snap]["fakes"] = self.D(pair_base(x, p1, p2, c)).detach()
            data[snap]["reals"] = self.D(pair_base(x, y, p2, c)).detach()

            # free memory
            del data[snap]["x"]
            del data[snap]["y"]
            del data[snap]["c"]

        return data

    def get_cfg(self, mode):
        """Wraps specified args from an input config into a dict which gets unwrapped some places but used as a dict others"""
        inp_channels = len(self.config["inputs"])
        if mode == "zoom":
            inp_channels = len(self.config["outputs"])
        out_channels = len(self.config["outputs"])

        context_dim = self.config["context_dim"]
        style_dim = self.config["style_dim"]
        style_depth = self.config["style_depth"]

        nl = self.config["nl"]
        nf = self.config["nf"]
        filters = self.filter_setup(nl, nf)

        msg = "dict_of is not implemented"
        raise NotImplementedError(msg)
        # return dict_of(inp_channels, out_channels, context_dim, style_dim, style_depth, nl, filters)

    def filter_setup(self, num_layers, nf, nf_max=256):
        """Generate a list of filters to use in the model, probably because it only supports layers divisible by 2"""
        return [min(nf * 2**i, nf_max) for i in range(num_layers)]

    def log(self, key, value):
        """Sends a tensor into tensorboard with right options set"""
        self.logger.add_scalar(key, data_get(value.mean()), self.step)

    def log_network(self, model, net):
        """Logs a whole network module into tensorboard via histogram"""
        for name, value in model.named_parameters():
            if value.grad is not None:
                self.logger.add_histogram(f"{net}_grad/{name}", value.grad.cpu(), self.step)
            self.logger.add_histogram(f"{net}_params/{name}", value.cpu(), self.step)


class EMA:
    """Effective moving average data container"""

    def __init__(self, beta):
        super().__init__()
        self.beta = beta

    def update_average(self, old, new):
        if old is None:
            return new
        return old * self.beta + (1 - self.beta) * new
