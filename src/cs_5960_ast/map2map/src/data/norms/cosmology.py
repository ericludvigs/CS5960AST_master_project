import numpy as np
import torch
from scipy.special import hyp2f1


def dis(
    x: float | torch.Tensor | np.ndarray, *, undo: bool = False, a: float = 0.3333, dis_std: float = 6000.0, **kwargs: dict
) -> None:
    z = 1 / a - 1
    dis_norm = dis_std * D(z)  # [Kpc/h]

    if not undo:
        dis_norm = 1 / dis_norm

    x *= dis_norm


def vel(
    x: float | torch.Tensor | np.ndarray, *, undo: bool = False, a: float = 0.3333, dis_std: float = 6.0, **kwargs: dict
) -> None:
    z = 1 / a - 1
    vel_norm = dis_std * D(z) * H(z) * f(z) / (1 + z)  # [km/s]

    if not undo:
        vel_norm = 1 / vel_norm

    x *= vel_norm


def D(z: float, Om: float = 0.31) -> float:
    """Linear growth function for flat LambdaCDM, normalized to 1 at redshift zero"""
    OL = 1 - Om
    a = 1 / (1 + z)
    return a * hyp2f1(1, 1 / 3, 11 / 6, -OL * a**3 / Om) / hyp2f1(1, 1 / 3, 11 / 6, -OL / Om)


def f(z: float, Om: float = 0.31) -> float:
    """Linear growth rate for flat LambdaCDM"""
    OL = 1 - Om
    a = 1 / (1 + z)
    aa3 = OL * a**3 / Om
    return 1 - 6 / 11 * aa3 * hyp2f1(2, 4 / 3, 17 / 6, -aa3) / hyp2f1(1, 1 / 3, 11 / 6, -aa3)


def H(z: float, Om: float = 0.31) -> float:
    """Hubble in [h km/s/Mpc] for flat LambdaCDM"""
    OL = 1 - Om
    a = 1 / (1 + z)
    return 100 * np.sqrt(Om / a**3 + OL)
