import torch

# NOTE:
# dataloader and map2map system applies norm function on each input file


def blank_norm(x: torch.Tensor, *, undo: bool = False, **kwargs: float) -> None:
    """Normalization that does nothing"""
    pass


def log_norm(x: torch.Tensor, *, undo: bool = False, **kwargs: float) -> None:
    """Normalization that does nothing"""
    x.log10_()


# @torch.compile()
def dm_in_norm(x: torch.Tensor, *, undo: bool = False, **kwargs: float) -> None:
    """Separate arguments passed to norm for dm input, but use the same mathematical expression"""
    dm_norm(x, undo=undo, **kwargs)


# @torch.compile()
def dm_norm(
    x: torch.Tensor,
    *,
    eps: float = 1e-8,
    undo: bool = False,
    exponent: float = -4,
    log_mult: float = 1.0 / 4.0,
    **kwargs: float,
) -> None:
    """Normalization function

    applies by default:
    sinh[(1/4) log(10^-4 * x [unitless] + 1)]

    x: field to be modified (should be sigma_d, dark matter density)
    undo: whether to reverse normalization
    a: scale factor, passed in as style by map2map
    """
    if not undo:
        x.mul_(10 ** (exponent))
        x.add_(eps)
        x.log10_()
        x.add_(1.0)
        x.mul_(log_mult)
        x.sinh_()
    elif undo:
        # use inverse of function on x
        raise NotImplementedError


# @torch.compile()
def gas_norm(x: torch.Tensor, *, undo: bool = False, exponent: float = -3, log_mult: float = 1.0 / 4.0, **kwargs: float) -> None:
    """Normalization function

    applies by default:
    sinh[(1/4) log(10^-3 * x [unitless] + 1)]

    x: field to be modified (should be sigma_g, baryon gas density)
    undo: whether to reverse normalization
    a: scale factor, passed in as style by map2map
    """
    if not undo:
        x.mul_(10 ** (exponent))
        x.log10_()
        x.add_(1.0)
        x.mul_(log_mult)
        x.sinh_()
    elif undo:
        # use inverse of function on x
        raise NotImplementedError


# @torch.compile()
def T_norm(  # noqa: N802
    x: torch.Tensor,
    *,
    undo: bool = False,
    a: float = 1,
    exponent: float = -3,
    m_t_mult: float = -0.193,
    m_t_add: float = 5.346,
    **kwargs: float,
) -> None:
    """Normalization function

    applies by default:
    sinh[log(10^-3 * x [unitless] + 1) / m_t(z)]

    x: field to be modified (should be T, gas temperature)
    undo: whether to reverse normalization
    a: scale factor, passed in as style by map2map
    """
    z = 1 / a - 1
    m_t = m_t_mult * z + m_t_add

    if not undo:
        x.mul_(10 ** (exponent))
        x.log10_()
        x.add_(1.0)
        x.div_(m_t)
        x.sinh_()
    elif undo:
        # use inverse of function on x
        raise NotImplementedError


# @torch.compile()
def E_norm(  # noqa: N802
    x: torch.Tensor,
    *,
    undo: bool = False,
    a: float = 1,
    exponent: float = -3,
    m_t_mult: float = -0.193,
    m_t_add: float = 5.346,
    **kwargs: float,
) -> None:
    """Normalization function

    applies by default:
    sinh[log(10^-3 * x [unitless] + 1) / m_t(z)]

    x: field to be modified (should be E, gas energy)
    undo: whether to reverse normalization
    a: scale factor, passed in as style by map2map
    """
    z = 1 / a - 1
    m_t = m_t_mult * z + m_t_add

    if not undo:
        x.mul_(10 ** (exponent))
        x.log10_()
        x.add_(1.0)
        x.div_(m_t)
        x.sinh_()
    elif undo:
        # use inverse of function on x
        raise NotImplementedError


# @torch.compile()
def generic_norm(
    x: torch.Tensor,
    *,
    eps: float = 1e-8,
    undo: bool = False,
    k: float = 3.0,
    x_0: float = 10**7,
    q: float = 1.0,
    **kwargs: float,
) -> None:
    """Generic normalization function from https://arxiv.org/pdf/2110.11970

    applies:
    x_modified = 1/k log_10[(x/x_0)^q + 1]

    x: field to be modified
    undo: whether to reverse normalization
    a: scale factor, passed in as style by map2map
    k: free parameter
    q: free parameter
    x_0: free parameter presumably based on input unit scale (involves solar mass)
    """
    if not undo:
        # modify x in place
        x.div_(x_0)
        x.pow_(q)
        x.add_(1.0 + eps)  # should fix log(0) problems
        x.log10_()
        x.mul_(1.0 / k)
    elif undo:
        # use inverse of function on x
        raise NotImplementedError


# @torch.compile()
def sigma_unit() -> float:
    """Correct to SI units"""
    solar_mass: float = 1.988_416 * 10**30  # [kg]
    c: float = 299_792_458  # [m/s]
    kpc: float = 1000 * 3.0857 * 10**16  # [m]
    return solar_mass / (c * kpc**2)
