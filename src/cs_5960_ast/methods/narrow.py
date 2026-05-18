"""Functions for removing edge-values from tensors

Practical use is removing padding from input/output tensors after model predictions are done
"""

import torch


def narrow_by(a: torch.Tensor, c: int) -> torch.Tensor:
    """Narrow a by size c symmetrically on all edges."""
    ind = (slice(None),) * 2 + (slice(c, -c),) * (a.dim() - 2)
    return a[ind]


def narrow_cast(*tensors: torch.Tensor) -> list[torch.Tensor]:
    """Narrow each tensor to the minimum length in each dimension.

    Try to be symmetric but cut more on the right for odd difference
    """
    dim_max = max(a.dim() for a in tensors)

    len_min = {d: min(a.shape[d] for a in tensors) for d in range(2, dim_max)}

    casted_tensors = []
    for a in tensors:
        for d in range(2, dim_max):
            width = a.shape[d] - len_min[d]
            half_width = width // 2
            a = a.narrow(d, half_width, a.shape[d] - width)

        casted_tensors.append(a)

    return casted_tensors


def narrow_like(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    """Narrow a to be like b.

    Try to be symmetric but cut more on the right for odd difference
    """
    for d in range(2, a.dim()):
        width = a.shape[d] - b.shape[d]
        half_width = width // 2
        a = a.narrow(d, half_width, a.shape[d] - width)
    return a
