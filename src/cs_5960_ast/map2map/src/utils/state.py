import sys
import warnings
from pprint import pformat
from typing import Any

import torch


def load_model_state_dict(module: torch.nn.Module, state_dict: dict[str, Any], *, strict: bool = True) -> None:
    """Wrapper method for pytorch `module.load_state_dict()` method

    Will output a warning on any missing or unexpected keys, as reported by pytorch
    """
    bad_keys = module.load_state_dict(state_dict, strict)

    if len(bad_keys.missing_keys) > 0:
        warnings.warn(f"Missing keys in state_dict:\n{pformat(bad_keys.missing_keys)}", stacklevel=2)
    if len(bad_keys.unexpected_keys) > 0:
        warnings.warn(f"Unexpected keys in state_dict:\n{pformat(bad_keys.unexpected_keys)}", stacklevel=2)
    sys.stderr.flush()
