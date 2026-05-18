import importlib
import importlib.util
import os
import sys
from functools import cache


@cache
def import_attr(name, *pkgs, callback_at=None):
    """Import attribute. Try package first and then callback directory.

    To use a callback, `name` must contain a module, formatted as 'mod.attr'.

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
            msg = "callback file not found"
            raise FileNotFoundError(msg)

        if mod in sys.modules:
            return getattr(sys.modules[mod], attr)

        spec = importlib.util.spec_from_file_location(mod, callback_at)
        if spec is None:
            msg = "failed to create module spec"
            raise AttributeError(msg)
        else:
            module = importlib.util.module_from_spec(spec)
            sys.modules[mod] = module
            spec.loader.exec_module(module)

        return getattr(module, attr)
