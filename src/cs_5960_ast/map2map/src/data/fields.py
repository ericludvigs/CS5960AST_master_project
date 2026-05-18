import os
import re
from glob import glob
from pathlib import Path
from pprint import pformat
from typing import Any, TypedDict

import numpy as np
import torch
from torch.utils.data import Dataset, WeightedRandomSampler

from cs_5960_ast.map2map.src.data import norms
from cs_5960_ast.map2map.src.utils import import_attr


class FieldDataOutputItem(TypedDict):
    """For properly type annotating the Dataset output items"""

    input: torch.Tensor
    target: torch.Tensor
    style: torch.Tensor
    input_relpath: list[str]
    target_relpath: list[str]


class FieldDataset(Dataset[FieldDataOutputItem]):
    """Dataset of lists of fields.

    `in_patterns` is a list of glob patterns for the input field files.
    For example, `in_patterns=['/train/field1_*.npy', '/train/field2_*.npy']`.
    Each pattern in the list is a new field.
    Likewise `tgt_patterns` is for target fields.
    Input and target fields are matched by sorting the globbed files.

    `in_norms` is a list of of functions to normalize the input fields.
    Likewise for `tgt_norms`.

    NOTE that vector fields are assumed if numbers of channels and dimensions are equal.

    Scalar and vector fields can be augmented by flipping and permutating the axes.
    In 3D these form the full octahedral symmetry, the Oh group of order 48.
    In 2D this is the dihedral group D4 of order 8.
    1D is not supported, but can be done easily by preprocessing.
    Fields can be augmented by random shift by a few pixels, useful for models
    that treat neighboring pixels differently, e.g. with strided convolutions.
    Additive and multiplicative augmentation are also possible, but with all fields
    added or multiplied by the same factor.

    Input and target fields can be cropped, to return multiple slices of size
    `crop` from each field.
    The crop anchors are controlled by `crop_start`, `crop_stop`, and `crop_step`.
    Input (but not target) fields can be padded beyond the crop size assuming
    periodic boundary condition.

    Setting integer `scale_factor` greater than 1 will crop target bigger than
    the input for super-resolution, in which case `crop` and `pad` are sizes of
    the input resolution.

    Parameter `norm_config` is a dictionary indexed by the norm function name (in `in_norms` and `tgt_norms`).
    It should have keyword arguments (like `exponent=-3`) as a dict which gets unwrapped and passed to norm function,
    allowing configuration.
    """

    def __init__(
        self,
        in_patterns: list[str],
        tgt_patterns: list[str],
        *,
        style_pattern: str | None = None,
        in_filter_patterns: list[str] | None = None,
        tgt_filter_patterns: list[str] | None = None,
        style_filter_pattern: str | None = None,
        in_norms: list[str] | None = None,
        tgt_norms: list[str] | None = None,
        norm_config: dict[str, dict[str, float]] | None = None,
        callback_at: str | Path | None = None,
        augment: bool | None = False,
        aug_shift: int | tuple[int, ...] | None = None,
        aug_add: float | None = None,
        aug_mul: float | None = None,
        crop: int | tuple[int, ...] | None = None,
        crop_start: int | tuple[int, ...] | None = None,
        crop_stop: int | tuple[int, ...] | None = None,
        crop_step: int | tuple[int, ...] | None = None,
        in_pad: int | tuple[int, ...] = 0,
        tgt_pad: int | tuple[int, ...] = 0,
        scale_factor: float = 1,
        dataset_name: str | None = None,
        **kwargs: str | float | list | dict | Path | None,
    ) -> None:
        # you can adjust the random seed to randomize the in pattern sequence
        sampling = True

        # early error for wrong inputs
        if isinstance(in_patterns, str) or isinstance(tgt_patterns, str):
            msg = "file pattern input must be a listlike and not directly a string (str itself being iterable breaks further code)"
            raise TypeError(msg)

        in_file_lists: list[list[str]] = [sorted(glob(p)) for p in in_patterns]  # noqa: PTH207
        # path.glob is not a simple replacement here
        # for one, absolute paths don't work the same

        # each pattern sent in is one field (technically can be a single channel from a bigger field)
        # we have a list of files for each field via pattern (makes a list of lists)

        # further filtering functionality - supports a regex pattern
        # returns a subset of the file list for each field, with only the files that match the pattern
        # `for x in in_filter_patterns or []` is equivalent to `in_filter_patterns if in_filter_patterns is not None else []`
        # blank list skips for loop
        for i, regex_filter_pattern in enumerate(in_filter_patterns or []):
            in_file_lists[i] = list(filter(re.compile(regex_filter_pattern).match, in_file_lists[i]))

        num_snapshots: int = len(in_file_lists[0])
        if num_snapshots == 0:
            msg = f"Did not find any files for input patterns {in_patterns} after filtering with patterns {in_filter_patterns}"
            raise FileNotFoundError(msg)
        sample_idx: list[int] = list(
            WeightedRandomSampler(
                weights=torch.ones(num_snapshots),  # ty: ignore false positive, torch tensor works in pytorch method
                num_samples=min(1000, num_snapshots),  # selects a maximum of 1000 random snapshots to work with
                replacement=False,
            )
        )

        if sampling:
            sampled_in_list: list[list[str]] = [[x[i] for i in sample_idx] for x in in_file_lists]
            self.in_files: list[str] = list(zip(*sampled_in_list, strict=True))
        else:
            self.in_files: list[str] = list(zip(*in_file_lists, strict=True))

        tgt_file_lists: list[list[str]] = [sorted(glob(p)) for p in tgt_patterns]  # noqa: PTH207 same as above

        for i, regex_filter_pattern in enumerate(tgt_filter_patterns or []):
            tgt_file_lists[i] = list(filter(re.compile(regex_filter_pattern).match, tgt_file_lists[i]))

        if sampling:
            sampled_tgt_list: list[list[str]] = [[x[i] for i in sample_idx] for x in tgt_file_lists]
            self.tgt_files: list[str] = list(zip(*sampled_tgt_list, strict=True))
        else:
            self.tgt_files: list[str] = list(zip(*tgt_file_lists, strict=True))

        if len(self.in_files) != len(self.tgt_files):
            msg = "number of input and target fields do not match"
            raise ValueError(msg)
        self.nfile: int = len(self.in_files)

        if self.nfile == 0:
            msg = f"file not found for {in_patterns}"
            raise FileNotFoundError(msg)
        self.is_read_once: np.ndarray = np.full(shape=self.nfile, fill_value=False)

        self.in_chan: list[np.ndarray] = [np.load(f, mmap_mode="r").shape[0] for f in self.in_files[0]]
        self.tgt_chan: list[np.ndarray] = [np.load(f, mmap_mode="r").shape[0] for f in self.tgt_files[0]]

        self.size: np.ndarray = np.load(self.in_files[0][0], mmap_mode="r").shape[1:]
        self.size: np.ndarray = np.asarray(self.size)
        self.ndim: int = len(self.size)

        self.style: bool = style_pattern is not None
        self.style_size: int = 0
        if style_pattern is not None:  # type checker can't recognize self.style as constraining style_pattern
            style_files: list[str] = sorted(glob(style_pattern))  # noqa: PTH207 path.glob is not a viable replacement here

            # remove style files that do not match the pattern provided (if there is one)
            if style_filter_pattern is not None:
                style_files: list[str] = list(filter(re.compile(style_filter_pattern).match, style_files))  # ty:ignore[no-matching-overload] this filter works fine

            if sampling:
                sampled_style_list: list[str] = [style_files[i] for i in sample_idx]
                self.style_files: list[str] = sampled_style_list
            else:
                self.style_files: list[str] = style_files

            if len(self.style_files) != len(self.in_files):
                msg = "number of style and input files do not match"
                raise ValueError(msg)
            self.style_size: int = np.load(self.style_files[0]).shape[0]

        if in_norms is not None and len(in_patterns) != len(in_norms):
            msg = "numbers of input normalization functions and fields do not match"
            raise ValueError(msg)
        self.in_norms = in_norms

        if tgt_norms is not None and len(tgt_patterns) != len(tgt_norms):
            msg = "numbers of target normalization functions and fields do not match"
            raise ValueError(msg)
        self.tgt_norms = tgt_norms

        self.callback_at = callback_at

        self.augment = augment
        if self.ndim == 1 and self.augment:
            msg = "cannot augment 1D fields"
            raise ValueError(msg)
        if aug_shift is not None:
            self.aug_shift = np.broadcast_to(aug_shift, shape=(self.ndim,))
        else:
            self.aug_shift: np.ndarray = np.broadcast_to(aug_shift, shape=(self.ndim,))  # ty:ignore[no-matching-overload]
            # numpy did not document that broadcast_to handles None just fine and makes an array of Nones
            # so just explicitly specifying that this works here
        self.aug_add = aug_add
        self.aug_mul = aug_mul

        if crop is None:
            self.crop = self.size
        else:
            self.crop = np.broadcast_to(crop, (self.ndim,))

        if crop_start is None:
            crop_start: np.ndarray = np.zeros_like(self.size)
        else:
            crop_start: np.ndarray = np.broadcast_to(crop_start, (self.ndim,))

        if crop_stop is None:
            crop_stop: np.ndarray = self.size
        else:
            crop_stop: np.ndarray = np.broadcast_to(crop_stop, (self.ndim,))

        if crop_step is None:
            crop_step: np.ndarray = self.crop
        else:
            crop_step: np.ndarray = np.broadcast_to(crop_step, (self.ndim,))
        self.crop_step = crop_step

        anchors_input: np.ndarray = np.mgrid[tuple(slice(crop_start[d], crop_stop[d], crop_step[d]) for d in range(self.ndim))]
        self.anchors: np.ndarray = np.stack(arrays=list(anchors_input), axis=-1).reshape(-1, self.ndim)
        self.ncrop = len(self.anchors)

        def format_pad(pad: int | tuple[int, ...], ndim: int) -> np.ndarray:
            if isinstance(pad, int):
                pad: np.ndarray = np.broadcast_to(pad, ndim * 2)
            elif isinstance(pad, tuple) and len(pad) == ndim:
                pad: np.ndarray = np.repeat(pad, 2)
            elif isinstance(pad, tuple) and len(pad) == ndim * 2:
                pad: np.ndarray = np.array(pad)
            else:
                msg = "pad and ndim mismatch"
                raise ValueError(msg)
            return pad.reshape(ndim, 2)

        self.in_pad: np.ndarray = format_pad(in_pad, self.ndim)
        self.tgt_pad: np.ndarray = format_pad(tgt_pad, self.ndim)

        if scale_factor != 1:
            tgt_size = np.load(self.tgt_files[0][0], mmap_mode="r").shape[1:]
            if any(self.size * scale_factor != tgt_size):
                msg = "input size x scale factor != target size"
                raise ValueError(msg)
        self.scale_factor = scale_factor

        self.nsample = self.nfile * self.ncrop

        self.kwargs = kwargs

        self.norm_config = norm_config

        self.assembly_line: dict[str, Any] = {}

        self.commonpath = os.path.commonpath(file for files in self.in_files[:2] + self.tgt_files[:2] for file in files)

        local_rank = int(os.getenv("LOCAL_RANK", default="-1"))
        if local_rank == 0:
            output_dir = Path(os.getenv("OUTPUT_DIR", default=Path.cwd()))
            datafile_log_filename = os.getenv("DATAFILES_LOG_FILENAME", default="datafiles_info.log")
            if not (output_dir / datafile_log_filename).exists():
                msg = f"Expected log file not found at {output_dir / datafile_log_filename}, must be created before dataset init"
                raise FileNotFoundError(msg)

            file_str = ""
            if dataset_name is not None:
                file_str += f"Dataset: {dataset_name}\n\n"

            input_files_info: dict[str, int | dict] = {
                "1_total_num_fields_in": len(self.in_files),
                "2_num_files_per_pattern": {},
                "3_all_files_used": {},
            }
            for i, pattern in enumerate(in_patterns):
                input_files_info["2_num_files_per_pattern"][Path(pattern).name] = len(
                    sampled_in_list[i] if sampling else in_file_lists[i]
                )
                input_files_info["3_all_files_used"][Path(pattern).name] = [
                    Path(filepath).name for filepath in (sampled_in_list[i] if sampling else in_file_lists[i])
                ]
            file_str += f"Input files info:\n{pformat(input_files_info, indent=1, underscore_numbers=True)}\n\n"

            tgt_files_info: dict[str, int | dict] = {
                "1_total_num_fields_tgt": len(self.tgt_files),
                "2_num_files_per_pattern": {},
                "3_all_files_used": {},
            }
            for i, pattern in enumerate(tgt_patterns):
                tgt_files_info["2_num_files_per_pattern"][Path(pattern).name] = len(
                    sampled_tgt_list[i] if sampling else tgt_file_lists[i]
                )
                tgt_files_info["3_all_files_used"][Path(pattern).name] = [
                    Path(filepath).parent.name + "/" + Path(filepath).name  # the final folder in the path + the filename
                    for filepath in (sampled_tgt_list[i] if sampling else tgt_file_lists[i])
                ]
                # done this way in order to list train/filename.npy, test/filename.npy and be sure that the set is correct
            file_str += f"Target files info:\n{pformat(tgt_files_info, indent=1, underscore_numbers=True)}\n\n"

            file_str += f"Total number of style files: {len(self.style_files) if self.style else 0}\n\n"

            Path.open(output_dir / datafile_log_filename, "a").write(file_str)

    def __len__(self) -> int:
        return self.nsample

    def __getitem__(self, index: int) -> FieldDataOutputItem:
        ifile, icrop = divmod(index, self.ncrop)

        # use memmap after reading a file once
        if self.is_read_once[ifile]:
            mmap_mode = "r"
        else:
            mmap_mode = None
            self.is_read_once[ifile] = True

        in_fields: list[np.ndarray] = [np.load(f, mmap_mode=mmap_mode) for f in self.in_files[ifile]]
        tgt_fields: list[np.ndarray] = [np.load(f, mmap_mode=mmap_mode) for f in self.tgt_files[ifile]]

        anchor: np.ndarray = self.anchors[icrop]

        for d, shift in enumerate(self.aug_shift):
            if shift is not None:
                anchor[d] += torch.randint(int(shift), (1,))

        # crop and pad are for the shapes after perm()
        # so before that they themselves need perm() in the opposite ways
        if self.augment:
            # let i and j index axes before and after perm()
            # then perm_axes is i_j, whose argsort is j_i
            # the latter is needed to index crop and pad for opposite perm()
            perm_axes = perm([], None, self.ndim)
            argsort_perm_axes = np.argsort(perm_axes.numpy())
        else:
            argsort_perm_axes = slice(None)

        crop(
            fields=in_fields,
            anchor=anchor,
            crop=self.crop[argsort_perm_axes],
            pad=self.in_pad[argsort_perm_axes],
        )
        crop(
            fields=tgt_fields,
            anchor=anchor * self.scale_factor,
            crop=self.crop[argsort_perm_axes] * self.scale_factor,
            pad=self.tgt_pad[argsort_perm_axes],
        )

        in_fields: list[torch.Tensor] = [torch.from_numpy(f.astype(np.float32)) for f in in_fields]
        tgt_fields: list[torch.Tensor] = [torch.from_numpy(f.astype(np.float32)) for f in tgt_fields]

        style = torch.empty(0, dtype=torch.float32)
        if self.style:
            style = np.load(self.style_files[ifile])
            style = torch.from_numpy(style.astype(np.float32))
        # print("field while loading files", style.shape)

        if self.in_norms is not None:
            for norm, x in zip(self.in_norms, in_fields, strict=True):
                if self.norm_config is not None:
                    norm_args: dict[str, float] = self.norm_config[norm]
                else:
                    norm_args: dict = {}
                norm = import_attr(norm, norms, callback_at=self.callback_at)
                # print(f"keyword args for norm {norm}: {norm_args}")
                # print(f"norming input tensor with shape: {x.shape}")
                norm(x, a=style[0], **norm_args, **self.kwargs)
        if self.tgt_norms is not None:
            for norm, x in zip(self.tgt_norms, tgt_fields, strict=True):
                if self.norm_config is not None:
                    norm_args: dict[str, float] = self.norm_config[norm]
                else:
                    norm_args: dict = {}
                norm = import_attr(norm, norms, callback_at=self.callback_at)
                # print(f"keyword args for norm {norm}: {norm_args}")
                # print(f"norming target tensor with shape: {x.shape}")
                norm(x, a=style[0], **norm_args, **self.kwargs)

        if self.augment:
            flip_axes = flip(in_fields, None, self.ndim)
            flip_axes = flip(tgt_fields, flip_axes, self.ndim)

            perm_axes = perm(in_fields, perm_axes, self.ndim)
            perm_axes = perm(tgt_fields, perm_axes, self.ndim)

        if self.aug_add is not None:
            add_fac = add(in_fields, None, self.aug_add)
            add_fac = add(tgt_fields, add_fac, self.aug_add)

        if self.aug_mul is not None:
            mul_fac = mul(in_fields, None, self.aug_mul)
            mul_fac = mul(tgt_fields, mul_fac, self.aug_mul)

        in_fields: torch.Tensor = torch.cat(in_fields, dim=0)
        tgt_fields: torch.Tensor = torch.cat(tgt_fields, dim=0)

        in_relpath: list[str] = [os.path.relpath(file, start=self.commonpath) for file in self.in_files[ifile]]
        tgt_relpath: list[str] = [os.path.relpath(file, start=self.commonpath) for file in self.tgt_files[ifile]]

        output_item: FieldDataOutputItem = {
            "input": in_fields,
            "target": tgt_fields,
            "style": style,
            "input_relpath": in_relpath,
            "target_relpath": tgt_relpath,
        }
        return output_item

    def assemble(self, label: str, chan: list[int], patches: np.ndarray | torch.Tensor, paths: list[list[str]]) -> None:
        """Assemble and write whole fields from patches, each being the end result from a cropped field by `__getitem__`.

        Repeat feeding spatially ordered field patches.
        After filled, the whole fields are assembled and saved to relative
        paths specified by `paths` and `label`.
        `chan` is used to split the channels to undo `cat` in `__getitem__`.

        As an example, patches of shape `(1, 4, X, Y, Z)`, `label='_out'`
        and `chan=[1, 3]`, with `paths=[['d/scalar.npy'], ['d/vector.npy']]`
        will write to `'d/scalar_out.npy'` and `'d/vector_out.npy'`.

        Note that `paths` assumes transposed shape due to pytorch auto batching
        """
        if self.scale_factor != 1:
            raise NotImplementedError

        if isinstance(patches, torch.Tensor):
            patches = patches.detach().cpu().numpy()

        if patches.ndim != 2 + self.ndim:
            msg = f"ndim mismatch: {patches.ndim, 2 + self.ndim}"
            raise RuntimeError(msg)
        if any(self.crop_step > patches.shape[2:]):
            msg = "patch too small to tile"
            raise RuntimeError(msg)

        # the batched paths are a list of lists with shape (channel, batch)
        # since pytorch default_collate batches list of strings transposedly
        # therefore we transpose below back to (batch, channel)
        if patches.shape[1] != sum(chan):
            msg = f"number of channels mismatch: {patches.shape[1], sum(chan)}"
            raise RuntimeError(msg)
        if len(paths) != len(chan):
            msg = f"number of fields mismatch: {len(paths), len(chan)}"
            raise RuntimeError(msg)
        paths = list(zip(*paths, strict=False))
        if patches.shape[0] != len(paths):
            msg = f"batch size mismatch: {patches.shape[0], len(paths)}"
            raise RuntimeError(msg)

        patches: list[np.ndarray | torch.Tensor] = list(patches)
        if label in self.assembly_line:
            self.assembly_line[label] += patches
            self.assembly_line[label + "path"] += paths
        else:
            self.assembly_line[label] = patches
            self.assembly_line[label + "path"] = paths

        del patches, paths
        patches = self.assembly_line[label]
        paths = self.assembly_line[label + "path"]

        # NOTE anchor positioning assumes sufficient target padding and
        # symmetric narrowing (more on the right if odd) see `models/narrow.py`
        narrow = self.crop + self.tgt_pad.sum(axis=1) - patches[0].shape[1:]
        anchors = self.anchors - self.tgt_pad[:, 0] + narrow // 2

        while len(patches) >= self.ncrop:
            fields: np.ndarray = np.zeros(patches[0].shape[:1] + tuple(self.size), patches[0].dtype)

            for patch, anchor in zip(patches, anchors, strict=True):
                fill(fields, patch, anchor)

            for field, path in zip(np.split(fields, np.cumsum(chan), axis=0), paths[0], strict=True):
                (Path(path).parent).mkdir(parents=True, exist_ok=True)

                path = label.join(os.path.splitext(path))  # noqa: PTH122
                # pathlib can't (easily) replicate string manipulation being done here
                np.save(path, field)

            del patches[: self.ncrop], paths[: self.ncrop]


def fill(field: np.ndarray, patch: np.ndarray | torch.Tensor, anchor: np.ndarray) -> None:
    ndim = len(anchor)
    if not field.ndim == patch.ndim == 1 + ndim:
        msg = f"ndim mismatch: {field.ndim, patch.ndim, 1 + ndim}"
        raise RuntimeError(msg)

    ind = [slice(None)]
    for d, (p, a, s) in enumerate(zip(patch.shape[1:], anchor, field.shape[1:], strict=True)):
        i: np.ndarray = np.arange(a, a + p)
        i %= s
        i: np.ndarray = i.reshape((-1,) + (1,) * (ndim - d - 1))
        ind.append(i)
    ind: tuple = tuple(ind)

    field[ind] = patch


def crop(fields: list[np.ndarray], anchor: np.ndarray, crop: np.ndarray, pad: np.ndarray) -> tuple[slice, ...]:
    if any(x.shape[1:] != fields[0].shape[1:] for x in fields[1:]):
        msg = f"shape mismatch: {[x.shape[1:] for x in fields]}"
        raise RuntimeError(msg)
    size = fields[0].shape[1:]
    ndim = len(size)
    if not ndim == len(anchor) == len(crop) == len(pad):
        msg = f"ndim mismatch: {ndim, len(anchor), len(crop), len(pad)}"
        raise RuntimeError(msg)

    ind = [slice(None)]
    for d, (a, c, (p0, p1), s) in enumerate(zip(anchor, crop, pad, size, strict=True)):
        i = np.arange(a - p0, a + c + p1)
        i %= s
        i = i.reshape((-1,) + (1,) * (ndim - d - 1))
        ind.append(i)
    ind = tuple(ind)

    for i, x in enumerate(fields):
        x = x[ind]

        fields[i] = x

    return ind


def flip(fields: list[torch.Tensor], axes: None | torch.Tensor, ndim: int) -> torch.Tensor:
    if ndim == 1:
        msg = "flipping is ambiguous for 1D scalars/vectors"
        raise RuntimeError(msg)

    if axes is None:
        axes: torch.Tensor = torch.randint(2, (ndim,), dtype=torch.bool)
        axes: torch.Tensor = torch.arange(ndim)[axes]

    for i, x in enumerate(fields):
        if x.shape[0] == ndim:  # flip vector components
            x[axes] = -x[axes]

        shifted_axes = (1 + axes).tolist()
        x = torch.flip(x, shifted_axes)

        fields[i] = x

    return axes


def perm(fields: list[torch.Tensor], axes: torch.Tensor | None, ndim: int) -> torch.Tensor:
    if ndim == 1:
        msg = "permutation is not necessary for 1D fields"
        raise RuntimeError(msg)

    if axes is None:
        axes = torch.randperm(ndim)

    for i, x in enumerate(fields):
        if x.shape[0] == ndim:  # permutate vector components
            x = x[axes]

        shifted_axes = [0] + (1 + axes).tolist()  # noqa: RUF005 preserve map2map implementation
        x = x.permute(shifted_axes)

        fields[i] = x

    return axes


def add(fields: list[torch.Tensor], fac: torch.Tensor | None, std: float) -> torch.Tensor:
    if fac is None:
        x = fields[0]
        fac = torch.zeros((x.shape[0],) + (1,) * (x.dim() - 1))
        fac.normal_(mean=0, std=std)

    for x in fields:
        x += fac

    return fac


def mul(fields: list[torch.Tensor], fac: torch.Tensor | None, std: float) -> torch.Tensor:
    if fac is None:
        x = fields[0]
        fac = torch.ones((x.shape[0],) + (1,) * (x.dim() - 1))
        fac.log_normal_(mean=0, std=std)

    for x in fields:
        x *= fac

    return fac
