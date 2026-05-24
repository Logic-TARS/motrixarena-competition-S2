# Copyright (C) 2020-2025 Motphys Technology Co., Ltd. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================

import dataclasses
import copy
from dataclasses import dataclass
from typing import Any, TypeVar

T = TypeVar("T")


@dataclass
class DeviceSupports:
    torch: bool = False
    torch_gpu: bool = False
    jax: bool = False
    jax_gpu: bool = False


def _check_gpu_available_for_torch():
    try:
        import torch

        if not torch.cuda.is_available():
            return False
        torch.zeros((1,)).cuda().numpy(force=True)
        return True
    except Exception:
        return False


def get_device_supports() -> DeviceSupports:
    supports = DeviceSupports()
    try:
        import torch  # noqa: F401

        supports.torch = True
        supports.torch_gpu = _check_gpu_available_for_torch()
    except ImportError:
        pass

    try:
        import jax  # noqa: F401

        supports.jax = True
        from jax.lib import xla_bridge

        platform = xla_bridge.get_backend().platform
        if platform == "gpu":
            supports.jax_gpu = True
    except ImportError:
        pass

    return supports


def class_to_dict(obj) -> dict | list | Any:
    """Recursively convert a dataclass to a dictionary.

    Args:
        obj: The object to convert (dataclass, list, dict, or primitive)

    Returns:
        Dictionary representation with nested dataclasses recursively converted
    """
    if dataclasses.is_dataclass(obj):
        return {k: class_to_dict(v) for k, v in dataclasses.asdict(obj).items()}
    elif isinstance(obj, list):
        return [class_to_dict(item) for item in obj]
    elif isinstance(obj, dict):
        return {k: class_to_dict(v) for k, v in obj.items()}
    else:
        return obj


def cfg_override(cfg: T, overrides: dict[str, Any]) -> T:
    """Override dataclass fields using dot-notation path keys.

    This function creates a new dataclass instance with specified field values
    overridden, leaving the original config unchanged. Nested dataclasses are
    handled using dot notation in the key path.

    Args:
        cfg: The original dataclass configuration object
        overrides: Dictionary with path keys (e.g., "runner.seed", "num_envs")
                   where each key is a dot-separated path to the field to override

    Returns:
        A new dataclass instance with overrides applied

    Raises:
        KeyError: If a path key is invalid or references a non-existent field
        TypeError: If an intermediate field is not a dataclass or if a value
                   type doesn't match the expected field type

    Examples:
        >>> from motrix_rl.rslrl.cfg import RslrlCfg
        >>> base_cfg = RslrlCfg()
        >>> overrides = {
        ...     "num_envs": 4096,
        ...     "runner.seed": 123,
        ...     "runner.algorithm.num_learning_epochs": 10,
        ... }
        >>> new_cfg = cfg_override(base_cfg, overrides)
        >>> assert new_cfg.num_envs == 4096
        >>> assert new_cfg.runner.seed == 123
        >>> assert new_cfg.runner.algorithm.num_learning_epochs == 10
    """
    if not overrides:
        return cfg

    if not dataclasses.is_dataclass(cfg):
        raise TypeError(f"cfg must be a dataclass, got {type(cfg).__name__}")

    result = copy.deepcopy(cfg)

    for key, value in overrides.items():
        parts = key.split(".")
        obj = result
        for part in parts[:-1]:
            if not dataclasses.is_dataclass(obj):
                raise TypeError(f"Cannot navigate into non-dataclass field '{part}' of type {type(obj).__name__}")
            obj_fields = {f.name for f in dataclasses.fields(obj)}
            if part not in obj_fields:
                raise KeyError(f"Invalid path component '{part}' for {type(obj).__name__}. Valid fields: {sorted(obj_fields)}")
            obj = getattr(obj, part)

        field_name = parts[-1]
        if not dataclasses.is_dataclass(obj):
            raise TypeError(f"Cannot override non-dataclass object at '{key}' of type {type(obj).__name__}")
        obj_fields = {f.name for f in dataclasses.fields(obj)}
        if field_name not in obj_fields:
            raise KeyError(f"Invalid field '{field_name}' for {type(obj).__name__}. Valid fields: {sorted(obj_fields)}")
        setattr(obj, field_name, value)

    return result
