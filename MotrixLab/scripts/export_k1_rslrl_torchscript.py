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

import argparse
from pathlib import Path

from _source_path import ensure_source_path

ensure_source_path()

import torch
from rsl_rl.runners import OnPolicyRunner

from motrix_envs import registry as env_registry
from motrix_rl import registry as rl_registry
from motrix_rl.rslrl.torch.wrap_vec_env import RslrlNpEnvWrap


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Export an RSLRL K1 checkpoint to a single-input/single-output 47->12 TorchScript policy."
    )
    parser.add_argument("checkpoint", type=Path, help="RSLRL checkpoint, for example runs/.../model_199.pt")
    parser.add_argument(
        "-o",
        "--output",
        type=Path,
        default=Path("exported/k1_flat_terrain_walk_torchscript.pt"),
        help="Output TorchScript path.",
    )
    parser.add_argument("--env", default="k1-flat-terrain-walk", help="MotrixLab env name.")
    parser.add_argument("--device", default="cpu", help="Export device, usually cpu.")
    return parser.parse_args()


def _check_shape(output: torch.Tensor, num_actions: int) -> None:
    if tuple(output.shape) != (1, num_actions):
        raise RuntimeError(f"Unexpected exported policy shape {tuple(output.shape)}")


def _export_torchscript(actor: torch.nn.Module, output_path: Path, num_obs: int, num_actions: int) -> None:
    exported = actor.as_jit().cpu().eval()
    example = torch.zeros(1, num_obs, dtype=torch.float32)
    with torch.no_grad():
        scripted = torch.jit.script(exported)
        _check_shape(scripted(example), num_actions)
    scripted.save(str(output_path))


def main() -> None:
    args = parse_args()
    device = torch.device(args.device)

    env = env_registry.make(args.env, sim_backend="np", num_envs=1)
    vec_env = RslrlNpEnvWrap(env, device)
    rlcfg = rl_registry.default_rl_cfg(args.env, "rslrl", backend="torch")
    runner = OnPolicyRunner(vec_env, rlcfg.runner.to_dict(), log_dir=None, device=device)
    runner.load(str(args.checkpoint))

    if vec_env.num_obs != 47 or vec_env.num_actions != 12:
        raise RuntimeError(f"K1 G1-style runtime expects 47->12, got {vec_env.num_obs}->{vec_env.num_actions}")

    args.output.parent.mkdir(parents=True, exist_ok=True)
    actor = runner.alg.actor
    if getattr(actor, "is_recurrent", False):
        raise RuntimeError("Submission-safe export requires a non-recurrent MLP actor")
    _export_torchscript(actor, args.output, vec_env.num_obs, vec_env.num_actions)
    print(f"Exported {args.output} (torchscript, {vec_env.num_obs}->{vec_env.num_actions})")


if __name__ == "__main__":
    main()
