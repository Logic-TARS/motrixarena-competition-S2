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

import torch
from rsl_rl.runners import OnPolicyRunner
from tensordict import TensorDict

from motrix_envs import registry as env_registry
from motrix_rl import registry as rl_registry
from motrix_rl.rslrl.torch.wrap_vec_env import RslrlNpEnvWrap


class TensorPolicyWrapper(torch.nn.Module):
    def __init__(self, policy: torch.nn.Module):
        super().__init__()
        self.policy = policy

    def forward(self, obs: torch.Tensor) -> torch.Tensor:
        td = TensorDict({"policy": obs}, batch_size=[obs.shape[0]], device=obs.device)
        return self.policy(td)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Export an RSLRL K1 checkpoint to a 52->12 TorchScript policy."
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


def main() -> None:
    args = parse_args()
    device = torch.device(args.device)

    env = env_registry.make(args.env, sim_backend="np", num_envs=1)
    vec_env = RslrlNpEnvWrap(env, device)
    rlcfg = rl_registry.default_rl_cfg(args.env, "rslrl", backend="torch")
    runner = OnPolicyRunner(vec_env, rlcfg.runner.to_dict(), log_dir=None, device=device)
    runner.load(str(args.checkpoint))

    policy = runner.get_inference_policy(device=device)
    wrapped = TensorPolicyWrapper(policy).to(device).eval()
    example = torch.zeros(1, vec_env.num_obs, dtype=torch.float32, device=device)
    with torch.no_grad():
        traced = torch.jit.trace(wrapped, example, strict=False)
        output = traced(example)

    if tuple(output.shape) != (1, vec_env.num_actions):
        raise RuntimeError(f"Unexpected exported policy shape {tuple(output.shape)}")
    if vec_env.num_obs != 52 or vec_env.num_actions != 12:
        raise RuntimeError(f"K1 runtime expects 52->12, got {vec_env.num_obs}->{vec_env.num_actions}")

    args.output.parent.mkdir(parents=True, exist_ok=True)
    traced.save(str(args.output))
    print(f"Exported {args.output} ({vec_env.num_obs}->{vec_env.num_actions})")


if __name__ == "__main__":
    main()
