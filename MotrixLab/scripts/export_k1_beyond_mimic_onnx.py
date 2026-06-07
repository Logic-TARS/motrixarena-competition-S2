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

"""Export an RSLRL K1 BeyondMimic checkpoint to ONNX (119->22)."""

import argparse
from pathlib import Path

from _source_path import ensure_source_path

ensure_source_path()

import numpy as np  # noqa: E402
import torch  # noqa: E402
from rsl_rl.runners import OnPolicyRunner  # noqa: E402

from motrix_envs import registry as env_registry  # noqa: E402
from motrix_envs.locomotion.k1.cfg import K1_BEYOND_MIMIC_NUM_ACT, K1_BEYOND_MIMIC_NUM_OBS  # noqa: E402
from motrix_rl import registry as rl_registry  # noqa: E402
from motrix_rl.rslrl.torch.wrap_vec_env import RslrlNpEnvWrap  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Export an RSLRL K1 BeyondMimic checkpoint to ONNX (119->22).")
    parser.add_argument("checkpoint", type=Path, help="RSLRL checkpoint path, e.g. runs/.../model_1000.pt")
    parser.add_argument(
        "-o",
        "--output",
        type=Path,
        default=Path("exported/k1_mj_dance_002.onnx"),
        help="Output ONNX path",
    )
    parser.add_argument("--env", default="k1-mj-dance-002", help="MotrixLab env name")
    parser.add_argument("--device", default="cpu", help="Export device")
    parser.add_argument("--opset", type=int, default=17, help="ONNX opset version")
    parser.add_argument("--dynamic-batch", action="store_true", help="Export with dynamic batch dimension")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    device = torch.device(args.device)

    env = env_registry.make(args.env, sim_backend="np", num_envs=1)
    vec_env = RslrlNpEnvWrap(env, device)
    rlcfg = rl_registry.default_rl_cfg(args.env, "rslrl", backend="torch")
    runner = OnPolicyRunner(vec_env, rlcfg.runner.to_dict(), log_dir=None, device=device)
    runner.load(str(args.checkpoint))

    if vec_env.num_obs != K1_BEYOND_MIMIC_NUM_OBS:
        raise RuntimeError(f"Unexpected BeyondMimic obs dim: {vec_env.num_obs} (expected {K1_BEYOND_MIMIC_NUM_OBS})")
    if vec_env.num_actions != K1_BEYOND_MIMIC_NUM_ACT:
        raise RuntimeError(
            f"Unexpected BeyondMimic action dim: {vec_env.num_actions} (expected {K1_BEYOND_MIMIC_NUM_ACT})"
        )

    actor = runner.alg.actor
    if getattr(actor, "is_recurrent", False):
        raise RuntimeError("BeyondMimic export requires a non-recurrent MLP actor")

    args.output.parent.mkdir(parents=True, exist_ok=True)
    model = actor.as_jit().cpu().eval()
    example = torch.zeros(1, vec_env.num_obs, dtype=torch.float32)
    dynamic_axes = {"obs": {0: "batch"}, "actions": {0: "batch"}} if args.dynamic_batch else None

    torch.onnx.export(
        model,
        example,
        str(args.output),
        input_names=["obs"],
        output_names=["actions"],
        dynamic_axes=dynamic_axes,
        opset_version=args.opset,
    )

    import onnxruntime as ort

    session = ort.InferenceSession(str(args.output), providers=["CPUExecutionProvider"])
    ins = session.get_inputs()
    outs = session.get_outputs()
    test_input = np.zeros((1, vec_env.num_obs), dtype=np.float32)
    test_output = session.run([outs[0].name], {ins[0].name: test_input})[0]
    if tuple(test_output.shape) != (1, vec_env.num_actions):
        raise RuntimeError(
            f"ONNX output shape mismatch: got {tuple(test_output.shape)}, expected (1, {vec_env.num_actions})"
        )
    print(f"Exported {args.output} ONNX opset={args.opset} shape: {vec_env.num_obs}->{vec_env.num_actions}")


if __name__ == "__main__":
    main()
