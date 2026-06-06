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

"""Convert a Booster K1 motion CSV to the lightweight AMP motion-reference NPZ."""

import argparse
from pathlib import Path

from _source_path import ensure_source_path

ensure_source_path()

import numpy as np  # noqa: E402

from motrix_envs.locomotion.k1.cfg import K1_AMP_DEFAULT_MOTION_FILE, K1_AMP_JOINT_ORDER  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Convert Booster K1 CSV motion data to AMP NPZ.")
    parser.add_argument("input_file", type=Path, help="Booster CSV: root_pos(3), root_quat_xyzw(4), joint_pos(22)")
    parser.add_argument(
        "-o",
        "--output-file",
        type=Path,
        default=Path(K1_AMP_DEFAULT_MOTION_FILE),
        help="Output NPZ path used by k1-amp-walk by default",
    )
    parser.add_argument("--input-fps", type=float, default=50.0, help="Input CSV frame rate")
    parser.add_argument("--output-fps", type=float, default=50.0, help="Output NPZ frame rate")
    parser.add_argument(
        "--frame-range",
        nargs=2,
        type=int,
        metavar=("START", "END"),
        help="Optional 1-based inclusive frame range",
    )
    return parser.parse_args()


def _load_csv(path: Path, frame_range: tuple[int, int] | None) -> np.ndarray:
    if frame_range is None:
        motion = np.loadtxt(path, delimiter=",", dtype=np.float32)
    else:
        start, end = frame_range
        if start < 1 or end < start:
            raise ValueError("--frame-range must be 1-based inclusive START END with END >= START")
        motion = np.loadtxt(
            path,
            delimiter=",",
            dtype=np.float32,
            skiprows=start - 1,
            max_rows=end - start + 1,
        )
    if motion.ndim == 1:
        motion = motion.reshape(1, -1)
    expected_cols = 7 + len(K1_AMP_JOINT_ORDER)
    if motion.shape[1] != expected_cols:
        raise ValueError(f"Expected {expected_cols} CSV columns, got {motion.shape[1]}")
    if motion.shape[0] < 2:
        raise ValueError("Motion conversion needs at least two frames.")
    return motion


def _normalize_quat(quat: np.ndarray) -> np.ndarray:
    norm = np.linalg.norm(quat, axis=1, keepdims=True)
    return (quat / np.clip(norm, 1.0e-8, None)).astype(np.float32)


def _resample(values: np.ndarray, input_fps: float, output_fps: float) -> np.ndarray:
    if np.isclose(input_fps, output_fps):
        return values.astype(np.float32)
    duration = (values.shape[0] - 1) / input_fps
    input_times = np.arange(values.shape[0], dtype=np.float32) / input_fps
    output_times = np.arange(0.0, duration + 0.5 / output_fps, 1.0 / output_fps, dtype=np.float32)
    result = np.empty((output_times.shape[0], values.shape[1]), dtype=np.float32)
    for col in range(values.shape[1]):
        result[:, col] = np.interp(output_times, input_times, values[:, col])
    return result


def main() -> None:
    args = parse_args()
    motion = _load_csv(args.input_file, tuple(args.frame_range) if args.frame_range else None)
    root_pos = _resample(motion[:, :3], args.input_fps, args.output_fps)
    root_quat_xyzw = _normalize_quat(_resample(motion[:, 3:7], args.input_fps, args.output_fps))
    joint_pos = _resample(motion[:, 7:], args.input_fps, args.output_fps)
    output_dt = 1.0 / args.output_fps
    joint_vel = np.gradient(joint_pos, output_dt, axis=0).astype(np.float32)
    root_lin_vel = np.gradient(root_pos, output_dt, axis=0).astype(np.float32)

    args.output_file.parent.mkdir(parents=True, exist_ok=True)
    np.savez(
        args.output_file,
        fps=np.asarray(args.output_fps, dtype=np.float32),
        root_pos=root_pos.astype(np.float32),
        root_quat_xyzw=root_quat_xyzw.astype(np.float32),
        root_lin_vel=root_lin_vel,
        joint_pos=joint_pos.astype(np.float32),
        joint_vel=joint_vel,
        joint_names=np.asarray(K1_AMP_JOINT_ORDER),
    )
    print(
        f"Wrote {args.output_file} with {joint_pos.shape[0]} frames, "
        f"{joint_pos.shape[1]} joints, {args.output_fps:g} fps"
    )


if __name__ == "__main__":
    main()
