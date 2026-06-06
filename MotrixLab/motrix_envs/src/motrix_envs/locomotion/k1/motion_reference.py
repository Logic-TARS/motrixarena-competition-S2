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

"""Booster-style K1 motion reference utilities for AMP reward shaping."""

from pathlib import Path

import numpy as np


class K1AmpMotionReference:
    """Loads K1 reference motion without changing the AMP observation/action API."""

    def __init__(self, path: str | Path, joint_order: list[str], input_fps: float, ctrl_dt: float):
        self.path = Path(path).expanduser()
        self.joint_order = list(joint_order)
        self.input_fps = float(input_fps)
        self.ctrl_dt = float(ctrl_dt)
        self.root_pos, self.root_quat_xyzw, self.joint_pos, self.joint_vel, self.fps = self._load()
        self.frame_dt = 1.0 / self.fps
        self.num_frames = int(self.joint_pos.shape[0])

    def _load(self) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, float]:
        suffix = self.path.suffix.lower()
        if suffix == ".csv":
            return self._load_csv()
        if suffix == ".npz":
            return self._load_npz()
        raise ValueError(f"Unsupported K1 AMP motion file type: {self.path}")

    def _load_csv(self) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, float]:
        motion = np.loadtxt(self.path, delimiter=",", dtype=np.float32)
        if motion.ndim == 1:
            motion = motion.reshape(1, -1)
        self._validate_motion_columns(motion)
        root_pos = motion[:, :3].astype(np.float32)
        root_quat_xyzw = self._normalize_quat(motion[:, 3:7].astype(np.float32))
        joint_pos = motion[:, 7:].astype(np.float32)
        joint_vel = self._finite_difference(joint_pos, 1.0 / self.input_fps)
        return root_pos, root_quat_xyzw, joint_pos, joint_vel, self.input_fps

    def _load_npz(self) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, float]:
        with np.load(self.path, allow_pickle=False) as data:
            if "joint_names" in data:
                joint_names = [str(name) for name in data["joint_names"].tolist()]
                if joint_names != self.joint_order:
                    raise ValueError(
                        "K1 AMP motion joint_names do not match K1_AMP_JOINT_ORDER. "
                        f"motion={joint_names!r} cfg={self.joint_order!r}"
                    )
            root_pos = np.asarray(data["root_pos"], dtype=np.float32)
            root_quat_xyzw = np.asarray(data["root_quat_xyzw"], dtype=np.float32)
            joint_pos = np.asarray(data["joint_pos"], dtype=np.float32)
            fps = float(np.asarray(data["fps"]).reshape(()))
            if "joint_vel" in data:
                joint_vel = np.asarray(data["joint_vel"], dtype=np.float32)
            else:
                joint_vel = self._finite_difference(joint_pos, 1.0 / fps)
        self._validate_arrays(root_pos, root_quat_xyzw, joint_pos, joint_vel)
        return root_pos, self._normalize_quat(root_quat_xyzw), joint_pos, joint_vel, fps

    def _validate_motion_columns(self, motion: np.ndarray) -> None:
        expected = 7 + len(self.joint_order)
        if motion.shape[1] != expected:
            raise ValueError(
                f"K1 AMP motion CSV must have {expected} columns "
                f"(root_pos 3 + root_quat 4 + joints {len(self.joint_order)}); got {motion.shape[1]}"
            )

    def _validate_arrays(
        self,
        root_pos: np.ndarray,
        root_quat_xyzw: np.ndarray,
        joint_pos: np.ndarray,
        joint_vel: np.ndarray,
    ) -> None:
        if root_pos.ndim != 2 or root_pos.shape[1] != 3:
            raise ValueError(f"K1 AMP root_pos must have shape (N, 3); got {root_pos.shape}")
        if root_quat_xyzw.ndim != 2 or root_quat_xyzw.shape[1] != 4:
            raise ValueError(f"K1 AMP root_quat_xyzw must have shape (N, 4); got {root_quat_xyzw.shape}")
        if joint_pos.ndim != 2 or joint_pos.shape[1] != len(self.joint_order):
            raise ValueError(f"K1 AMP joint_pos must have shape (N, {len(self.joint_order)}); got {joint_pos.shape}")
        if joint_vel.shape != joint_pos.shape:
            raise ValueError(f"K1 AMP joint_vel must match joint_pos shape {joint_pos.shape}; got {joint_vel.shape}")
        frame_counts = {root_pos.shape[0], root_quat_xyzw.shape[0], joint_pos.shape[0], joint_vel.shape[0]}
        if len(frame_counts) != 1:
            raise ValueError("K1 AMP motion arrays must have the same frame count.")
        if joint_pos.shape[0] < 2:
            raise ValueError("K1 AMP motion reference needs at least two frames.")

    def frame_indices(self, episode_length: np.ndarray) -> np.ndarray:
        times = episode_length.astype(np.float32) * self.ctrl_dt
        return np.mod(np.rint(times / self.frame_dt).astype(np.int64), self.num_frames)

    def sample(self, episode_length: np.ndarray) -> dict[str, np.ndarray]:
        indices = self.frame_indices(episode_length)
        return {
            "root_pos": self.root_pos[indices],
            "root_quat_xyzw": self.root_quat_xyzw[indices],
            "joint_pos": self.joint_pos[indices],
            "joint_vel": self.joint_vel[indices],
        }

    @staticmethod
    def _finite_difference(values: np.ndarray, dt: float) -> np.ndarray:
        if values.shape[0] < 2:
            return np.zeros_like(values, dtype=np.float32)
        return np.gradient(values.astype(np.float32), float(dt), axis=0).astype(np.float32)

    @staticmethod
    def _normalize_quat(quat: np.ndarray) -> np.ndarray:
        norm = np.linalg.norm(quat, axis=1, keepdims=True)
        return (quat / np.clip(norm, 1.0e-8, None)).astype(np.float32)
