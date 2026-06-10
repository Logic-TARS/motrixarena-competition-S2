"""Shared K1 command envelope used by training and deployment controllers."""

import numpy as np


def forward_speed_limit(
    yaw_rate,
    max_forward_speed: float,
    yaw_full_speed: float = 0.25,
    yaw_zero_speed: float = 1.5,
):
    """Return the allowed forward speed for a yaw-rate command."""
    yaw = np.abs(np.asarray(yaw_rate, dtype=np.float32))
    span = max(float(yaw_zero_speed) - float(yaw_full_speed), 1.0e-6)
    scale = np.clip((float(yaw_zero_speed) - yaw) / span, 0.0, 1.0)
    scale = np.where(yaw <= float(yaw_full_speed), 1.0, scale)
    return scale * float(max_forward_speed)


def apply_forward_yaw_envelope(
    commands: np.ndarray,
    max_forward_speed: float,
    yaw_full_speed: float = 0.25,
    yaw_zero_speed: float = 1.5,
) -> np.ndarray:
    """Clamp forward commands in-place-compatible form without changing yaw."""
    result = np.asarray(commands, dtype=np.float32).copy()
    limit = forward_speed_limit(
        result[..., 2],
        max_forward_speed=max_forward_speed,
        yaw_full_speed=yaw_full_speed,
        yaw_zero_speed=yaw_zero_speed,
    )
    result[..., 0] = np.clip(result[..., 0], 0.0, limit)
    return result
