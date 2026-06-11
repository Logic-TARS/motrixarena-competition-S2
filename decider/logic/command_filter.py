# command_filter.py
#
#   @description: Velocity command filter with clipping, acceleration limiting,
#                 and low-pass smoothing for stable robot control.
#

import math


def forward_speed_limit(vx, w, yaw_full_speed=0.25, yaw_zero_speed=1.5, vx_max=1.0):
    """Scale down forward speed when |w| exceeds yaw_full_speed.

    At |w| <= yaw_full_speed, full vx is allowed.
    At |w| >= yaw_zero_speed, vx is clamped to 0.
    Between these thresholds, vx is linearly interpolated.

    This is the same envelope used during training — it prevents the policy
    from receiving forward commands at yaw rates it has never seen.
    """
    yaw_abs = abs(w)
    if yaw_abs <= yaw_full_speed:
        return vx
    span = max(yaw_zero_speed - yaw_full_speed, 1.0e-6)
    scale = max(0.0, min(1.0, (yaw_zero_speed - yaw_abs) / span))
    return min(vx, vx_max * scale)


class CommandFilter:
    """Post-processes velocity commands to ensure safe, smooth robot motion.

    Applies three stages:
      1. Absolute velocity clipping (safety bounds)
      2. Acceleration limiting (prevents sudden jumps)
      3. Low-pass filtering (smooths high-frequency noise)

    All parameters are read from the agent config under the ``cmd_filter`` key.
    """

    def __init__(self, config):
        # --- absolute limits ---
        cf = config.get("cmd_filter", {})
        self.vx_max = float(cf.get("vx_max", 0.75))
        # This policy was trained without backward commands. Zero must also
        # remain reachable so BRAKE can always converge to a full stop.
        self.vx_min = 0.0
        self.vy_max = abs(float(cf.get("vy_max", 0.15)))
        self.w_max  = abs(float(cf.get("w_max", 1.5)))
        self.yaw_full_speed = abs(float(cf.get("yaw_full_speed", 0.25)))
        self.yaw_zero_speed = abs(float(cf.get("yaw_zero_speed", 1.5)))

        # --- acceleration limits (per 20ms frame) ---
        # abs() guards against accidental negative config values — a negative
        # limit would be silently ignored by math.copysign in _limit_delta.
        self.vx_accel = abs(float(cf.get("vx_accel", 0.03)))
        self.vy_accel = abs(float(cf.get("vy_accel", 0.02)))
        self.w_accel  = abs(float(cf.get("w_accel", 0.08)))

        # --- low-pass smoothing (0 = output frozen, 1 = passthrough) ---
        self.alpha = float(cf.get("smooth_alpha", 0.3))

        # --- state ---
        self.last_cmd = (0.0, 0.0, 0.0)

    @staticmethod
    def _limit_delta(prev, target, max_delta):
        """Clamp *target* so |target - prev| <= max_delta."""
        delta = target - prev
        if abs(delta) <= max_delta:
            return target
        return prev + math.copysign(max_delta, delta)

    def _clip(self, vx, vy, w):
        """Apply absolute command limits without changing filter state."""
        vx = max(self.vx_min, min(self.vx_max, vx))
        vy = max(-self.vy_max, min(self.vy_max, vy))
        w = max(-self.w_max, min(self.w_max, w))
        vx = forward_speed_limit(
            vx, w,
            yaw_full_speed=self.yaw_full_speed,
            yaw_zero_speed=self.yaw_zero_speed,
            vx_max=self.vx_max,
        )
        return vx, vy, w

    def apply(self, vx, vy, w):
        """Filter a raw velocity command and return (vx, vy, w)."""
        # 1. Clip absolute limits
        vx, vy, w = self._clip(vx, vy, w)

        # 2. Rate limit (acceleration)
        vx = self._limit_delta(self.last_cmd[0], vx, self.vx_accel)
        vy = self._limit_delta(self.last_cmd[1], vy, self.vy_accel)
        w  = self._limit_delta(self.last_cmd[2], w,  self.w_accel)

        # 3. Low-pass filter
        a = self.alpha
        vx = (1.0 - a) * self.last_cmd[0] + a * vx
        vy = (1.0 - a) * self.last_cmd[1] + a * vy
        w  = (1.0 - a) * self.last_cmd[2] + a * w

        self.last_cmd = (vx, vy, w)
        return vx, vy, w

    def apply_clip_only(self, vx, vy, w):
        """Clip a short action without acceleration limiting or state updates."""
        return self._clip(vx, vy, w)

    def apply_turn_only(self, w):
        """Immediately stop translation while smoothly filtering rotation."""
        _, _, w = self._clip(0.0, 0.0, w)
        w = self._limit_delta(self.last_cmd[2], w, self.w_accel)
        w = (1.0 - self.alpha) * self.last_cmd[2] + self.alpha * w
        self.last_cmd = (0.0, 0.0, w)
        return self.last_cmd

    def apply_brake(self):
        """Smoothly bring all command axes to zero."""
        return self.apply(0.0, 0.0, 0.0)

    def is_translation_stopped(self, tol=0.02):
        """Return whether the filtered planar command is effectively stopped."""
        return (
            abs(self.last_cmd[0]) <= float(tol)
            and abs(self.last_cmd[1]) <= float(tol)
        )

    def reset(self):
        """Reset filter state (call on state transitions if desired)."""
        self.last_cmd = (0.0, 0.0, 0.0)
