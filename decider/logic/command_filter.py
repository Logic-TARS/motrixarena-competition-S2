# command_filter.py
#
#   @description: Velocity command filter with clipping, acceleration limiting,
#                 and low-pass smoothing for stable robot control.
#

import math


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
        self.vx_min = float(cf.get("vx_min", -0.25))
        self.vy_max = float(cf.get("vy_max", 0.15))
        self.w_max  = float(cf.get("w_max", 1.5))

        # --- acceleration limits (per 20ms frame) ---
        self.vx_accel = float(cf.get("vx_accel", 0.03))
        self.vy_accel = float(cf.get("vy_accel", 0.02))
        self.w_accel  = float(cf.get("w_accel", 0.08))

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

    def apply(self, vx, vy, w):
        """Filter a raw velocity command and return (vx, vy, w)."""
        # 1. Clip absolute limits
        vx = max(self.vx_min, min(self.vx_max, vx))
        vy = max(-self.vy_max, min(self.vy_max, vy))
        w  = max(-self.w_max,  min(self.w_max,  w))

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

    def reset(self):
        """Reset filter state (call on state transitions if desired)."""
        self.last_cmd = (0.0, 0.0, 0.0)
