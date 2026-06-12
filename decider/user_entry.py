# user_entry.py
#
#   @description:   Logic entry point for mos-brain decider (Simplified)
#                   Game logic is separated into game() function.
#

import time
import traceback
import sys
import os
import math
import numpy as np

# Ensure we can find the logic package
CUR_DIR = os.path.dirname(os.path.abspath(__file__))
if CUR_DIR not in sys.path:
    sys.path.append(CUR_DIR)

from logic.sub_statemachines import chase_ball, find_ball, go_back_to_field, dribble
from logic.policy_statemachines import goalkeeper

import csv
from datetime import datetime


class DataRecorder:
    def __init__(self, log_dir):
        self.log_dir = log_dir
        if not os.path.exists(log_dir):
            os.makedirs(log_dir)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.filepath = os.path.join(log_dir, f"dribble_debug_{timestamp}.csv")
        self.file = open(self.filepath, 'w', newline='')
        self.writer = csv.writer(self.file)
        self.header_written = False
        
    def log(self, data):
        if not self.header_written:
            self.writer.writerow(data.keys())
            self.header_written = True
        self.writer.writerow(data.values())
        self.file.flush()

    def close(self):
        self.file.close()

class AdvancedDribbler:
    def __init__(self, agent):
        self.agent = agent
        self.logger = agent.get_logger().get_child("AdvDribble")
        
        # Instrumentation
        # Use project-relative debug_logs directory (parent of `decider`)
        log_dir = os.path.abspath(os.path.join(CUR_DIR, '..', 'debug_logs'))
        print(f"[DEBUG] DataRecorder log_dir: {log_dir}")
        self.recorder = DataRecorder(log_dir)
        
        # Parameters
        self.bturn_p = 2.0
        self.side_correction_p = 2.5
        self.forward_p = 1.0
        
        self.setup_dist = 0.40
        self.dribble_dist = 0.20 # Ball should be slightly in front
        self.max_fw_vel = 0.8
        
        # Read field dimensions from config (supports S/M/L league)
        config = agent.get_config()
        league = config.get("league", "M")
        field_dims = config.get("field_size", {}).get(league, [14.0, 9.0])
        self.field_length = float(field_dims[0])
        self.field_width = float(field_dims[1])
        
        # Anti-Oscillation
        self.spread_factor_max = 20.0 # degrees
        self.spread_factor_min = 5.0 # degrees

        # Hysteresis to avoid mode chattering near b_x threshold
        self.turn_to_ball_enter_bx = 0.03
        self.turn_to_ball_exit_bx = 0.08
        self.turn_to_ball_mode = False
        self.direction_consistency_bx = 0.12  # keep turn direction consistent near mode boundary
        
    def get_target_vector(self):
        """
        Calculate dribbling direction (Vector Field)
        COORDINATE SYSTEM (NEW):
        - Origin: Center of field
        - X+: Points to Opponent Goal (Front)
        - Y+: Points to Left side
        - Field Length: X-axis
        - Field Width: Y-axis
        """
        # 1. Goal Attraction
        # Goal is at (L/2, 0) - forward on X, center on Y
        goal_x = self.field_length / 2.0
        goal_y = 0.0
        
        my_pos = self.agent.get_self_pos()
        if my_pos is None:
            return np.array([1.0, 0.0]), 0 # Default forward (X+)
            
        # Global Vector to Goal
        g_dx = goal_x - my_pos[0]
        g_dy = goal_y - my_pos[1]
        
        # 2. Boundary Repulsion (Side Lines are at Y = +/- W/2)
        # If too close to side lines, push towards center (Y=0)
        dist_to_left = (self.field_width / 2.0) - my_pos[1]   # Y+ is left
        dist_to_right = my_pos[1] - (-self.field_width / 2.0)  # Y- is right
        
        repulsion_y = 0.0
        margin = 1.0 # Buffer
        
        # If close to Left Boundary (Y > 0), push Right (Y-)
        if dist_to_left < margin:
            repulsion_y -= (margin - dist_to_left) * 2.0
            
        # If close to Right Boundary (Y < 0), push Left (Y+)
        if dist_to_right < margin:
            repulsion_y += (margin - dist_to_right) * 2.0
            
        # Combine
        final_dx = g_dx  # Goal attraction provides X component
        final_dy = g_dy + repulsion_y
        
        norm = math.hypot(final_dx, final_dy)
        if norm < 0.001:
            return np.array([1.0, 0.0]), 0 # Default forward (X+)
            
        target_vec = np.array([final_dx / norm, final_dy / norm])
        
        # Zone Safety check for Deadband
        # If central area (Y close to 0), safe.
        is_safe = (abs(my_pos[1]) < self.field_width / 3.0)
        
        return target_vec, is_safe

    def run(self):
        if not self.agent.get_if_ball():
            self.logger.info("Lost ball, stopping.")
            self.agent.cmd_vel(0,0,0)
            return

        # 1. Get State
        # Ball in Robot Body Frame [Forward, Left] (X+前, Y+左)
        ball_pos = self.agent.get_ball_pos()
        b_x = ball_pos[0]  # Forward
        b_y = ball_pos[1]  # Left
        
        my_pos = self.agent.get_self_pos()
        my_yaw = self.agent.get_self_yaw()
        
        # [NEW] Ball Behind Check - Turn to face ball first
        # If ball is behind robot (b_x < threshold), we need to turn around
        ball_dist = math.hypot(b_x, b_y)
        ball_angle_to_robot = math.atan2(b_y, b_x)  # Angle from robot forward to ball
        ball_angle_deg = math.degrees(ball_angle_to_robot)
        self.logger.info(
            f"[AdvDribble] ball_angle_to_robot={ball_angle_to_robot:.4f}rad ({ball_angle_deg:.1f}deg), b=({b_x:.3f}, {b_y:.3f})"
        )
        
        prev_turn_to_ball_mode = self.turn_to_ball_mode
        if self.turn_to_ball_mode:
            self.turn_to_ball_mode = b_x < self.turn_to_ball_exit_bx
        else:
            self.turn_to_ball_mode = b_x < self.turn_to_ball_enter_bx

        if self.turn_to_ball_mode != prev_turn_to_ball_mode:
            self.logger.info(
                f"[AdvDribble] MODE_SWITCH turn_to_ball={self.turn_to_ball_mode} b_x={b_x:.3f} (enter<{self.turn_to_ball_enter_bx:.2f}, exit<{self.turn_to_ball_exit_bx:.2f})"
            )

        if self.turn_to_ball_mode:  # Ball behind mode (with hysteresis)
            # Turn towards the ball instead of dribbling
            turn_speed = 1.5 * ball_angle_to_robot  # P-control to face ball
            turn_speed = max(min(turn_speed, 1.5), -1.5)  # Clamp
            
            # Also move slightly towards ball if it's far
            approach_speed = 0.0
            if ball_dist > 0.3:
                # Move forward/backward based on ball position
                # If ball is behind, we should approach after turning
                approach_speed = 0.3 * b_x / (ball_dist + 0.01)  # Will be negative if ball behind
                approach_speed = max(min(approach_speed, 0.5), -0.3)
            
            self.logger.info(f"[AdvDribble] TURN_TO_BALL: BallAngle={ball_angle_deg:.1f} TurnSpd={turn_speed:.2f}")
            self.agent.cmd_vel(approach_speed, 0, turn_speed)
            self.agent.move_head(math.inf, math.inf)
            
            # Log for debugging
            log_data = {
                "time": time.time(),
                "rx": my_pos[0] if my_pos is not None else 0,
                "ry": my_pos[1] if my_pos is not None else 0,
                "ryaw": my_yaw,
                "ball_x": b_x,
                "ball_y": b_y,
                "t_vec_gx": 0, "t_vec_gy": 0,
                "t_ang_local": ball_angle_deg,
                "cmd_x": approach_speed, "cmd_y": 0, "cmd_w": turn_speed,
                "aligned": 0, "safe_zone": 0,
                "err_y": 0, "err_x": 0
            }
            self.recorder.log(log_data)
            return  # Exit early, don't do normal dribble logic
        
        # Target Vector
        target_vec_global, is_safe_zone = self.get_target_vector()
        
        # [DEBUG] Log coordinate values for diagnosis
        self.logger.info(f"[COORD] pos={my_pos}, yaw={my_yaw:.1f}, t_vec={target_vec_global}")
        
        # Rotate Target to Local Frame
        yaw_rad = math.radians(my_yaw)
        # NEW Coordinate System: 
        # Global: X(Forward), Y(Left)
        # Body: X(Forward), Y(Left)
        # Robot Yaw=0 means facing X+ (Forward)
        # Rotation: v_body = R(-yaw) * v_global
        # t_local_x = Gx * cos(yaw) + Gy * sin(yaw)
        # t_local_y = -Gx * sin(yaw) + Gy * cos(yaw)
        
        t_local_x = target_vec_global[0] * math.cos(yaw_rad) + target_vec_global[1] * math.sin(yaw_rad)
        t_local_y = -target_vec_global[0] * math.sin(yaw_rad) + target_vec_global[1] * math.cos(yaw_rad)
        
        target_angle_local = math.atan2(t_local_y, t_local_x)
        target_angle_deg = math.degrees(target_angle_local)
        
        # 2. Omnidirectional Control
        
        # A. Turn (da)
        # Minimize angle to target
        da = -self.bturn_p * target_angle_local  # invert sign to align with TURN_TO_BALL rotation direction
        
        # [NEW] Dampen Turn when very close to ball to prevent oscillation ("Large angle change" issue)
        # If b_x is small (e.g. 0.1), simple turning changes relative geometry fast.
        # Scale down da when close.
        if my_pos is not None: # Ensure we have data
             # Use b_x from previous step or estimate? 
             # Actually we calculated b_x_virt later. 
             # Let's move da clamping/scaling to AFTER b_x_virt calculation or estimate it here.
             # Better: Apply scaling at the end of this block or simply use dist to ball.
             dist_to_ball = math.hypot(b_x, b_y)
             turn_damp = max(0.4, min(1.0, dist_to_ball / 0.4))
             da *= turn_damp
        
        # B. Orbit/Sway (dy)
        # We want the ball to be on the "line" to target.
        # Project Ball Pos onto the Normal of Target Vector?
        # Simpler: Rotate Ball Pos so that Target Vector lies on X-axis (Virtual Frame)
        # In Virtual Frame:
        # Ball Y should be 0.
        # Ball X should be setup_dist.
        
        # Rotation from Robot Body to Virtual Target Frame
        # rot = -target_angle_local
        c_r = math.cos(-target_angle_local)
        s_r = math.sin(-target_angle_local)
        
        b_x_virt = b_x * c_r - b_y * s_r
        b_y_virt = b_x * s_r + b_y * c_r
        
        # PID Controls in Virtual Frame
        # Lateral Error: We want b_y_virt = 0
        err_y = b_y_virt
        cmd_y_virt = self.side_correction_p * err_y 
        
        # Forward Error: We want b_x_virt = setup_dist (or dribble_dist if aligned)
        # Adaptive Deadband
        deadband = self.spread_factor_max if is_safe_zone else self.spread_factor_min
        # [NEW] Require Ball to be In Front (b_x_virt > 0) to consider Aligned
        aligned = abs(target_angle_deg) < deadband and abs(err_y) < 0.1 and b_x_virt > 0.1
        
        # [NEW] 临门一脚 Mode: Near goal, be more aggressive
        # Goal is at X = field_length/2, so threshold is ~80% of the way
        near_goal = my_pos is not None and my_pos[0] > self.field_length * 0.35
        if near_goal:
            # Relax alignment requirement
            aligned = abs(target_angle_deg) < 25.0 and b_x_virt > 0.05
        
        # [NEW] Interpolate Target Distance for smooth transition (Creep)
        target_dist = self.setup_dist
        if aligned:
            target_dist = self.dribble_dist
        elif abs(target_angle_deg) < 25.0:
             # Interpolate between setup_dist (0.4) and dribble_dist (0.2) based on alignment
             # 25 deg -> 0.4, 5 deg -> 0.2
             ratio = (25.0 - abs(target_angle_deg)) / (25.0 - 5.0)
             ratio = max(0.0, min(1.0, ratio))
             target_dist = self.setup_dist - ratio * (self.setup_dist - self.dribble_dist)

        err_x = b_x_virt - target_dist

        # Velocity Damping (Anti-Oscillation)
        # Don't rush if not aligned
        forward_factor = 1.0
        if not aligned:
            # Dampen based on angle error but allow "creep" if error is not huge
            angle_err = abs(target_angle_deg)
            if angle_err < 25.0:
                # Creep zone: Interpolate 1.0 -> 0.2
                forward_factor = 1.0 - (angle_err / 25.0) * 0.5 
            else:
                forward_factor = 0.0
        
        # [NEW] Near goal, always push forward aggressively
        if near_goal and b_x_virt > 0.05:
            forward_factor = max(forward_factor, 0.8)  # At least 80% power near goal
            
        cmd_x_virt = self.forward_p * err_x * forward_factor

        # [NEW] Minimum Push Velocity when Aligned
        # Always maintain minimum forward velocity when aligned (PUSH mode)
        if aligned:
             if cmd_x_virt < 0.5:  # Always push forward at min 0.5
                 cmd_x_virt = 0.5
        
        # [NEW] Near goal, even if not aligned, keep pushing forward
        if near_goal and b_x_virt > 0.1:
            if cmd_x_virt < 0.4:  # Minimum near-goal push
                cmd_x_virt = 0.4
        
        
        # [NEW] Clamp Virtual Velocities individually first
        cmd_x_virt = max(min(cmd_x_virt, self.max_fw_vel), -0.5)
        # cmd_y_virt is proportional to error, clamping it is key
        # Using a slightly higher limit for lateral correction if needed, but safe to clamp to max velocity
        cmd_y_virt = max(min(cmd_y_virt, self.max_fw_vel), -self.max_fw_vel)
        
        # Transform Commands back to Body Frame
        c = math.cos(target_angle_local)
        s = math.sin(target_angle_local)
        
        cmd_x = cmd_x_virt * c - cmd_y_virt * s
        cmd_y = cmd_x_virt * s + cmd_y_virt * c
        
        # [NEW] Global Clamp on Linear Velocity
        lin_vel_norm = math.hypot(cmd_x, cmd_y)
        if lin_vel_norm > self.max_fw_vel:
            scale = self.max_fw_vel / lin_vel_norm
            cmd_x *= scale
            cmd_y *= scale
            
        # [NEW] Clamp Angular Velocity
        da = max(min(da, 1.5), -1.5)

        # Keep angular direction consistent with TURN_TO_BALL near threshold
        # to avoid opposite commands around mode boundary.
        if b_x < self.direction_consistency_bx and abs(ball_angle_to_robot) > 0.2 and abs(da) > 1e-6:
            da = math.copysign(abs(da), ball_angle_to_robot)
        
        # LOGGING
        log_data = {
            "time": time.time(),
            "rx": my_pos[0] if my_pos is not None else 0,
            "ry": my_pos[1] if my_pos is not None else 0,
            "ryaw": my_yaw,
            "ball_x": b_x,
            "ball_y": b_y,
            "t_vec_gx": target_vec_global[0],
            "t_vec_gy": target_vec_global[1],
            "t_ang_local": target_angle_deg,
            "cmd_x": cmd_x,
            "cmd_y": cmd_y,
            "cmd_w": da,
            "aligned": int(aligned),
            "safe_zone": int(is_safe_zone),
            "err_y": err_y,
            "err_x": err_x
        }
        self.recorder.log(log_data)
        
        # 3. Kick Decision — fire when close to opponent goal and ball is in front
        if near_goal and b_x_virt > 0.05 and ball_dist < 0.35:
            self.logger.info(f"[AdvDribble] KICK near goal! dist={ball_dist:.2f} b_x={b_x:.2f}")
            self.agent.kick(foot=0, death=0)
            self.agent.move_head(math.inf, math.inf)
            return
        
        # 4. Final Command
        self.logger.info(f"[AdvDribble] Safe:{is_safe_zone} Alg:{aligned} T_Ang:{target_angle_deg:.1f} Cmd:({cmd_x:.2f}, {cmd_y:.2f}, {da:.2f})")
        self.agent.cmd_vel(cmd_x, cmd_y, da)
        self.agent.move_head(math.inf, math.inf)


class PushToGoalController:
    """Stay behind the ball (relative to opponent goal) and push it forward.

    Single unified behaviour — no SETUP/PUSH mode switching.
    Always targets a position ~12 cm behind the ball so the robot
    naturally pushes the ball toward the opponent goal.
    """

    def __init__(self, agent):
        self.agent = agent
        self.logger = agent.get_logger().get_child("PushToGoal")

        config = agent.get_config()
        league = config.get("league", "M")
        field_dims = config.get("field_size", {}).get(league, [14.0, 9.0])
        self.field_length = float(field_dims[0])

        # Tunable parameters
        self.push_dist = 0.12       # target distance behind the ball (m)
        self.kp_pos = 2.5           # position P-gain
        self.kp_yaw = 2.5           # yaw P-gain
        self.max_vel = 1.0          # max linear speed (m/s)
        self.max_yaw = 2.5          # max angular speed (rad/s)
        self.kick_dist = 0.3        # distance to goal to trigger kick (m)

    def run(self):
        # 1. No ball – stop and let find_ball takeover
        if not self.agent.get_if_ball():
            self.logger.info("[PushToGoal] No ball.")
            self.agent.cmd_vel(0, 0, 0)
            return

        # 2. World-frame positions
        ball_w = self.agent.get_ball_pos_in_map()   # [x, y]
        robot_w = self.agent.get_self_pos()          # [x, y]
        robot_yaw_deg = self.agent.get_self_yaw()    # degrees
        robot_yaw_rad = math.radians(robot_yaw_deg)

        # Guard against None
        if ball_w is None or robot_w is None or ball_w[0] is None or robot_w[0] is None:
            self.logger.warning("[PushToGoal] Bad position data, stopping.")
            self.agent.cmd_vel(0, 0, 0)
            return

        goal = np.array([self.field_length / 2.0, 0.0])
        ball_w = np.array(ball_w, dtype=float)
        robot_w = np.array(robot_w, dtype=float)

        # 3. Direction from ball to goal
        to_goal = goal - ball_w
        dist_to_goal = float(np.linalg.norm(to_goal))
        if dist_to_goal < 0.01:
            to_goal = np.array([1.0, 0.0])  # fallback
        else:
            to_goal = to_goal / dist_to_goal

        # 4. Target position: behind ball, opposite to goal direction
        target_pos = ball_w - to_goal * self.push_dist

        # 5. Position error (world frame → body frame)
        err_world = target_pos - robot_w
        c = math.cos(robot_yaw_rad)
        s = math.sin(robot_yaw_rad)
        err_body_x = err_world[0] * c + err_world[1] * s
        err_body_y = -err_world[0] * s + err_world[1] * c

        # 6. Yaw error: face the goal
        target_yaw = math.atan2(goal[1] - robot_w[1], goal[0] - robot_w[0])
        yaw_err = self.agent.angle_normalize(target_yaw - robot_yaw_rad)

        # 7. P-control
        cmd_x = float(np.clip(self.kp_pos * err_body_x, -self.max_vel, self.max_vel))
        cmd_y = float(np.clip(self.kp_pos * err_body_y, -self.max_vel, self.max_vel))
        cmd_w = float(np.clip(self.kp_yaw * yaw_err, -self.max_yaw, self.max_yaw))

        # 8. Kick when close to opponent goal and ball is in front
        ball_dist = self.agent.get_ball_distance()
        ball_pos = self.agent.get_ball_pos()
        near_goal = dist_to_goal < 2.0 or robot_w[0] > self.field_length * 0.35
        if near_goal and ball_pos[0] is not None and ball_pos[0] > 0.05 and ball_dist < 0.35:
            self.logger.info(f"[PushToGoal] KICK near goal! dist_to_goal={dist_to_goal:.2f}")
            self.agent.kick(foot=0, death=0)

        self.logger.info(
            f"[PushToGoal] err_body=({err_body_x:.2f},{err_body_y:.2f}) "
            f"yaw_err={math.degrees(yaw_err):.1f}deg "
            f"cmd=({cmd_x:.2f},{cmd_y:.2f},{cmd_w:.2f})"
        )
        self.agent.cmd_vel(cmd_x, cmd_y, cmd_w)
        self.agent.move_head(math.inf, math.inf)


class ContinuousPushController:
    """Continuous error-space controller — no FSM states, no mode switches.

    Five error terms computed every frame, blended through a sigmoid
    distance-dependent weight.  Far from the ball → approach behaviour;
    near the ball → behind-depth / lateral / yaw alignment → push to goal.

    Key invariants:
      - behind_depth = dot(ball - robot, to_goal)  →  positive == robot
        is behind the ball (good for pushing).
      - Sideline repulsion is computed in world frame then rotated into
        the body frame before composing with cmd_x / cmd_y.
      - vy and w are soft-clipped with tanh to prevent long-duration
        command saturation.
    """

    # ------------------------------------------------------------------
    # Tunable parameters (can be overridden via config.yaml)
    # ------------------------------------------------------------------
    _DEFAULTS = {
        "k_approach": 0.5,          # approach P-gain (vx ∝ ball distance when far)
        "k_depth": 2.0,             # behind-depth P-gain
        "k_lat": 1.5,               # lateral error P-gain
        "k_yaw": 1.8,               # yaw error P-gain
        "k_sideline": 1.2,          # sideline repulsion gain

        "target_behind": 0.15,      # desired metres behind ball (along to_goal)
        "d_transition": 1.0,        # ball distance at 50/50 blend (metres)
        "d_scale": 0.4,             # sigmoid steepness

        "vx_max_approach": 0.9,     # max approach vx when far (unitless)
        "vx_max_push": 0.8,         # max push vx when near
        "vx_max_push_rev": 0.2,     # max reverse speed when too close
        "vy_max": 0.7,              # lateral velocity cap
        "w_max": 1.5,               # angular velocity cap

        "sideline_margin": 0.8,     # metres from sideline where repulsion starts
        "soft_clip_threshold": 0.7, # vy / w soft-clip entry point
    }

    def __init__(self, agent):
        self.agent = agent
        self.logger = agent.get_logger().get_child("ContPush")

        config = agent.get_config()
        league = config.get("league", "M")
        field_dims = config.get("field_size", {}).get(league, [14.0, 9.0])
        self.field_length = float(field_dims[0])
        self.field_width = float(field_dims[1])

        # Load parameters (config overrides hardcoded defaults)
        user_params = config.get("continuous_push", {})
        for key, default in self._DEFAULTS.items():
            setattr(self, key, user_params.get(key, default))

        # Diagnostics (read by _record_trajectory)
        self._last_errors = {}
        self._last_cmd = (0.0, 0.0, 0.0)

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _sigmoid(x):
        """Sigmoid 1/(1+exp(-x))."""
        # Clamp input to avoid overflow
        x = max(-20.0, min(20.0, x))
        return 1.0 / (1.0 + math.exp(-x))

    @staticmethod
    def _soft_clip(value, threshold):
        """Soft-clip: linear below *threshold*, tanh-compressed above.

        The output is always strictly in (-1, 1) — it never reaches ±1.
        The tanh input is clamped at 6.0 to avoid float64 precision loss
        (tanh(6) ≈ 0.9999877, distinguishable from 1.0).
        """
        if abs(value) <= threshold:
            return value
        excess = min((abs(value) - threshold) / (1.0 - threshold + 1e-9), 6.0)
        return math.copysign(threshold + (1.0 - threshold) * math.tanh(excess), value)

    def _send_final_cmd(self, vx, vy, w):
        """Send desired final sim command without re-saturating in SimAgent.

        SimAgent.cmd_vel() applies config scaling before publishing to the
        ZMQ action layer.  The continuous controller computes in that final
        command space, so divide by the configured scale first.
        """
        config = self.agent.get_config()
        sx = float(config.get("max_walk_vel_x", 1.0)) or 1.0
        sy = float(config.get("max_walk_vel_y", 1.0)) or 1.0
        sw = float(config.get("max_walk_vel_theta", 1.0)) or 1.0
        self.agent.cmd_vel(
            float(np.clip(vx / sx, -1.0, 1.0)),
            float(np.clip(vy / sy, -1.0, 1.0)),
            float(np.clip(w / sw, -1.0, 1.0)),
        )

    # ------------------------------------------------------------------
    # Error computation
    # ------------------------------------------------------------------

    def _compute_errors(self, ball_w, robot_w, robot_yaw_rad, goal):
        """Return a dict with all five error terms plus derived geometry."""
        # -- ball → goal direction --
        to_goal = goal - ball_w
        dist_to_goal = float(np.linalg.norm(to_goal))
        if dist_to_goal < 0.01:
            to_goal_unit = np.array([1.0, 0.0], dtype=float)
            perp_unit = np.array([0.0, 1.0], dtype=float)
        else:
            to_goal_unit = to_goal / dist_to_goal
            perp_unit = np.array([-to_goal_unit[1], to_goal_unit[0]], dtype=float)

        # -- behind depth:  ball_w - robot_w  projected onto to_goal --
        #    positive = robot is behind the ball (desired for pushing).
        delta_w = ball_w - robot_w   # vector from robot TO ball
        behind_depth = float(np.dot(delta_w, to_goal_unit))

        # -- lateral error: robot offset from ball→goal line --
        delta_rb = robot_w - ball_w   # vector from ball TO robot
        lateral_err = float(np.dot(delta_rb, perp_unit))
        # positive = robot is left of the ball→goal line

        # -- yaw error: face the goal --
        target_yaw_rad = math.atan2(
            goal[1] - robot_w[1], goal[0] - robot_w[0]
        )
        yaw_err = self.agent.angle_normalize(target_yaw_rad - robot_yaw_rad)

        # -- ball distance --
        ball_dist = self.agent.get_ball_distance()

        # -- sideline risk (world frame) --
        half_w = self.field_width / 2.0
        dist_to_right = float(robot_w[1] - (-half_w))
        dist_to_left = float(half_w - robot_w[1])
        sideline_risk = max(0.0, self.sideline_margin - min(dist_to_right, dist_to_left))
        if dist_to_left < dist_to_right:
            sideline_repulsion_y = -sideline_risk
        elif dist_to_right < dist_to_left:
            sideline_repulsion_y = sideline_risk
        else:
            sideline_repulsion_y = 0.0

        return {
            "behind_depth": behind_depth,
            "lateral_err": lateral_err,
            "to_goal_unit": to_goal_unit,
            "yaw_err_rad": yaw_err,
            "yaw_err_deg": math.degrees(yaw_err),
            "ball_dist": ball_dist,
            "sideline_risk": sideline_risk,
            "sideline_repulsion_y": sideline_repulsion_y,
            "dist_to_goal": dist_to_goal,
        }

    # ------------------------------------------------------------------
    # Main control loop
    # ------------------------------------------------------------------

    def run(self):
        # --- 1. Guards ---
        if not self.agent.get_if_ball():
            self.logger.debug("[ContPush] No ball — stop, let find_ball takeover.")
            self.agent.cmd_vel(0, 0, 0)
            self._last_errors = {}
            self._last_cmd = (0.0, 0.0, 0.0)
            return

        ball_w = self.agent.get_ball_pos_in_map()
        robot_w = self.agent.get_self_pos()
        if ball_w is None or robot_w is None or ball_w[0] is None or robot_w[0] is None:
            self.logger.warning("[ContPush] Bad position data, stopping.")
            self.agent.cmd_vel(0, 0, 0)
            self._last_errors = {}
            self._last_cmd = (0.0, 0.0, 0.0)
            return

        ball_w = np.array(ball_w, dtype=float)
        robot_w = np.array(robot_w, dtype=float)
        robot_yaw_deg = self.agent.get_self_yaw()
        robot_yaw_rad = math.radians(robot_yaw_deg)
        goal = np.array([self.field_length / 2.0, 0.0], dtype=float)

        # --- 2. Compute all errors ---
        err = self._compute_errors(ball_w, robot_w, robot_yaw_rad, goal)
        d_ball = err["ball_dist"]
        goal_dist = err["dist_to_goal"]

        # --- 3. Distance-dependent blend ---
        alpha = self._sigmoid((d_ball - self.d_transition) / self.d_scale)
        # alpha → 0  near ball (push regime)
        # alpha → 1  far from ball (approach regime)
        w_near = 1.0 - alpha   # push / alignment weight
        w_far = alpha           # approach / chase weight

        # Continuous target: far away, move to the ball; near the ball,
        # drift toward the ball-behind point.  Convert world error to body
        # frame so approach works regardless of robot heading.
        approach_target = ball_w - err["to_goal_unit"] * (self.target_behind * w_near)
        target_delta = approach_target - robot_w
        c = math.cos(robot_yaw_rad)
        s = math.sin(robot_yaw_rad)
        target_body_x = float(target_delta[0] * c + target_delta[1] * s)
        target_body_y = float(-target_delta[0] * s + target_delta[1] * c)

        # --- 4. VX — forward velocity ---
        # Far: navigate toward the continuous target in body frame.
        vx_approach = float(np.clip(
            self.k_approach * target_body_x,
            -self.vx_max_push_rev,
            self.vx_max_approach,
        ))

        # Near: behind-depth P-control.
        depth_err_vx = err["behind_depth"] - self.target_behind
        vx_push = float(np.clip(
            self.k_depth * depth_err_vx,
            -self.vx_max_push_rev,
            self.vx_max_push,
        ))

        # Scale forward push by alignment quality — don't rush if badly misaligned.
        alignment_factor = max(0.3, 1.0 - abs(err["yaw_err_rad"]) / math.pi)
        # Blend near / far
        vx_raw = w_near * vx_push * alignment_factor + w_far * vx_approach
        vx = float(np.clip(vx_raw, -0.3, 1.0))

        # --- 5. VY — lateral correction + sideline repulsion ---
        vy_approach = float(np.clip(self.k_approach * target_body_y, -self.vy_max, self.vy_max))
        vy_lat = float(np.clip(-self.k_lat * err["lateral_err"], -self.vy_max, self.vy_max))

        # Sideline repulsion: push toward field centre (Y=0) in WORLD frame,
        # then rotate into body frame before adding to cmd_x / cmd_y.
        world_repulsion_y = float(err["sideline_repulsion_y"] * self.k_sideline)
        # Rotate world repulsion [0, world_repulsion_y] into body frame.
        body_rep_x = world_repulsion_y * s   # from vx = wx*cos+wy*sin, wx=0
        body_rep_y = world_repulsion_y * c   # from vy = -wx*sin+wy*cos, wx=0

        # Fade sideline repulsion near goal (don't push away from goal line).
        near_goal_factor = float(np.clip((goal_dist - 2.0) / 4.0, 0.0, 1.0))
        body_rep_x *= near_goal_factor
        body_rep_y *= near_goal_factor

        vy = w_far * vy_approach + w_near * vy_lat + body_rep_y
        vy = float(np.clip(vy, -self.vy_max, self.vy_max))

        # --- 6. W — yaw control with near-ball damping ---
        w_raw = self.k_yaw * err["yaw_err_rad"]
        w_damping = 0.3 + 0.7 * float(
            np.clip((d_ball - 0.3) / 0.5, 0.0, 1.0)
        )
        w = float(np.clip(w_raw * w_damping, -self.w_max, self.w_max))

        # --- 7. Anti-saturation soft-clip (vy, w only; vx stays linear) ---
        vy = self._soft_clip(vy, self.soft_clip_threshold)
        w = self._soft_clip(w, self.soft_clip_threshold)

        # --- 8. Apply sideline repulsion vx component ---
        vx = float(np.clip(vx + body_rep_x * 0.2, -0.3, 1.0))
        # Only a fraction (0.2x) so forward push dominates.

        # --- 9. Store diagnostics ---
        self._last_errors = err
        self._last_cmd = (vx, vy, w)

        # --- 10. Issue command ---
        distance_info = (
            f"d_ball={d_ball:.2f} alpha={alpha:.2f} "
            f"behind={err['behind_depth']:.3f} lat={err['lateral_err']:.3f} "
            f"yaw={err['yaw_err_deg']:.1f}deg sideline={err['sideline_risk']:.2f}"
        )
        self.logger.debug(f"[ContPush] {distance_info} → cmd=({vx:.2f},{vy:.2f},{w:.2f})")

        self._send_final_cmd(vx, vy, w)
        self.agent.move_head(math.inf, math.inf)


def init(agent) -> None:
    agent.get_logger().info("[UserEntry] Initializing Logic...")
    
    # Initialize State Machines
    agent.chase_ball_machine = chase_ball.ChaseBallStateMachine(agent)
    agent.find_ball_machine = find_ball.FindBallStateMachine(agent)
    agent.go_back_machine = go_back_to_field.GoBackToFieldStateMachine(agent)
    agent.dribble_machine = dribble.DribbleStateMachine(agent)
    agent.goalkeeper_machine = goalkeeper.GoalkeeperStateMachine(agent)
    
    # Initialize Advanced Dribbler
    agent.adv_dribbler = AdvancedDribbler(agent)

    # Initialize Push-to-Goal controller (kept for reference / fallback)
    agent.push_to_goal = PushToGoalController(agent)

    # Initialize Continuous Push controller (primary attacker strategy)
    agent.continuous_push = ContinuousPushController(agent)

    agent.state_machine_runners = {
        "chase_ball": agent.chase_ball_machine.run,
        "find_ball": agent.find_ball_machine.run,
        "go_back_to_field": agent.go_back_machine.run,
        "dribble": agent.dribble_machine.run,
        "adv_dribble": agent.adv_dribbler.run, # Register new runner
        "stop": agent.stop,
        "goalkeeper": agent.goalkeeper_machine.run,
    }
    
    # Basic Configs
    agent.default_chase_distance = agent.get_config().get("chase",{}).get("default_chase_distance", 0.7)
    
    # Relocalize
    agent.relocate()

def loop(agent) -> None:
    try:
        game(agent)
    except Exception as e:
        agent.get_logger().error(f"Error in user_entry loop: {e}")
        traceback.print_exc()


def _navigate_to_pose(agent, target_x, target_y, target_yaw_deg=None,
                      kp_pos=1.5, kp_yaw=1.5, max_vel=0.6, max_yaw=1.5):
    """Navigate to a world-coordinate target using P-control body-frame commands.

    Args:
        agent: SimAgent instance.
        target_x, target_y: World-coordinate target position (metres).
        target_yaw_deg: Desired yaw in degrees.  None = face opponent goal.
        kp_pos: Position P-gain.
        kp_yaw: Yaw P-gain.
        max_vel: Max linear velocity (config-scaled unitless).
        max_yaw: Max angular velocity (config-scaled unitless).

    Returns:
        True when within 0.3 m and 0.2 rad of the target.
    """
    robot_pos = agent.get_self_pos()
    robot_yaw_deg = agent.get_self_yaw()
    if robot_pos is None or robot_pos[0] is None:
        agent.cmd_vel(0, 0, 0)
        return False

    robot_yaw_rad = math.radians(robot_yaw_deg)

    # World-frame position error
    err_wx = target_x - robot_pos[0]
    err_wy = target_y - robot_pos[1]

    # Rotate into body frame
    c = math.cos(robot_yaw_rad)
    s = math.sin(robot_yaw_rad)
    err_body_x = err_wx * c + err_wy * s
    err_body_y = -err_wx * s + err_wy * c

    # Yaw error
    if target_yaw_deg is not None:
        target_yaw_rad = math.radians(target_yaw_deg)
    else:
        league = agent.get_config().get("league", "M")
        field_dims = agent.get_config().get("field_size", {}).get(league, [14.0, 9.0])
        goal_x = float(field_dims[0]) / 2.0
        target_yaw_rad = math.atan2(0.0 - robot_pos[1], goal_x - robot_pos[0])

    yaw_err = agent.angle_normalize(target_yaw_rad - robot_yaw_rad)

    # P-control with clipping
    cmd_x = float(np.clip(kp_pos * err_body_x, -max_vel, max_vel))
    cmd_y = float(np.clip(kp_pos * err_body_y, -max_vel, max_vel))
    cmd_w = float(np.clip(kp_yaw * yaw_err, -max_yaw, max_yaw))

    agent.cmd_vel(cmd_x, cmd_y, cmd_w)
    agent.move_head(math.inf, math.inf)

    dist = math.hypot(err_wx, err_wy)
    return dist < 0.3 and abs(yaw_err) < 0.2


def _support_role(agent):
    """Support role (id=1): position behind the ball on the ball-to-own-goal line.

    Stays ~1.2 m behind the ball so the attacker can push forward unimpeded.
    Stops if the ball is closer than 1.0 m to avoid interfering.
    """
    ball_map = agent.get_ball_pos_in_map()
    robot_pos = agent.get_self_pos()

    if ball_map is None or ball_map[0] is None or robot_pos is None or robot_pos[0] is None:
        agent.cmd_vel(0, 0, 0)
        return

    league = agent.get_config().get("league", "M")
    field_dims = agent.get_config().get("field_size", {}).get(league, [14.0, 9.0])
    field_length = float(field_dims[0])
    field_width = float(field_dims[1])
    own_goal_x = -field_length / 2.0

    # Direction from ball toward own goal
    dx = own_goal_x - ball_map[0]
    dy = 0.0 - ball_map[1]
    dist_goal = math.hypot(dx, dy)
    if dist_goal < 0.01:
        dx, dy = -1.0, 0.0
        dist_goal = 1.0

    # Target: 1.2 m behind ball on the ball-to-own-goal line
    support_dist = 1.2
    target_x = ball_map[0] + (dx / dist_goal) * support_dist
    target_y = ball_map[1] + (dy / dist_goal) * support_dist

    # Clamp to field bounds
    half_w = field_width / 2.0
    target_x = max(-field_length / 2.0, min(field_length / 2.0, target_x))
    target_y = max(-half_w, min(half_w, target_y))

    # Stay away from the ball
    if agent.get_if_ball() and agent.get_ball_distance() < 1.0:
        agent.cmd_vel(0, 0, 0)
        return

    _navigate_to_pose(agent, target_x, target_y, max_vel=0.4, kp_pos=1.2)


def _defender_role(agent):
    """Defender role (id=2): hold position in own half, follow ball Y laterally.

    Anchors at X = -2 m (own half), tracks ball Y clamped to +/- 3 m.
    Moves backward if the ball gets closer than 2.0 m.
    """
    ball_map = agent.get_ball_pos_in_map()
    robot_pos = agent.get_self_pos()

    if robot_pos is None or robot_pos[0] is None:
        agent.cmd_vel(0, 0, 0)
        return

    league = agent.get_config().get("league", "M")
    field_dims = agent.get_config().get("field_size", {}).get(league, [14.0, 9.0])
    field_length = float(field_dims[0])
    field_width = float(field_dims[1])

    # Anchor in own half (~2/7 of own-half depth from centre)
    anchor_x = -field_length * 0.15

    # Follow ball Y, damped
    ball_y = 0.0
    if ball_map is not None and ball_map[0] is not None:
        ball_y = ball_map[1]

    half_w = field_width / 2.0
    target_y = float(np.clip(ball_y, -3.0, 3.0))
    target_x = anchor_x

    # Back away if ball is too close
    if agent.get_if_ball() and agent.get_ball_distance() < 2.0:
        agent.cmd_vel(-0.3, 0, 0)
        return

    _navigate_to_pose(agent, target_x, target_y, max_vel=0.35, kp_pos=1.0)


def game(agent) -> None:
    if getattr(agent, "is_simulation", False):
        role = getattr(agent, "_player_id", getattr(agent, "id", 0))
        if role == 0:
            # Attacker: find_ball + continuous push-to-goal controller
            if not agent.get_if_ball():
                agent.state_machine_runners["find_ball"]()
            else:
                agent.continuous_push.run()
        elif role == 1:
            _support_role(agent)
        elif role == 2:
            _defender_role(agent)
        else:
            # Fallback for extra ids: treat as attacker
            if not agent.get_if_ball():
                agent.state_machine_runners["find_ball"]()
            else:
                agent.continuous_push.run()
        return

    # --- Select Test to Run ---
    # _playing_logic(agent)        # Default: Full Playing Logic
    # _test_adv_dribble(agent)     # TEST ARGUMENT: Using Advanced Dribble
    # _playing_logic(agent)
    _gc_test_go_back_to_field(agent)

def _gc_test_go_back_to_field(agent):
    """
    GameController test using go_back_to_field.
    """
    gc = agent.gamecontroller
    state = gc.game_state
    
    logger = agent.get_logger()

    
    if state == "STATE_INITIAL" or state == "STATE_READY":
        agent.state_machine_runners['go_back_to_field'](aim_x = 1.3, aim_y = 0.001, aim_yaw = 150.)
        return

    if state == "STATE_SET":
        logger.info("[GC_TEST] Action: STATE_SET -> stop")
        agent.stop()
        return

    elif state in ("STATE_FINISHED", "STATE_STANDBY"):
        logger.info(f"[GC_TEST] Action: {state} -> stop")
        agent.stop()
        return

    if state == "STATE_PLAYING":
        logger.info("[GC_TEST] Action: STATE_PLAYING -> active behavior")
        
        # 1. Look for ball
        if not agent.get_if_ball():
            logger.info("[GC_TEST] -> find_ball (ball not detected)")
            agent.state_machine_runners['find_ball']()
            return

        # 2. Chase Ball
        ball_dist = agent.get_ball_distance()
        if getattr(agent, "is_simulation", False):
            _simple_sim_chase(agent, ball_dist)
            return

        if ball_dist > agent.default_chase_distance:
            logger.info(f"[GC_TEST] -> chase_ball (distance={ball_dist:.2f} > {agent.default_chase_distance:.2f})")
            agent.state_machine_runners['chase_ball']()
            return
    
        # 3. Ball Interaction (Close enough) -> NEW Dribble
        logger.info(f"[GC_TEST] -> adv_dribble (distance={ball_dist:.2f} <= {agent.default_chase_distance:.2f})")
        agent.state_machine_runners['adv_dribble']()
        return
    
    logger.warning(f"[GC_TEST] Unknown state: {state} -> stop")
    agent.stop()  

def _simple_sim_chase(agent, ball_dist: float) -> None:
    """K1 ball chasing with k1-point-navigate policy.

    Computes body-frame navigation commands from ball position (heading + distance),
    matching the training-time _compute_commands logic:
      cmd_vx = clip(0.45 * dist, 0, 0.8)
      cmd_wz = clip(1.2 * heading, -1.0, 1.0)
    """
    import sys
    logger = agent.get_logger()
    ball_pos = agent.get_ball_pos()
    if ball_pos is None or ball_pos[0] is None or ball_pos[1] is None:
        agent.state_machine_runners['find_ball']()
        return

    ball_x = float(ball_pos[0])
    ball_y = float(ball_pos[1])
    ball_angle = math.atan2(ball_y, ball_x)
    ball_dist = math.hypot(ball_x, ball_y)

    if ball_dist < 0.35:
        cmd_x, cmd_y, cmd_w = 0.0, 0.0, 0.0
    else:
        cmd_x = float(np.clip(0.45 * ball_dist, 0.0, 0.8))
        cmd_y = 0.0
        cmd_w = float(np.clip(1.2 * ball_angle, -1.0, 1.0))

    logger.info(
        f"[SIM_CHASE] ball=({ball_x:.2f},{ball_y:.2f}) dist={ball_dist:.2f} "
        f"angle={math.degrees(ball_angle):.1f} "
        f"cmd=({cmd_x:.2f},{cmd_y:.2f},{cmd_w:.2f})"
    )
    print(f"[CHASE] a={math.degrees(ball_angle):.1f} d={ball_dist:.2f} cmd=({cmd_x:.2f},{cmd_y:.2f},{cmd_w:.2f})", file=sys.stderr, flush=True)
    if getattr(agent, "is_simulation", False):
        agent.current_cmd = [cmd_x, cmd_y, cmd_w]
    else:
        agent.cmd_vel(cmd_x, cmd_y, cmd_w)
    agent.move_head(math.inf, math.inf)
    rpos = agent.get_self_pos()
    ryaw = agent.get_self_yaw()
    with open("/tmp/chase_trace.txt", "a") as f:
        f.write("cmd=(%.2f,%.2f,%.2f) ball=(%.2f,%.2f) dist=%.2f angle=%.1f rpos=(%.2f,%.2f) ryaw=%.2f\n" % (
            cmd_x, cmd_y, cmd_w, ball_x, ball_y, ball_dist, math.degrees(ball_angle),
            rpos[0] if rpos[0] is not None else -999, rpos[1] if rpos[1] is not None else -999, ryaw if ryaw is not None else -999))

def _test_agents(agent):
    """
    测试各状态机
    """
    # 3调用kick
    agent.state_machine_runners['find_ball']()

    
def _playing_logic(agent):
    """
    Simplified playing logic without GameController.
    """
    # 1. Look for ball
    if not agent.get_if_ball():
        agent.state_machine_runners['find_ball']()
        return

    # 2. Chase Ball
    if agent.get_ball_distance() > agent.default_chase_distance:
        agent.state_machine_runners['chase_ball']()
        return
    
    # 3. Ball Interaction (Close enough) -> NEW Dribble
    agent.state_machine_runners['adv_dribble']()

def _test_adv_dribble(agent) -> None:
    if not agent.get_if_ball():
        agent.state_machine_runners['find_ball']()
    else:
        agent.state_machine_runners['adv_dribble']()

def _test_dribble(agent) -> None:
    if not agent.get_if_ball():
        agent.state_machine_runners['find_ball']()
    else:
        agent.state_machine_runners['dribble'](aim_yaw=0)

def _test_find_ball(agent) -> None:
    agent.state_machine_runners['find_ball']()

def _test_chase_ball(agent) -> None:
    if not agent.get_if_ball():
        agent.state_machine_runners['find_ball']()
    else:
        agent.state_machine_runners['chase_ball']()
