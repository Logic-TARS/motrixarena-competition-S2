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
from logic.command_filter import CommandFilter

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
        self._flush_counter = 0

    def log(self, data):
        if not self.header_written:
            self.writer.writerow(data.keys())
            self.header_written = True
        self.writer.writerow(data.values())
        self._flush_counter += 1
        if self._flush_counter % 10 == 0:
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
        
        # 3. Final Command
        self.logger.info(f"[AdvDribble] Safe:{is_safe_zone} Alg:{aligned} T_Ang:{target_angle_deg:.1f} Cmd:({cmd_x:.2f}, {cmd_y:.2f}, {da:.2f})")
        self.agent.cmd_vel(cmd_x, cmd_y, da)
        self.agent.move_head(math.inf, math.inf)


class PushToGoalController:
    """Stay behind the ball (relative to opponent goal) and push it forward.

    Two-stage distance control:
      - setup_distance (default 0.35 m): used when robot is not yet aligned
        behind the ball — stays further back to avoid touching the ball
        while positioning.
      - dribble_distance (default 0.20 m): used when the robot is aligned
        behind the ball and ready to push — closer for controlled dribbling.

    The ``run()`` method accepts an optional *aligned* parameter.  When
    omitted, alignment is detected automatically from body-frame ball
    position and yaw error.
    """

    def __init__(self, agent):
        self.agent = agent
        self.logger = agent.get_logger().get_child("PushToGoal")

        config = agent.get_config()
        league = config.get("league", "M")
        field_dims = config.get("field_size", {}).get(league, [14.0, 9.0])
        self.field_length = float(field_dims[0])

        # Two-stage distances (was single push_dist = 0.12)
        self.setup_distance = 0.35    # further back when not aligned
        self.dribble_distance = 0.20  # closer when aligned behind ball
        self.kp_pos = 2.5             # position P-gain
        self.kp_yaw = 2.5             # yaw P-gain
        self.max_vel = 1.0            # max linear speed (m/s)
        self.max_yaw = 2.5            # max angular speed (rad/s)

    def is_aligned_behind_ball(self, ball_pos, yaw_err):
        """Return True when the robot is behind the ball facing the goal.

        Checks that the ball is at a reasonable distance directly in front
        and the robot is roughly facing the opponent goal.
        """
        if ball_pos is None or ball_pos[0] is None:
            return False
        bx, by = float(ball_pos[0]), float(ball_pos[1])
        return (
            0.15 < bx < 0.40
            and abs(by) < 0.10
            and abs(yaw_err) < math.radians(25)
        )

    def run(self, aligned=None):
        """Compute and send a velocity command for one frame.

        Args:
            aligned: Optional[bool].  If None, alignment is auto-detected.
        """
        # 1. No ball – stop
        if not self.agent.get_if_ball():
            self.logger.info("[PushToGoal] No ball.")
            self.agent.cmd_vel(0, 0, 0)
            return

        # 2. World-frame positions
        ball_w = self.agent.get_ball_pos_in_map()
        robot_w = self.agent.get_self_pos()
        robot_yaw_deg = self.agent.get_self_yaw()
        robot_yaw_rad = math.radians(robot_yaw_deg)

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
            to_goal = np.array([1.0, 0.0])
        else:
            to_goal = to_goal / dist_to_goal

        # 4. Yaw error (face the goal)
        target_yaw = math.atan2(goal[1] - robot_w[1], goal[0] - robot_w[0])
        yaw_err = self.agent.angle_normalize(target_yaw - robot_yaw_rad)

        # 5. Choose target distance based on alignment
        if aligned is None:
            ball_pos = self.agent.get_ball_pos()
            aligned = self.is_aligned_behind_ball(ball_pos, yaw_err)
        push_dist = self.dribble_distance if aligned else self.setup_distance

        # 6. Target position: behind ball, opposite to goal direction
        target_pos = ball_w - to_goal * push_dist

        # 7. Position error (world frame → body frame)
        err_world = target_pos - robot_w
        c = math.cos(robot_yaw_rad)
        s = math.sin(robot_yaw_rad)
        err_body_x = err_world[0] * c + err_world[1] * s
        err_body_y = -err_world[0] * s + err_world[1] * c

        # 8. P-control
        cmd_x = float(np.clip(self.kp_pos * err_body_x, -self.max_vel, self.max_vel))
        cmd_y = float(np.clip(self.kp_pos * err_body_y, -self.max_vel, self.max_vel))
        cmd_w = float(np.clip(self.kp_yaw * yaw_err, -self.max_yaw, self.max_yaw))

        self.logger.info(
            f"[PushToGoal] aligned={aligned} push_dist={push_dist:.2f} "
            f"err_body=({err_body_x:.2f},{err_body_y:.2f}) "
            f"yaw_err={math.degrees(yaw_err):.1f}deg "
            f"cmd=({cmd_x:.2f},{cmd_y:.2f},{cmd_w:.2f})"
        )
        self.agent.cmd_vel(cmd_x, cmd_y, cmd_w)
        self.agent.move_head(math.inf, math.inf)


class DeciderFSM:
    """Top-level decision FSM for robot soccer behaviour.

    States
    ------
    STOP              – game over / standby, zero velocity.
    RETURN_TO_FIELD   – walk to the ready position (STATE_READY).
    SEARCH_BALL       – rotate in place to find the ball.
    APPROACH_BALL     – walk toward the ball from behind.
    ALIGN_BEHIND_BALL – fine-position behind the ball facing the goal.
    DRIBBLE           – push the ball toward the opponent goal.
    KICK              – execute a single kick, then wait.
    RECOVER           – brief pause after kicking to re-observe.

    Hysteresis
    ----------
    Entering DRIBBLE uses tighter thresholds; exiting (→APPROACH_BALL)
    requires ``ball_dist > 1.05`` to prevent oscillation.

    Ball visibility is debounced: *seen_ball_count* must reach 3 before
    the ball is considered found, and *lost_ball_count* must reach 5
    before it is considered lost.
    """

    # -- config keys -------------------------------------------------
    CFG = "decider_fsm"

    def __init__(self, agent):
        self.agent = agent
        self.logger = agent.get_logger().get_child("DeciderFSM")

        cfg = agent.get_config().get(self.CFG, {})

        # --- thresholds (with defaults) ---
        self._ball_seen_frames = int(cfg.get("ball_seen_confirm_frames", 3))
        self._ball_lost_frames = int(cfg.get("ball_lost_confirm_frames", 5))
        self._approach_to_align_dist = float(cfg.get("approach_to_align_dist", 0.75))
        self._approach_to_align_angle_deg = float(cfg.get("approach_to_align_angle_deg", 35.0))
        self._align_ball_x_min = float(cfg.get("align_to_dribble_ball_x_min", 0.15))
        self._align_ball_x_max = float(cfg.get("align_to_dribble_ball_x_max", 0.40))
        self._align_ball_y_max = float(cfg.get("align_to_dribble_ball_y_max", 0.10))
        self._align_facing_err_deg = float(cfg.get("align_to_dribble_facing_error_deg", 25.0))
        self._dribble_exit_dist = float(cfg.get("dribble_to_approach_dist", 1.05))
        self._kick_ball_x_min = float(cfg.get("kick_ball_x_min", 0.15))
        self._kick_ball_x_max = float(cfg.get("kick_ball_x_max", 0.35))
        self._kick_ball_y_max = float(cfg.get("kick_ball_y_max", 0.08))
        self._kick_facing_err_deg = float(cfg.get("kick_facing_error_deg", 15.0))
        self._kick_max_dist_to_goal = float(cfg.get("kick_max_dist_to_goal", 2.0))
        self._kick_cooldown_sec = float(cfg.get("kick_cooldown_sec", 2.0))
        self._kick_ball_dist_max = float(cfg.get("kick_ball_dist_max", 0.35))
        self._kick_action_duration_sec = float(cfg.get("kick_action_duration_sec", 0.5))
        self._kick_settle_duration_sec = float(cfg.get("kick_settle_duration_sec", 0.3))
        self._recover_duration_sec = float(cfg.get("recover_duration_sec", 0.5))
        self._search_timeout_sec = float(cfg.get("search_timeout_sec", 15.0))

        # --- cached field dimensions ---
        league = agent.get_config().get("league", "M")
        self._field_len = float(agent.get_config().get("field_size", {}).get(league, [14.0, 9.0])[0])

        # --- state ---
        self.state = "SEARCH_BALL"
        self.state_enter_time = time.time()
        self.seen_ball_count = 0
        self.lost_ball_count = 0
        self.search_start_time = time.time()
        self.search_direction = 1  # +1 = CCW, -1 = CW
        self.last_kick_time = -999.0
        self.kick_triggered = False
        self._frame_counter = 0

        # --- command filter ---
        self.cmd_filter = CommandFilter(agent.get_config())

        # --- skill controllers (reuse agent-level instances) ---
        self.push_ctrl = agent.push_to_goal
        self.adv_dribbler = agent.adv_dribbler

        self.logger.info(
            f"[DeciderFSM] initialized state={self.state}"
        )

    # -- public API ---------------------------------------------------

    def switch(self, new_state):
        if self.state != new_state:
            self.logger.info(
                f"[FSM] {self.state} -> {new_state} "
                f"(seen={self.seen_ball_count} lost={self.lost_ball_count})"
            )
            self.state = new_state
            self.state_enter_time = time.time()
            # Reset filter only on "hard" state changes (stop, search, recover)
            # where a clean velocity restart is desirable.  Soft transitions
            # (APPROACH→ALIGN→DRIBBLE) keep the filter state for smooth
            # continuity — avoids a one-frame velocity dip.
            if new_state in ("STOP", "SEARCH_BALL", "RECOVER",
                             "RETURN_TO_FIELD", "KICK"):
                self.cmd_filter.reset()

    def tick(self):
        """Run one frame of the decision FSM.  Returns (vx, vy, w)."""
        self._frame_counter += 1

        # 1. Update ball visibility counters --------------------------
        if self.agent.get_if_ball():
            self.seen_ball_count += 1
            self.lost_ball_count = 0
        else:
            self.lost_ball_count += 1
            self.seen_ball_count = 0

        # 2. GameController priority ----------------------------------
        gc_state = self.agent.gamecontroller.game_state

        if gc_state in ("STATE_FINISHED", "STATE_STANDBY"):
            self.switch("STOP")
            return self._do_stop()

        if gc_state == "STATE_READY":
            self.switch("RETURN_TO_FIELD")
            return self._do_return_to_field()

        if gc_state in ("STATE_INITIAL", "STATE_SET"):
            return self._do_stop()

        # 3. Lost-ball protection (skip states that manage it themselves)
        if self.state not in ("SEARCH_BALL", "KICK", "RECOVER", "STOP",
                              "RETURN_TO_FIELD"):
            if self.lost_ball_count >= self._ball_lost_frames:
                self.switch("SEARCH_BALL")

        # 4. State dispatch --------------------------------------------
        if self.state == "SEARCH_BALL":
            return self._do_search_ball()
        elif self.state == "APPROACH_BALL":
            return self._do_approach_ball()
        elif self.state == "ALIGN_BEHIND_BALL":
            return self._do_align_behind_ball()
        elif self.state == "DRIBBLE":
            return self._do_dribble()
        elif self.state == "KICK":
            return self._do_kick()
        elif self.state == "RECOVER":
            return self._do_recover()
        elif self.state == "RETURN_TO_FIELD":
            return self._do_return_to_field()
        elif self.state == "STOP":
            return self._do_stop()
        else:
            self.switch("SEARCH_BALL")
            return 0.0, 0.0, 0.0

    # -- state handlers -----------------------------------------------

    def _do_stop(self):
        return self._emit(0.0, 0.0, 0.0)

    def _do_return_to_field(self):
        # Reuse existing go_back machine (per-frame, non-blocking after fix)
        go_back = getattr(self.agent, "go_back_machine", None)
        if go_back is None:
            self.logger.warning(
                "[RETURN_TO_FIELD] go_back_machine not initialized, stopping."
            )
            return self._emit(0.0, 0.0, 0.0)
        go_back.run(aim_x=1.3, aim_y=0.0, aim_yaw=150.0)
        # Read the command set by go_back_machine
        cmd = self.agent.current_cmd
        return self._emit(cmd[0], cmd[1], cmd[2])

    def _do_search_ball(self):
        # Transition check: confirmed ball
        if self.seen_ball_count >= self._ball_seen_frames:
            self.switch("APPROACH_BALL")
            return self._emit(0.0, 0.0, 0.0)

        # Timeout: flip search direction by overriding the
        # find_ball_machine's internal rotation-direction state.
        elapsed = time.time() - self.search_start_time
        if elapsed > self._search_timeout_sec:
            self.search_direction *= -1
            self.search_start_time = time.time()
            # Push the flipped direction into the find_ball machine so it
            # actually changes rotation on the next tick.
            self.agent.find_ball_machine.last_rotation = self.search_direction
            self.logger.info(
                f"[Search] timeout {elapsed:.1f}s, "
                f"flip dir={self.search_direction}"
            )

        # Rotate in place — reuse find_ball machine (per-frame)
        self.agent.find_ball_machine.run()
        cmd = self.agent.current_cmd
        return self._emit(cmd[0], cmd[1], cmd[2])

    def _do_approach_ball(self):
        ball_pos = self.agent.get_ball_pos()
        if ball_pos is None or ball_pos[0] is None:
            self.switch("SEARCH_BALL")
            return self._emit(0.0, 0.0, 0.0)

        bx = float(ball_pos[0])
        by = float(ball_pos[1])
        ball_dist = math.hypot(bx, by)
        ball_angle = math.atan2(by, bx)

        # Fast path: can we kick right now?
        if self._can_kick():
            self.switch("KICK")
            return self._emit(0.0, 0.0, 0.0)

        # Transition: close enough and roughly facing → fine-position behind ball
        if (ball_dist < self._approach_to_align_dist
                and abs(ball_angle) < math.radians(self._approach_to_align_angle_deg)):
            self.switch("ALIGN_BEHIND_BALL")
            return self._emit(0.0, 0.0, 0.0)

        # Simple chase: go directly toward the ball (not behind it).
        # NOTE: velocities here are generated directly in the normalized
        # [-1,1] space — they do NOT go through agent.cmd_vel() scaling.
        # This is intentional: we want higher turn rates (up to ±1.5 rad/s
        # via CommandFilter w_max) than the 1.0 cap cmd_vel would impose.
        kp_dist = 1.5
        kp_angle = 2.0
        kp_lat = 1.5

        if bx > 0.05:
            # Ball in front: drive forward while steering + sidestep
            cmd_x = float(np.clip(kp_dist * ball_dist, 0.0, 0.8))
            cmd_y = float(np.clip(kp_lat * by, -0.35, 0.35))
            cmd_w = float(np.clip(kp_angle * ball_angle, -1.5, 1.5))
        else:
            # Ball behind / at side: turn in place (no lateral needed)
            cmd_x = 0.0
            cmd_y = 0.0
            cmd_w = float(np.clip(kp_angle * ball_angle, -1.5, 1.5))

        self.logger.info(
            f"[APPROACH] chase: dist={ball_dist:.2f} angle={math.degrees(ball_angle):.1f}deg "
            f"cmd=({cmd_x:.2f},{cmd_y:.2f},{cmd_w:.2f})"
        )
        return self._emit(cmd_x, cmd_y, cmd_w)

    def _do_align_behind_ball(self):
        # Fast path: can we kick right now?
        if self._can_kick():
            self.switch("KICK")
            return self._emit(0.0, 0.0, 0.0)

        ball_dist = self.agent.get_ball_distance()

        # Fallback: ball too far → re-approach
        if ball_dist > self._dribble_exit_dist:
            self.switch("APPROACH_BALL")
            return self._emit(0.0, 0.0, 0.0)

        # Check alignment
        ball_pos = self.agent.get_ball_pos()
        yaw_err = self._goal_facing_error()

        aligned = (
            ball_pos is not None
            and ball_pos[0] is not None
            and yaw_err is not None
            and self._align_ball_x_min < float(ball_pos[0]) < self._align_ball_x_max
            and abs(float(ball_pos[1])) < self._align_ball_y_max
            and abs(yaw_err) < math.radians(self._align_facing_err_deg)
        )

        if aligned:
            self.switch("DRIBBLE")
            return self._emit(0.0, 0.0, 0.0)

        # Keep positioning behind the ball (not aligned → setup distance)
        self.push_ctrl.run(aligned=False)
        cmd = self.agent.current_cmd
        return self._emit(cmd[0], cmd[1], cmd[2])

    def _do_dribble(self):
        ball_dist = self.agent.get_ball_distance()

        # Hysteresis exit: ball drifted too far
        if ball_dist > self._dribble_exit_dist:
            self.switch("APPROACH_BALL")
            return self._emit(0.0, 0.0, 0.0)

        # Check kick window
        if self._can_kick():
            self.switch("KICK")
            return self._emit(0.0, 0.0, 0.0)

        # Use advanced dribbler (existing tick-based implementation)
        self.adv_dribbler.run()
        cmd = self.agent.current_cmd

        # Fallback: if ball is close and in front but dribbler produces
        # near-zero forward speed (e.g. forward_factor=0 due to angle
        # misalignment), push straight forward to touch the ball so
        # _can_kick() can trigger on the next frame.
        # Zero out vy/w to avoid arcing (original w may be large from turn mode).
        ball_pos = self.agent.get_ball_pos()
        if (ball_pos is not None and ball_pos[0] is not None
                and float(ball_pos[0]) > 0.08 and ball_dist < 0.30
                and abs(cmd[0]) < 0.05):
            self.logger.info("[DRIBBLE] fallback push: overriding near-zero vx")
            return self._emit(0.5, 0.0, 0.0)

        return self._emit(cmd[0], cmd[1], cmd[2])

    def _do_kick(self):
        now = time.time()
        elapsed = now - self.state_enter_time

        # Phase 1: Settle — wait for robot to stop before kicking
        if elapsed < self._kick_settle_duration_sec:
            return self._emit(0.0, 0.0, 0.0)

        # Phase 2: Trigger the kick (once after settle)
        if not self.kick_triggered:
            self.kick_triggered = True
            self.last_kick_time = now
            # Choose foot based on ball lateral position (body-frame Y).
            # ball_y > 0 → ball is left  → use left foot  (foot=1)
            # ball_y ≤ 0 → ball is right → use right foot (foot=0)
            ball_pos = self.agent.get_ball_pos()
            foot = 1 if (ball_pos is not None and ball_pos[1] is not None
                         and float(ball_pos[1]) > 0) else 0
            self.logger.info(f"[KICK] Triggering kick! foot={foot}")
            self.agent.kick(foot=foot, death=0)
            return self._emit(0.0, 0.0, 0.0)

        # Phase 3: Wait for kick to complete
        if now - self.last_kick_time > self._kick_action_duration_sec:
            self.kick_triggered = False
            self.switch("RECOVER")
            return self._emit(0.0, 0.0, 0.0)

        return self._emit(0.0, 0.0, 0.0)

    def _do_recover(self):
        elapsed = time.time() - self.state_enter_time

        if elapsed > self._recover_duration_sec:
            if self.seen_ball_count >= self._ball_seen_frames:
                self.switch("APPROACH_BALL")
            else:
                self.switch("SEARCH_BALL")
            return self._emit(0.0, 0.0, 0.0)

        return self._emit(0.0, 0.0, 0.0)

    # -- helpers ------------------------------------------------------

    def _emit(self, vx, vy, w):
        """Filter command and write to agent.

        .. note::

           The values arriving here have already been scaled by
           ``SimAgent.cmd_vel()`` (multiplied by ``max_walk_vel_*`` and
           clipped to [-1, 1]).  CommandFilter operates on these
           *post-scaling* values, so its ``vx_max`` is the **effective**
           forward speed limit.  Tuning ``max_walk_vel_x`` in config has
           no effect unless it is *lower* than ``cmd_filter.vx_max``.
        """
        vx, vy, w = self.cmd_filter.apply(vx, vy, w)
        if getattr(self.agent, "is_simulation", False):
            self.agent.current_cmd = [vx, vy, w]
        else:
            self.agent.cmd_vel(vx, vy, w)
        return vx, vy, w

    def _goal_facing_error(self):
        """Return yaw error (radians) between robot heading and goal direction.

        Returns:
            float or None: yaw error in radians, or None if position is unknown.
        """
        robot_w = self.agent.get_self_pos()
        if robot_w is None or robot_w[0] is None:
            return None
        goal = np.array([self._field_len / 2.0, 0.0])
        target_yaw = math.atan2(goal[1] - float(robot_w[1]),
                                goal[0] - float(robot_w[0]))
        robot_yaw_rad = math.radians(float(self.agent.get_self_yaw()))
        return self.agent.angle_normalize(target_yaw - robot_yaw_rad)

    def _can_kick(self):
        """True when the robot is in a good position to kick toward goal."""
        now = time.time()
        ball_pos = self.agent.get_ball_pos()
        if ball_pos is None or ball_pos[0] is None:
            return False

        bx, by = float(ball_pos[0]), float(ball_pos[1])
        ball_dist = math.hypot(bx, by)
        yaw_err = self._goal_facing_error()
        if yaw_err is None:
            return False

        # Cooldown
        if now - self.last_kick_time < self._kick_cooldown_sec:
            return False

        # Distance to goal
        robot_w = self.agent.get_self_pos()
        if robot_w is None or robot_w[0] is None:
            return False
        dist_to_goal = self._field_len / 2.0 - float(robot_w[0])

        return (
            self._kick_ball_x_min < bx < self._kick_ball_x_max
            and abs(by) < self._kick_ball_y_max
            and abs(yaw_err) < math.radians(self._kick_facing_err_deg)
            and dist_to_goal < self._kick_max_dist_to_goal
            and ball_dist < self._kick_ball_dist_max
        )

    def _debug_log(self):
        """Print a compact status line every 5 frames."""
        if self._frame_counter % 5 != 0:
            return
        ball_seen = self.agent.get_if_ball()
        ball_dist = self.agent.get_ball_distance()
        ball_angle = self.agent.get_ball_angle()
        cmd = getattr(self.agent, "current_cmd", [0, 0, 0])
        # Compute angle string safely: ball_angle may be None (no ball)
        # or 0.0 (ball directly ahead, which is falsy).
        if ball_angle is not None:
            angle_str = f"{math.degrees(ball_angle):.1f}"
        else:
            angle_str = "None"
        self.logger.info(
            f"[FSM] state={self.state} "
            f"ball_seen={ball_seen} dist={ball_dist:.2f} "
            f"angle={angle_str}deg "
            f"seen_cnt={self.seen_ball_count} lost_cnt={self.lost_ball_count} "
            f"cmd=({cmd[0]:.2f},{cmd[1]:.2f},{cmd[2]:.2f})"
        )


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

    # Initialize Push-to-Goal controller
    agent.push_to_goal = PushToGoalController(agent)

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

def game(agent) -> None:
    """Top-level game logic — dispatched every frame (~50 Hz).

    Simulation mode uses the DeciderFSM state machine.
    ROS mode falls through to the legacy GameController test path.
    """
    if getattr(agent, "is_simulation", False):
        # --- Simulation: FSM-based decision making ---
        decider = getattr(agent, "_decider_fsm", None)
        if decider is None:
            agent._decider_fsm = DeciderFSM(agent)
            decider = agent._decider_fsm
        vx, vy, w = decider.tick()
        # _emit() already writes to agent.current_cmd
        decider._debug_log()
        return

    # --- ROS mode: legacy path (unchanged) ---
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
    if os.environ.get("SIM_CHASE_TRACE"):
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
