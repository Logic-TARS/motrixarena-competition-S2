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
from dataclasses import dataclass

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


def resolve_field_size(config):
    """Return the active (length, width), preferring simulator runtime data."""
    active_size = config.get("active_field_size")
    if isinstance(active_size, (list, tuple)) and len(active_size) >= 2:
        try:
            length = float(active_size[0])
            width = float(active_size[1])
            if length > 0.0 and width > 0.0:
                return length, width
        except (TypeError, ValueError):
            pass

    league = config.get("league", "M")
    field_dims = config.get("field_size", {}).get(league, [14.0, 9.0])
    return float(field_dims[0]), float(field_dims[1])


class AdvancedDribbler:
    def __init__(self, agent):
        self.agent = agent
        self.logger = agent.get_logger().get_child("AdvDribble")

        fsm_cfg = agent.get_config().get("decider_fsm", {})
        self.verbose_log = bool(fsm_cfg.get("verbose_log", False))
        self.record_debug_csv = bool(fsm_cfg.get("dribble_debug_csv", False))
        self.recorder = None
        if self.record_debug_csv:
            # Use project-relative debug_logs directory (parent of `decider`).
            log_dir = os.path.abspath(os.path.join(CUR_DIR, '..', 'debug_logs'))
            self.logger.info(f"[AdvDribble] DataRecorder log_dir: {log_dir}")
            self.recorder = DataRecorder(log_dir)
        
        # Parameters
        self.bturn_p = 2.0
        self.side_correction_p = 2.5
        self.forward_p = 1.0
        
        self.setup_dist = 0.40
        self.dribble_dist = 0.20 # Ball should be slightly in front
        self.max_fw_vel = 0.8
        
        config = agent.get_config()
        self.field_length, self.field_width = resolve_field_size(config)
        
        # Anti-Oscillation
        self.spread_factor_max = 20.0 # degrees
        self.spread_factor_min = 5.0 # degrees

        # Hysteresis to avoid mode chattering near b_x threshold
        self.turn_to_ball_enter_bx = 0.03
        self.turn_to_ball_exit_bx = 0.08
        self.turn_to_ball_mode = False
        
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

        ball_pos = self.agent.get_ball_pos_in_map()
        if ball_pos is not None and ball_pos[0] is not None:
            target_origin = ball_pos
        else:
            target_origin = my_pos

        # The desired shooting direction is defined by ball -> goal.
        g_dx = goal_x - target_origin[0]
        g_dy = goal_y - target_origin[1]
        
        # 2. Boundary Repulsion (Side Lines are at Y = +/- W/2)
        # Use the same origin as goal attraction. When the ball is available,
        # its boundary risk determines the desired dribbling direction.
        origin_y = float(target_origin[1])
        dist_to_left = (self.field_width / 2.0) - origin_y   # Y+ is left
        dist_to_right = origin_y - (-self.field_width / 2.0)  # Y- is right
        
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

    def compute_command(self):
        if not self.agent.get_if_ball():
            self.logger.info("Lost ball, stopping.")
            return 0.0, 0.0, 0.0

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
        if self.verbose_log:
            self.logger.info(
                f"[AdvDribble] ball_angle_to_robot={ball_angle_to_robot:.4f}rad "
                f"({ball_angle_deg:.1f}deg), b=({b_x:.3f}, {b_y:.3f})"
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
            
            # Log for debugging
            if self.recorder is not None:
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
            return approach_speed, 0.0, turn_speed
        
        # Target Vector
        target_vec_global, is_safe_zone = self.get_target_vector()
        
        # [DEBUG] Log coordinate values for diagnosis
        if self.verbose_log:
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
        da = self.bturn_p * target_angle_local
        
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
        
        # Near-goal mode is based on distance to the real active goal.
        ball_w = self.agent.get_ball_pos_in_map()
        near_goal = False
        if ball_w is not None and ball_w[0] is not None:
            goal_distance = math.hypot(
                self.field_length / 2.0 - float(ball_w[0]),
                -float(ball_w[1]),
            )
            near_goal = goal_distance < self.field_length * 0.15
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

        if self.recorder is not None:
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
        if self.verbose_log:
            self.logger.info(
                f"[AdvDribble] Safe:{is_safe_zone} Alg:{aligned} "
                f"T_Ang:{target_angle_deg:.1f} Cmd:({cmd_x:.2f}, {cmd_y:.2f}, {da:.2f})"
            )
        return cmd_x, cmd_y, da

    def run(self):
        cmd = self.compute_command()
        self.agent.cmd_vel(cmd[0], cmd[1], cmd[2])
        self.agent.move_head(math.inf, math.inf)
        return cmd


@dataclass(frozen=True)
class MotionIntent:
    """Raw command plus the filtering mode required by the controller."""

    vx: float = 0.0
    vy: float = 0.0
    w: float = 0.0
    mode: str = "NORMAL"

    def __iter__(self):
        return iter((self.vx, self.vy, self.w))

    def __getitem__(self, index):
        return (self.vx, self.vy, self.w)[index]


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
        self.verbose_log = bool(config.get("decider_fsm", {}).get("verbose_log", False))
        fsm_cfg = config.get("decider_fsm", {})
        self.field_length, self.field_width = resolve_field_size(config)

        # Two-stage distances (was single push_dist = 0.12)
        self.setup_distance = float(fsm_cfg.get("behind_ball_setup_dist", 0.35))
        self.dribble_distance = 0.20  # closer when aligned behind ball
        self.kp_pos = 2.5             # position P-gain
        self.kp_yaw = 2.5             # yaw P-gain
        self.max_yaw = 2.5            # max angular speed (rad/s)
        self.turn_only_yaw = math.radians(
            float(fsm_cfg.get("align_turn_only_yaw_deg", 35.0))
        )
        self.direct_vx_max = float(fsm_cfg.get("align_direct_vx_max", 0.45))
        self.navigation_min_vx = float(
            fsm_cfg.get("navigation_min_vx", 0.24)
        )
        self.orbit_clearance = float(fsm_cfg.get("orbit_path_clearance", 0.28))
        self.orbit_side_offset = float(
            fsm_cfg.get("orbit_side_lateral_offset", 0.50)
        )
        self.orbit_behind_offset = float(
            fsm_cfg.get("orbit_behind_lateral_offset", 0.35)
        )
        self.orbit_waypoint_tol = float(
            fsm_cfg.get("orbit_waypoint_tolerance", 0.12)
        )
        self.orbit_side_deadband = float(
            fsm_cfg.get("orbit_side_deadband", 0.05)
        )
        self.orbit_side = None
        self.orbit_phase = None
        self.escape_phase = None
        self.escape_yaw = None
        self.escape_enter_dist = float(
            fsm_cfg.get("align_escape_enter_dist", 0.70)
        )
        self.escape_exit_dist = float(
            fsm_cfg.get("align_escape_exit_dist", 0.78)
        )
        self.escape_vx = float(fsm_cfg.get("align_escape_vx", 0.30))
        self.escape_face_enter = math.radians(
            float(fsm_cfg.get("align_escape_face_enter_deg", 10.0))
        )
        self.escape_face_exit = math.radians(
            float(fsm_cfg.get("align_escape_face_exit_deg", 18.0))
        )
        self.last_mode = "ENTRY_STOP"
        self.last_target = None

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

    @staticmethod
    def _point_to_segment_distance(point, start, end):
        segment = end - start
        segment_len_sq = float(np.dot(segment, segment))
        if segment_len_sq < 1e-9:
            return float(np.linalg.norm(point - start)), 0.0
        projection = float(np.dot(point - start, segment) / segment_len_sq)
        clamped = float(np.clip(projection, 0.0, 1.0))
        closest = start + clamped * segment
        return float(np.linalg.norm(point - closest)), projection

    def _path_crosses_ball(self, robot_w, target_pos, ball_w):
        distance, projection = self._point_to_segment_distance(
            ball_w, robot_w, target_pos
        )
        return 0.0 < projection < 1.0 and distance < self.orbit_clearance

    def _choose_orbit_side(self, robot_w, ball_w, lateral_unit):
        signed_lateral = float(np.dot(robot_w - ball_w, lateral_unit))
        if abs(signed_lateral) >= self.orbit_side_deadband:
            return 1.0 if signed_lateral > 0.0 else -1.0

        if abs(float(ball_w[1])) > 1e-6:
            toward_center = np.array([0.0, -math.copysign(1.0, ball_w[1])])
            center_dot = float(np.dot(lateral_unit, toward_center))
            if abs(center_dot) > 1e-6:
                return 1.0 if center_dot > 0.0 else -1.0
        return 1.0

    def compute_pose_command(
        self,
        target_pos,
        target_yaw,
        vx_max,
        turn_only_yaw=None,
    ):
        """Navigate toward a world target using turn-then-forward motion."""
        robot_w = self.agent.get_self_pos()
        if robot_w is None or robot_w[0] is None:
            return MotionIntent()
        robot_w = np.array(robot_w, dtype=float)
        robot_yaw_rad = math.radians(float(self.agent.get_self_yaw()))
        yaw_err = self.agent.angle_normalize(target_yaw - robot_yaw_rad)
        cmd_w = float(np.clip(self.kp_yaw * yaw_err, -self.max_yaw, self.max_yaw))
        if turn_only_yaw is not None and abs(yaw_err) > turn_only_yaw:
            return MotionIntent(w=cmd_w, mode="TURN_ONLY")

        distance = float(np.linalg.norm(np.array(target_pos, dtype=float) - robot_w))
        cmd_x = min(
            float(vx_max),
            max(self.navigation_min_vx, self.kp_pos * distance),
        )
        return MotionIntent(vx=cmd_x, w=cmd_w)

    def reset_orbit(self):
        self.orbit_side = None
        self.orbit_phase = None
        self.escape_phase = None
        self.escape_yaw = None
        self.last_mode = "ENTRY_STOP"
        self.last_target = None

    def _orbit_target(self, phase, ball_w, behind_target, lateral_unit):
        if phase == "ORBIT_SIDE":
            return (
                ball_w
                + self.orbit_side * lateral_unit * self.orbit_side_offset
            )
        if phase == "ORBIT_BEHIND":
            return (
                behind_target
                + self.orbit_side * lateral_unit * self.orbit_behind_offset
            )
        return behind_target

    def compute_command(self, aligned=None):
        """Compute one positioning intent for the ALIGN state.

        Args:
            aligned: Optional[bool].  If None, alignment is auto-detected.
        """
        if not self.agent.get_if_ball():
            self.logger.info("[PushToGoal] No ball.")
            return MotionIntent()

        ball_w = self.agent.get_ball_pos_in_map()
        robot_w = self.agent.get_self_pos()
        robot_yaw_deg = self.agent.get_self_yaw()
        robot_yaw_rad = math.radians(robot_yaw_deg)

        if ball_w is None or robot_w is None or ball_w[0] is None or robot_w[0] is None:
            self.logger.warning("[PushToGoal] Bad position data, stopping.")
            return MotionIntent()

        goal = np.array([self.field_length / 2.0, 0.0])
        ball_w = np.array(ball_w, dtype=float)
        robot_w = np.array(robot_w, dtype=float)

        to_goal = goal - ball_w
        dist_to_goal = float(np.linalg.norm(to_goal))
        if dist_to_goal < 0.01:
            to_goal = np.array([1.0, 0.0])
        else:
            to_goal = to_goal / dist_to_goal

        goal_yaw = math.atan2(to_goal[1], to_goal[0])
        goal_yaw_err = self.agent.angle_normalize(goal_yaw - robot_yaw_rad)

        if aligned is None:
            ball_pos = self.agent.get_ball_pos()
            aligned = self.is_aligned_behind_ball(ball_pos, goal_yaw_err)
        push_dist = self.dribble_distance if aligned else self.setup_distance

        ball_dist = float(np.linalg.norm(ball_w - robot_w))
        behind_target = ball_w - to_goal * push_dist
        lateral_unit = np.array([-to_goal[1], to_goal[0]], dtype=float)
        path_blocked = self._path_crosses_ball(robot_w, behind_target, ball_w)

        # Near-ball escape uses a trained positive forward command. The away
        # heading is locked for the whole escape to avoid frame-to-frame flips.
        if (
            self.escape_phase is None
            and self.orbit_phase is None
            and not path_blocked
            and float(np.linalg.norm(robot_w - behind_target))
            > self.orbit_waypoint_tol
            and ball_dist < self.escape_enter_dist
            and abs(goal_yaw_err) > self.turn_only_yaw
        ):
            away = robot_w - ball_w
            if float(np.linalg.norm(away)) > 1e-6:
                self.escape_yaw = math.atan2(away[1], away[0])
                self.escape_phase = "ESCAPE_FACE"

        if self.escape_phase is not None:
            if ball_dist >= self.escape_exit_dist:
                self.escape_phase = None
                self.escape_yaw = None
            else:
                escape_err = self.agent.angle_normalize(
                    self.escape_yaw - robot_yaw_rad
                )
                escape_w = float(np.clip(
                    self.kp_yaw * escape_err,
                    -self.max_yaw,
                    self.max_yaw,
                ))
                if self.escape_phase == "ESCAPE_FORWARD":
                    if abs(escape_err) > self.escape_face_exit:
                        self.escape_phase = "ESCAPE_FACE"
                    else:
                        self.last_mode = "ESCAPE_FORWARD"
                        self.last_target = None
                        return MotionIntent(vx=self.escape_vx, w=escape_w)
                if abs(escape_err) <= self.escape_face_enter:
                    self.escape_phase = "ESCAPE_FORWARD"
                    self.last_mode = "ESCAPE_FORWARD"
                    self.last_target = None
                    return MotionIntent(vx=self.escape_vx, w=escape_w)
                self.last_mode = "ESCAPE_FACE"
                self.last_target = None
                return MotionIntent(w=escape_w, mode="TURN_ONLY")

        # Select and lock the collision-free waypoint sequence before choosing
        # the heading. Orbit movement faces the waypoint, not the goal.
        if self.orbit_phase is None and path_blocked:
            if self.orbit_side is None:
                self.orbit_side = self._choose_orbit_side(
                    robot_w, ball_w, lateral_unit
                )
            self.orbit_phase = "ORBIT_SIDE"

        target_pos = behind_target
        if self.orbit_phase is not None:
            target_pos = self._orbit_target(
                self.orbit_phase, ball_w, behind_target, lateral_unit
            )
            target_reached = (
                float(np.linalg.norm(robot_w - target_pos))
                <= self.orbit_waypoint_tol
            )
            if target_reached:
                if self.orbit_phase == "ORBIT_SIDE":
                    candidate = self._orbit_target(
                        "ORBIT_BEHIND", ball_w, behind_target, lateral_unit
                    )
                    if not self._path_crosses_ball(robot_w, candidate, ball_w):
                        self.orbit_phase = "ORBIT_BEHIND"
                        target_pos = candidate
                elif self.orbit_phase == "ORBIT_BEHIND":
                    if not self._path_crosses_ball(
                        robot_w, behind_target, ball_w
                    ):
                        self.orbit_phase = None
                        self.orbit_side = None
                        target_pos = behind_target

        self.last_target = target_pos.copy()
        target_dist = float(np.linalg.norm(target_pos - robot_w))
        if target_dist <= self.orbit_waypoint_tol:
            self.last_mode = "DIRECT_FACE"
            face_w = float(np.clip(
                self.kp_yaw * goal_yaw_err,
                -self.max_yaw,
                self.max_yaw,
            ))
            return MotionIntent(w=face_w, mode="TURN_ONLY")

        travel_delta = target_pos - robot_w
        travel_yaw = math.atan2(travel_delta[1], travel_delta[0])
        intent = self.compute_pose_command(
            target_pos,
            travel_yaw,
            self.direct_vx_max,
            turn_only_yaw=self.turn_only_yaw,
        )
        position_mode = self.orbit_phase or "DIRECT_POSITION"
        self.last_mode = "TURN" if intent.mode == "TURN_ONLY" else position_mode

        if self.verbose_log:
            self.logger.info(
                f"[PushToGoal] mode={self.last_mode} aligned={aligned} "
                f"push_dist={push_dist:.2f} "
                f"target_dist={target_dist:.2f} "
                f"goal_yaw_err={math.degrees(goal_yaw_err):.1f}deg "
                f"cmd=({intent.vx:.2f},{intent.vy:.2f},{intent.w:.2f})"
            )
        return intent

    def run(self, aligned=None):
        intent = self.compute_command(aligned=aligned)
        self.agent.cmd_vel(intent.vx, intent.vy, intent.w)
        self.agent.move_head(math.inf, math.inf)
        return intent


class DeciderFSM:
    """Top-level decision FSM for robot soccer behaviour.

    States
    ------
    STOP              – game over / standby, zero velocity.
    RETURN_TO_FIELD   – walk to the ready position (STATE_READY).
    SEARCH_BALL       – rotate in place to find the ball.
    APPROACH_BALL     – walk toward the ball from behind.
    ALIGN_BEHIND_BALL – fine-position behind the ball facing the goal.
    SIDE_RECOVERY     – recover the robot or ball from a touchline.
    DRIBBLE           – push the ball toward the opponent goal.
    KICK              – execute a single kick, then wait.
    RECOVER           – brief pause after kicking to re-observe.
    FALL_RECOVERY      – suspend football behavior while the get-up policy runs.

    Hysteresis
    ----------
    Entering DRIBBLE uses tighter thresholds; exiting (→APPROACH_BALL)
    requires ``ball_dist > 0.9`` to prevent oscillation.

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
        self._approach_setup_dist = float(
            cfg.get("approach_behind_setup_dist", 0.75)
        )
        self._approach_waypoint_tol = float(
            cfg.get("approach_waypoint_tolerance", 0.18)
        )
        self._approach_far_ball_dist = float(
            cfg.get("approach_far_ball_dist", 1.50)
        )
        self._approach_far_vmax = float(cfg.get("approach_far_vmax", 0.65))
        self._approach_near_vmax = float(cfg.get("approach_near_vmax", 0.35))
        self._brake_translation_tol = float(
            cfg.get("brake_translation_tolerance", 0.02)
        )
        self._align_ball_x_min = float(cfg.get("align_to_dribble_ball_x_min", 0.15))
        self._align_ball_x_max = float(cfg.get("align_to_dribble_ball_x_max", 0.40))
        self._align_ball_y_max = float(cfg.get("align_to_dribble_ball_y_max", 0.10))
        self._align_facing_err_deg = float(cfg.get("align_to_dribble_facing_error_deg", 25.0))
        self._align_lateral_tol = float(
            cfg.get("align_to_dribble_lateral_tol", 0.05)
        )
        self._align_min_behind_depth = float(
            cfg.get("align_to_dribble_min_behind_depth", 0.22)
        )
        self._dribble_exit_dist = float(cfg.get("dribble_to_approach_dist", 1.05))
        self._kick_ball_x_min = float(cfg.get("kick_ball_x_min", 0.15))
        self._kick_ball_x_max = float(cfg.get("kick_ball_x_max", 0.35))
        self._kick_ball_y_max = float(cfg.get("kick_ball_y_max", 0.08))
        self._kick_facing_err_deg = float(cfg.get("kick_facing_error_deg", 15.0))
        self._kick_max_dist_to_goal = float(cfg.get("kick_max_dist_to_goal", 2.0))
        self._kick_cooldown_sec = float(cfg.get("kick_cooldown_sec", 2.0))
        self._kick_ball_dist_max = float(cfg.get("kick_ball_dist_max", 0.35))
        self._behind_ball_setup_dist = float(cfg.get("behind_ball_setup_dist", 0.35))
        self._behind_ball_lateral_tol = float(cfg.get("behind_ball_lateral_tol", 0.18))
        self._behind_ball_depth_tol = float(cfg.get("behind_ball_depth_tol", 0.08))
        self._kick_behind_ball_lateral_tol = float(
            cfg.get("kick_behind_ball_lateral_tol", 0.08)
        )
        self._sim_kick_push_vx = float(cfg.get("sim_kick_push_vx", 0.45))
        self._dribble_fallback_push_vx = float(cfg.get("dribble_fallback_push_vx", 0.30))
        self._kick_action_duration_sec = float(cfg.get("kick_action_duration_sec", 0.5))
        self._kick_settle_duration_sec = float(cfg.get("kick_settle_duration_sec", 0.3))
        self._recover_duration_sec = float(cfg.get("recover_duration_sec", 0.5))
        self._search_timeout_sec = float(cfg.get("search_timeout_sec", 15.0))
        self._debug_log_interval_frames = max(1, int(cfg.get("debug_log_interval_frames", 25)))
        self._sideline_ball_enter_margin = float(
            cfg.get("sideline_ball_enter_margin", 0.25)
        )
        self._sideline_ball_exit_margin = float(
            cfg.get("sideline_ball_exit_margin", 0.55)
        )
        self._sideline_exit_stable_frames = max(
            1, int(cfg.get("sideline_exit_stable_frames", 5))
        )
        self._sideline_robot_outside_margin = float(
            cfg.get("sideline_robot_outside_margin", 0.30)
        )
        self._sideline_robot_inside_margin = float(
            cfg.get("sideline_robot_inside_margin", 0.15)
        )
        self._side_recovery_setup_dist = float(
            cfg.get("side_recovery_setup_dist", 0.33)
        )
        self._side_recovery_forward_weight = float(
            cfg.get("side_recovery_forward_weight", 0.60)
        )
        self._side_recovery_max_linear = float(
            cfg.get("side_recovery_max_linear", 0.35)
        )
        self._side_recovery_push_vx = float(
            cfg.get("side_recovery_push_vx", 0.30)
        )
        self._side_recovery_safe_ball_dist = float(
            cfg.get("side_recovery_safe_ball_dist", 0.75)
        )
        self._side_recovery_retreat_step = float(
            cfg.get("side_recovery_retreat_step", 0.40)
        )
        self._side_recovery_bypass_x_offset = float(
            cfg.get("side_recovery_bypass_x_offset", 0.50)
        )
        self._side_recovery_bypass_infield_offset = float(
            cfg.get("side_recovery_bypass_infield_offset", 0.55)
        )
        self._side_recovery_cross_outside_offset = float(
            cfg.get("side_recovery_cross_outside_offset", 0.12)
        )
        self._side_recovery_stage_outside_margin = float(
            cfg.get("side_recovery_stage_outside_margin", 0.28)
        )
        self._side_recovery_cross_speed = float(
            cfg.get("side_recovery_cross_speed", 0.28)
        )
        self._side_recovery_stage_tolerance = float(
            cfg.get("side_recovery_stage_tolerance", 0.10)
        )
        self._side_recovery_face_enter = math.radians(
            float(cfg.get("side_recovery_face_enter_deg", 10.0))
        )
        self._side_recovery_face_exit = math.radians(
            float(cfg.get("side_recovery_face_exit_deg", 18.0))
        )

        # --- cached field dimensions ---
        self._field_len, self._field_width = resolve_field_size(agent.get_config())

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
        self._last_can_kick_reason = "not_checked"
        self._side_recovery_phase = "BYPASS_INFIELD"
        self._sideline_clear_frames = 0

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
            previous_state = self.state
            self.logger.info(
                f"[FSM] {self.state} -> {new_state} "
                f"(seen={self.seen_ball_count} lost={self.lost_ball_count})"
            )
            self.state = new_state
            self.state_enter_time = time.time()
            if previous_state == "ALIGN_BEHIND_BALL" or new_state == "ALIGN_BEHIND_BALL":
                self.push_ctrl.reset_orbit()
            # Positioning transitions preserve filter state. A controller that
            # needs pure rotation requests TURN_ONLY and passes through BRAKE.
            if new_state in ("STOP", "SEARCH_BALL", "RECOVER",
                             "RETURN_TO_FIELD", "KICK", "FALL_RECOVERY"):
                self.cmd_filter.reset()

    def tick(self):
        """Run one frame of the decision FSM.  Returns (vx, vy, w)."""
        self._frame_counter += 1

        get_recovery_state = getattr(self.agent, "get_recovery_state", None)
        recovery_state = (
            get_recovery_state() if callable(get_recovery_state) else "LOCOMOTION"
        )
        # FAILED and INACTIVE are terminal recovery states — the sim-side
        # get-up policy cannot (or is not configured to) recover the robot.
        # Treat them the same as LOCOMOTION so the decider resumes football
        # behaviour instead of deadlocking in FALL_RECOVERY forever.
        if recovery_state not in ("LOCOMOTION", "FAILED", "INACTIVE"):
            self.switch("FALL_RECOVERY")
            return self._emit(0.0, 0.0, 0.0, brake=True)
        if self.state == "FALL_RECOVERY":
            self.switch("SEARCH_BALL")

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

        if gc_state == "STATE_PLAYING" and self.state in ("RETURN_TO_FIELD", "STOP"):
            self.switch("SEARCH_BALL")

        # 3. Field-boundary protection has priority over normal ball handling.
        if self.state not in (
            "STOP", "RETURN_TO_FIELD", "KICK", "RECOVER"
        ):
            if self._robot_outside_controlled_band():
                self._side_recovery_phase = "RETURN_FIELD"
                self._sideline_clear_frames = 0
                self.switch("SIDE_RECOVERY")
            elif (
                self.state in ("APPROACH_BALL", "ALIGN_BEHIND_BALL", "DRIBBLE")
                and self._ball_near_sideline()
            ):
                self._begin_side_recovery()
                self._sideline_clear_frames = 0
                self.switch("SIDE_RECOVERY")

        # 4. Lost-ball protection (skip states that manage it themselves)
        if self.state not in ("SEARCH_BALL", "KICK", "RECOVER", "STOP",
                              "RETURN_TO_FIELD", "SIDE_RECOVERY"):
            if self.lost_ball_count >= self._ball_lost_frames:
                self.switch("SEARCH_BALL")

        # 5. State dispatch --------------------------------------------
        if self.state == "SEARCH_BALL":
            return self._do_search_ball()
        elif self.state == "APPROACH_BALL":
            return self._do_approach_ball()
        elif self.state == "ALIGN_BEHIND_BALL":
            return self._do_align_behind_ball()
        elif self.state == "SIDE_RECOVERY":
            return self._do_side_recovery()
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

    def get_snapshot(self):
        """Return a read-only, JSON-serializable diagnostic snapshot."""
        now = time.time()
        diagnostics = self._kick_diagnostics(now=now)
        return {
            "fsm_state": self.state,
            "align_mode": (
                self.push_ctrl.last_mode
                if self.state == "ALIGN_BEHIND_BALL"
                else ""
            ),
            "side_recovery_phase": (
                self._side_recovery_phase
                if self.state == "SIDE_RECOVERY"
                else ""
            ),
            "state_duration_s": max(0.0, now - self.state_enter_time),
            "behind_depth": diagnostics.get("behind_depth"),
            "depth_err": diagnostics.get("depth_err"),
            "lateral_err": diagnostics.get("lateral_err"),
            "ball_to_goal_yaw_err_deg": diagnostics.get(
                "ball_to_goal_yaw_err_deg"
            ),
            "distance_to_goal": diagnostics.get("distance_to_goal"),
            "can_kick": bool(diagnostics["can_kick"]),
            "can_kick_reason": diagnostics["reason"],
            "kick_push": bool(
                getattr(self.agent, "is_simulation", False)
                and self.state == "KICK"
                and self.kick_triggered
                and now - self.last_kick_time <= self._kick_action_duration_sec
            ),
        }

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
        robot_w, ball_w = self._world_pose_and_ball()
        if robot_w is None or ball_w is None:
            self.switch("SEARCH_BALL")
            return self._emit(0.0, 0.0, 0.0)

        ball_dist = float(np.linalg.norm(ball_w - robot_w))

        # Fast path: can we kick right now?
        if self._can_kick():
            self.switch("KICK")
            return self._emit(0.0, 0.0, 0.0)

        approach_target = self._approach_target(ball_w)
        target_error = approach_target - robot_w
        target_dist = float(np.linalg.norm(target_error))

        if (
            target_dist < self._approach_waypoint_tol
            or ball_dist < self._approach_to_align_dist
        ):
            self.switch("ALIGN_BEHIND_BALL")
            self.push_ctrl.last_mode = "BRAKE"
            return self._emit(0.0, 0.0, 0.0, brake=True)

        vmax = (
            self._approach_far_vmax
            if ball_dist >= self._approach_far_ball_dist
            else self._approach_near_vmax
        )
        travel_yaw = math.atan2(target_error[1], target_error[0])
        cmd = self.push_ctrl.compute_pose_command(
            approach_target,
            travel_yaw,
            vmax,
            turn_only_yaw=self.push_ctrl.turn_only_yaw,
        )

        self.logger.debug(
            f"[APPROACH] target_dist={target_dist:.2f} ball_dist={ball_dist:.2f} "
            f"cmd=({cmd.vx:.2f},{cmd.vy:.2f},{cmd.w:.2f})"
        )
        return self._emit_intent(cmd)

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

        aligned = self._ready_to_dribble()

        if aligned:
            self.switch("DRIBBLE")
            return self._emit(0.0, 0.0, 0.0)

        # Keep positioning behind the ball (not aligned → setup distance)
        intent = self.push_ctrl.compute_command(aligned=False)
        self.agent.move_head(math.inf, math.inf)
        return self._emit_intent(intent, align_brake=True)

    def _do_side_recovery(self):
        robot_w = self.agent.get_self_pos()
        if robot_w is None or robot_w[0] is None:
            return self._emit(0.0, 0.0, 0.0)
        robot_w = np.array(robot_w, dtype=float)
        field_y_max = self._field_width / 2.0

        if (
            abs(float(robot_w[1]))
            > field_y_max + self._sideline_robot_outside_margin
            and self._side_recovery_phase != "RETURN_FIELD"
        ):
            self._side_recovery_phase = "RETURN_FIELD"
            self.cmd_filter.reset()

        if self._side_recovery_phase == "RETURN_FIELD":
            side = 1.0 if robot_w[1] >= 0.0 else -1.0
            target_pos = np.array(
                [
                    robot_w[0],
                    side * (field_y_max - self._sideline_robot_inside_margin),
                ],
                dtype=float,
            )
            if abs(float(robot_w[1])) <= field_y_max:
                if self.agent.get_if_ball() and self._ball_near_sideline():
                    self._begin_side_recovery()
                    return self._emit(0.0, 0.0, 0.0, brake=True)
                elif self.agent.get_if_ball():
                    self.switch("APPROACH_BALL")
                    return self._emit(0.0, 0.0, 0.0)
                else:
                    self.switch("SEARCH_BALL")
                    return self._emit(0.0, 0.0, 0.0)
            else:
                return self._drive_to_world_target(
                    target_pos,
                    self._side_recovery_max_linear,
                    turn_only_yaw=self.push_ctrl.turn_only_yaw,
                )

        if not self.agent.get_if_ball():
            self.switch("SEARCH_BALL")
            return self._emit(0.0, 0.0, 0.0)

        ball_w = self.agent.get_ball_pos_in_map()
        if ball_w is None or ball_w[0] is None:
            self.switch("SEARCH_BALL")
            return self._emit(0.0, 0.0, 0.0)
        ball_w = np.array(ball_w, dtype=float)

        if abs(float(ball_w[1])) <= (
            field_y_max - self._sideline_ball_exit_margin
        ):
            self._sideline_clear_frames += 1
        else:
            self._sideline_clear_frames = 0

        if self._sideline_clear_frames >= self._sideline_exit_stable_frames:
            self._sideline_clear_frames = 0
            next_state = (
                "ALIGN_BEHIND_BALL"
                if self.agent.get_ball_distance() < self._approach_to_align_dist
                else "APPROACH_BALL"
            )
            self.switch(next_state)
            return self._emit(0.0, 0.0, 0.0)

        geometry = self._side_recovery_geometry(ball_w)
        recovery_dir = geometry["recovery_dir"]
        recovery_yaw = math.atan2(recovery_dir[1], recovery_dir[0])

        if self._side_recovery_phase == "RETREAT_INFIELD":
            if (
                float(np.linalg.norm(ball_w - robot_w))
                >= self._side_recovery_safe_ball_dist
            ):
                self._side_recovery_phase = "BYPASS_INFIELD"
                return self._emit(0.0, 0.0, 0.0)
            retreat_target = np.array(
                [
                    robot_w[0],
                    geometry["sideline_side"] * min(
                        max(
                            0.0,
                            abs(float(robot_w[1]))
                            - self._side_recovery_retreat_step,
                        ),
                        abs(float(geometry["infield_y"])),
                    ),
                ],
                dtype=float,
            )
            return self._drive_to_world_target(
                retreat_target,
                self._side_recovery_max_linear,
                turn_only_yaw=self.push_ctrl.turn_only_yaw,
            )

        if self._side_recovery_phase == "BYPASS_INFIELD":
            bypass_target = geometry["bypass_target"]
            if self.push_ctrl._path_crosses_ball(
                robot_w, bypass_target, ball_w
            ):
                bypass_target = np.array(
                    [robot_w[0], geometry["infield_y"]],
                    dtype=float,
                )
            if (
                float(np.linalg.norm(robot_w - geometry["bypass_target"]))
                <= self._side_recovery_stage_tolerance
            ):
                if not self.push_ctrl._path_crosses_ball(
                    robot_w, geometry["cross_target"], ball_w
                ):
                    self._side_recovery_phase = "CROSS_OUTSIDE"
                    return self._emit(0.0, 0.0, 0.0)
            return self._drive_to_world_target(
                bypass_target,
                self._side_recovery_max_linear,
                turn_only_yaw=self.push_ctrl.turn_only_yaw,
            )

        if self._side_recovery_phase == "CROSS_OUTSIDE":
            cross_target = geometry["cross_target"]
            if (
                float(np.linalg.norm(robot_w - cross_target))
                <= self._side_recovery_stage_tolerance
            ):
                if not self.push_ctrl._path_crosses_ball(
                    robot_w, geometry["staging_target"], ball_w
                ):
                    self._side_recovery_phase = "STAGE_OUTSIDE"
                    return self._emit(0.0, 0.0, 0.0)
            return self._drive_to_world_target(
                cross_target,
                self._side_recovery_cross_speed,
                turn_only_yaw=self.push_ctrl.turn_only_yaw,
            )

        if self._side_recovery_phase == "STAGE_OUTSIDE":
            staging_target = geometry["staging_target"]
            if (
                float(np.linalg.norm(robot_w - staging_target))
                <= self._side_recovery_stage_tolerance
            ):
                self._side_recovery_phase = "BRAKE_FACE_IN"
                return self._emit(0.0, 0.0, 0.0, brake=True)
            return self._drive_to_world_target(
                staging_target,
                self._side_recovery_cross_speed,
                turn_only_yaw=self.push_ctrl.turn_only_yaw,
            )

        if self._side_recovery_phase == "BRAKE_FACE_IN":
            if self.cmd_filter.is_translation_stopped(
                self._brake_translation_tol
            ):
                self._side_recovery_phase = "FACE_IN"
                return self._emit(0.0, 0.0, 0.0)
            return self._emit(0.0, 0.0, 0.0, brake=True)

        robot_yaw_rad = math.radians(float(self.agent.get_self_yaw()))
        yaw_err = self.agent.angle_normalize(recovery_yaw - robot_yaw_rad)
        cmd_w = float(np.clip(
            self.push_ctrl.kp_yaw * yaw_err,
            -self.push_ctrl.max_yaw,
            self.push_ctrl.max_yaw,
        ))
        if self._side_recovery_phase == "FACE_IN":
            if abs(yaw_err) <= self._side_recovery_face_enter:
                self._side_recovery_phase = "PUSH"
                return self._emit(0.0, 0.0, 0.0)
            return self._emit(0.0, 0.0, cmd_w, turn_only=True)

        if self._side_recovery_phase == "PUSH":
            if abs(yaw_err) > self._side_recovery_face_exit:
                self._side_recovery_phase = "BRAKE_FACE_IN"
                return self._emit(0.0, 0.0, 0.0, brake=True)
            return self._emit(self._side_recovery_push_vx, 0.0, cmd_w)

        unknown_phase = self._side_recovery_phase
        self.logger.error(
            f"[SIDE_RECOVERY] unknown phase={unknown_phase!r}; resetting safely"
        )
        self._begin_side_recovery()
        return self._emit(0.0, 0.0, 0.0, brake=True)

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

        # Use advanced dribbler without pre-scaling through agent.cmd_vel().
        cmd = self.adv_dribbler.compute_command()
        self.agent.move_head(math.inf, math.inf)

        # Fallback: if ball is close and in front but dribbler produces
        # near-zero forward speed (e.g. forward_factor=0 due to angle
        # misalignment), push straight forward to touch the ball so
        # _can_kick() can trigger on the next frame.
        # Zero out vy/w to avoid arcing (original w may be large from turn mode).
        ball_pos = self.agent.get_ball_pos()
        yaw_err = self._goal_facing_error()
        metrics = self._behind_ball_metrics()
        if (ball_pos is not None and ball_pos[0] is not None
                and float(ball_pos[0]) > 0.08 and ball_dist < 0.30
                and abs(cmd[0]) < 0.05
                and yaw_err is not None
                and abs(yaw_err) < math.radians(self._kick_facing_err_deg)
                and metrics is not None
                and bool(metrics["behind_ok"])):
            self.logger.info("[DRIBBLE] fallback push: aligned low-speed push")
            return self._emit(self._dribble_fallback_push_vx, 0.0, 0.0)

        return self._emit(cmd[0], cmd[1], cmd[2])

    def _do_kick(self):
        now = time.time()
        elapsed = now - self.state_enter_time

        # Phase 1: Settle — wait for robot to stop before kicking
        if elapsed < self._kick_settle_duration_sec:
            if not self._can_kick():
                self.logger.info(f"[KICK] cancel during settle: {self._last_can_kick_reason}")
                self.kick_triggered = False
                self.switch("ALIGN_BEHIND_BALL")
                return self._emit(0.0, 0.0, 0.0)
            return self._emit(0.0, 0.0, 0.0)

        if getattr(self.agent, "is_simulation", False):
            if not self.kick_triggered:
                if not self._can_kick():
                    self.logger.info(f"[KICK] cancel before push: {self._last_can_kick_reason}")
                    self.switch("ALIGN_BEHIND_BALL")
                    return self._emit(0.0, 0.0, 0.0)
                self.kick_triggered = True
                self.last_kick_time = now
                yaw_err = self._goal_facing_error()
                yaw_deg = math.degrees(yaw_err) if yaw_err is not None else float("nan")
                self.logger.info(f"[KICK] Sim kick push vx={self._sim_kick_push_vx:.2f} yaw_err={yaw_deg:.1f}deg")
            if now - self.last_kick_time > self._kick_action_duration_sec:
                self.kick_triggered = False
                self.switch("RECOVER")
                return self._emit(0.0, 0.0, 0.0)
            return self._emit(self._sim_kick_push_vx, 0.0, 0.0, clip_only=True)

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

    def _emit(
        self,
        vx,
        vy,
        w,
        clip_only=False,
        turn_only=False,
        brake=False,
    ):
        """Filter command and write to agent.

        .. note::

           Normal FSM commands use acceleration limiting and smoothing. The
           short simulation kick uses clipping only so it reaches the requested
           push speed without modifying the normal filter state.
        """
        if sum(bool(value) for value in (clip_only, turn_only, brake)) > 1:
            raise ValueError(
                "clip_only, turn_only, and brake are mutually exclusive"
            )
        if clip_only:
            vx, vy, w = self.cmd_filter.apply_clip_only(vx, vy, w)
        elif turn_only:
            vx, vy, w = self.cmd_filter.apply_turn_only(w)
        elif brake:
            vx, vy, w = self.cmd_filter.apply_brake()
        else:
            vx, vy, w = self.cmd_filter.apply(vx, vy, w)
        if getattr(self.agent, "is_simulation", False):
            self.agent.current_cmd = [vx, vy, w]
        else:
            self.agent.cmd_vel(vx, vy, w)
        return vx, vy, w

    def _emit_intent(self, intent, align_brake=False):
        """Emit a controller intent through its single requested filter mode."""
        if intent.mode == "TURN_ONLY":
            if not self.cmd_filter.is_translation_stopped(
                self._brake_translation_tol
            ):
                if align_brake:
                    self.push_ctrl.last_mode = "BRAKE"
                return self._emit(0.0, 0.0, 0.0, brake=True)
            return self._emit(0.0, 0.0, intent.w, turn_only=True)
        if intent.mode != "NORMAL":
            raise ValueError(f"unknown motion intent mode: {intent.mode}")
        return self._emit(intent.vx, intent.vy, intent.w)

    def _goal_point(self):
        return np.array([self._field_len / 2.0, 0.0], dtype=float)

    def _approach_target(self, ball_w):
        ball_w = np.array(ball_w, dtype=float)
        return (
            ball_w
            - self._ball_to_goal_unit(ball_w) * self._approach_setup_dist
        )

    def _robot_outside_controlled_band(self):
        robot_w = self.agent.get_self_pos()
        if robot_w is None or robot_w[0] is None:
            return False
        return abs(float(robot_w[1])) > (
            self._field_width / 2.0 + self._sideline_robot_outside_margin
        )

    def _ball_near_sideline(self):
        if not self.agent.get_if_ball():
            return False
        ball_w = self.agent.get_ball_pos_in_map()
        if ball_w is None or ball_w[0] is None:
            return False
        return abs(float(ball_w[1])) >= (
            self._field_width / 2.0 - self._sideline_ball_enter_margin
        )

    def _begin_side_recovery(self):
        self._side_recovery_phase = (
            "RETREAT_INFIELD"
            if self.agent.get_ball_distance()
            < self._side_recovery_safe_ball_dist
            else "BYPASS_INFIELD"
        )

    def _side_recovery_geometry(self, ball_w):
        ball_w = np.array(ball_w, dtype=float)
        sideline_side = 1.0 if ball_w[1] >= 0.0 else -1.0
        recovery_dir = np.array(
            [self._side_recovery_forward_weight, -sideline_side],
            dtype=float,
        )
        recovery_dir /= float(np.linalg.norm(recovery_dir))
        staging_target = ball_w - recovery_dir * self._side_recovery_setup_dist
        field_y_max = self._field_width / 2.0
        staging_target[1] = float(np.clip(
            staging_target[1],
            -field_y_max - self._side_recovery_stage_outside_margin,
            field_y_max + self._side_recovery_stage_outside_margin,
        ))
        infield_y = sideline_side * (
            field_y_max - self._side_recovery_bypass_infield_offset
        )
        bypass_target = np.array(
            [
                ball_w[0] - self._side_recovery_bypass_x_offset,
                infield_y,
            ],
            dtype=float,
        )
        cross_target = np.array(
            [
                ball_w[0] - self._side_recovery_bypass_x_offset,
                sideline_side * (
                    field_y_max + self._side_recovery_cross_outside_offset
                ),
            ],
            dtype=float,
        )
        return {
            "recovery_dir": recovery_dir,
            "sideline_side": sideline_side,
            "infield_y": infield_y,
            "bypass_target": bypass_target,
            "cross_target": cross_target,
            "staging_target": staging_target,
        }

    def _drive_to_world_target(self, target_pos, vmax, turn_only_yaw=None):
        robot_w = self.agent.get_self_pos()
        if robot_w is None or robot_w[0] is None:
            return self._emit(0.0, 0.0, 0.0)
        robot_w = np.array(robot_w, dtype=float)
        delta = np.array(target_pos, dtype=float) - robot_w
        if float(np.linalg.norm(delta)) < 1e-6:
            return self._emit(0.0, 0.0, 0.0)
        travel_yaw = math.atan2(delta[1], delta[0])
        cmd = self.push_ctrl.compute_pose_command(
            target_pos,
            travel_yaw,
            vmax,
            turn_only_yaw=turn_only_yaw,
        )
        return self._emit_intent(cmd)

    def _world_pose_and_ball(self):
        robot_w = self.agent.get_self_pos()
        ball_w = self.agent.get_ball_pos_in_map()
        if robot_w is None or ball_w is None:
            return None, None
        if robot_w[0] is None or ball_w[0] is None:
            return None, None
        return np.array(robot_w, dtype=float), np.array(ball_w, dtype=float)

    def _ball_to_goal_unit(self, ball_w):
        to_goal = self._goal_point() - ball_w
        norm = float(np.linalg.norm(to_goal))
        if norm < 1e-6:
            return np.array([1.0, 0.0], dtype=float)
        return to_goal / norm

    def _behind_ball_metrics(self):
        robot_w, ball_w = self._world_pose_and_ball()
        if robot_w is None or ball_w is None:
            return None
        unit_to_goal = self._ball_to_goal_unit(ball_w)
        lateral_unit = np.array([-unit_to_goal[1], unit_to_goal[0]], dtype=float)
        rel_robot_from_ball = robot_w - ball_w
        behind_depth = float(np.dot(ball_w - robot_w, unit_to_goal))
        lateral_err = float(np.dot(rel_robot_from_ball, lateral_unit))
        target_pos = ball_w - unit_to_goal * self._behind_ball_setup_dist
        target_err = robot_w - target_pos
        depth_err = float(np.dot(target_err, unit_to_goal))
        target_dist = float(np.linalg.norm(target_err))
        behind_ok = (
            behind_depth >= self._behind_ball_setup_dist - self._behind_ball_depth_tol
            and abs(lateral_err) <= self._behind_ball_lateral_tol
        )
        return {
            "behind_depth": behind_depth,
            "depth_err": depth_err,
            "behind_err": abs(depth_err),
            "lateral_err": lateral_err,
            "target_dist": target_dist,
            "behind_ok": behind_ok,
        }

    def _ready_to_dribble(self):
        ball_pos = self.agent.get_ball_pos()
        yaw_err = self._goal_facing_error()
        metrics = self._behind_ball_metrics()
        if ball_pos is None or ball_pos[0] is None or yaw_err is None or metrics is None:
            return False
        return (
            bool(metrics["behind_ok"])
            and abs(float(metrics["depth_err"])) <= self._behind_ball_depth_tol
            and float(metrics["behind_depth"]) > self._align_min_behind_depth
            and abs(float(metrics["lateral_err"])) < self._align_lateral_tol
            and self._align_ball_x_min < float(ball_pos[0]) < self._align_ball_x_max
            and abs(float(ball_pos[1])) < self._align_ball_y_max
            and abs(yaw_err) < math.radians(self._align_facing_err_deg)
        )

    def _goal_facing_error(self):
        """Return yaw error from robot heading to the ball -> goal direction.

        Returns:
            float or None: yaw error in radians, or None if position is unknown.
        """
        _, ball_w = self._world_pose_and_ball()
        if ball_w is None:
            return None
        unit_to_goal = self._ball_to_goal_unit(ball_w)
        target_yaw = math.atan2(unit_to_goal[1], unit_to_goal[0])
        robot_yaw_rad = math.radians(float(self.agent.get_self_yaw()))
        return self.agent.angle_normalize(target_yaw - robot_yaw_rad)

    def _kick_diagnostics(self, now=None):
        """Evaluate the kick window without mutating FSM state."""
        now = time.time() if now is None else float(now)
        result = {
            "can_kick": False,
            "reason": "not_checked",
            "behind_depth": None,
            "depth_err": None,
            "lateral_err": None,
            "ball_to_goal_yaw_err_deg": None,
            "distance_to_goal": None,
        }
        ball_pos = self.agent.get_ball_pos()
        if ball_pos is None or ball_pos[0] is None:
            result["reason"] = "no_ball"
            return result

        bx, by = float(ball_pos[0]), float(ball_pos[1])
        ball_dist = math.hypot(bx, by)
        yaw_err = self._goal_facing_error()
        if yaw_err is None:
            result["reason"] = "no_yaw"
            return result
        result["ball_to_goal_yaw_err_deg"] = math.degrees(yaw_err)

        # Collect all available geometry even when a gate such as cooldown
        # prevents kicking. This keeps trajectory diagnostics continuous.
        robot_w, ball_w = self._world_pose_and_ball()
        metrics = None
        dist_to_goal = None
        if robot_w is not None and ball_w is not None:
            dist_to_goal = float(np.linalg.norm(self._goal_point() - ball_w))
            result["distance_to_goal"] = dist_to_goal
            metrics = self._behind_ball_metrics()
            if metrics is not None:
                result.update(
                    {
                        "behind_depth": float(metrics["behind_depth"]),
                        "depth_err": float(metrics["depth_err"]),
                        "lateral_err": float(metrics["lateral_err"]),
                    }
                )

        # Preserve the existing kick-gate priority.
        if now - self.last_kick_time < self._kick_cooldown_sec:
            result["reason"] = "cooldown"
            return result
        if robot_w is None or ball_w is None:
            result["reason"] = "no_world_pose"
            return result
        if metrics is None:
            result["reason"] = "no_behind_metrics"
            return result
        if not bool(metrics["behind_ok"]):
            result["reason"] = (
                f"not_behind depth={metrics['behind_depth']:.2f} lat={metrics['lateral_err']:.2f}"
            )
            return result
        if abs(float(metrics["lateral_err"])) >= self._kick_behind_ball_lateral_tol:
            result["reason"] = f"kick_lateral={metrics['lateral_err']:.2f}"
            return result
        if not (self._kick_ball_x_min < bx < self._kick_ball_x_max):
            result["reason"] = f"ball_x={bx:.2f}"
            return result
        if abs(by) >= self._kick_ball_y_max:
            result["reason"] = f"ball_y={by:.2f}"
            return result
        if abs(yaw_err) >= math.radians(self._kick_facing_err_deg):
            result["reason"] = f"goal_yaw_err={math.degrees(yaw_err):.1f}"
            return result
        if dist_to_goal >= self._kick_max_dist_to_goal:
            result["reason"] = f"dist_goal={dist_to_goal:.2f}"
            return result
        if ball_dist >= self._kick_ball_dist_max:
            result["reason"] = f"ball_dist={ball_dist:.2f}"
            return result
        result["can_kick"] = True
        result["reason"] = "ok"
        return result

    def _can_kick(self):
        """True when the robot is in a good position to kick toward goal."""
        diagnostics = self._kick_diagnostics()
        self._last_can_kick_reason = diagnostics["reason"]
        return bool(diagnostics["can_kick"])

    def _debug_log(self):
        """Print a compact status line periodically."""
        if self._frame_counter % self._debug_log_interval_frames != 0:
            return
        ball_seen = self.agent.get_if_ball()
        ball_dist = self.agent.get_ball_distance()
        ball_angle = self.agent.get_ball_angle()
        cmd = getattr(self.agent, "current_cmd", [0, 0, 0])
        yaw_err = self._goal_facing_error()
        metrics = self._behind_ball_metrics()
        # Compute angle string safely: ball_angle may be None (no ball)
        # or 0.0 (ball directly ahead, which is falsy).
        if ball_angle is not None:
            angle_str = f"{math.degrees(ball_angle):.1f}"
        else:
            angle_str = "None"
        yaw_str = f"{math.degrees(yaw_err):.1f}" if yaw_err is not None else "None"
        behind_str = f"{metrics['behind_err']:.2f}" if metrics is not None else "None"
        lateral_str = f"{metrics['lateral_err']:.2f}" if metrics is not None else "None"
        self.logger.info(
            f"[FSM] state={self.state} "
            f"ball_seen={ball_seen} dist={ball_dist:.2f} "
            f"angle={angle_str}deg "
            f"goal_yaw_err={yaw_str}deg "
            f"behind_err={behind_str} lateral_err={lateral_str} "
            f"can_kick={self._last_can_kick_reason} "
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
