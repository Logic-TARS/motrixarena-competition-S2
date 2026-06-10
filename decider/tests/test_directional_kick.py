import importlib.util
import math
import sys
import time
import unittest
from pathlib import Path

import numpy as np


DECIDER_DIR = Path(__file__).resolve().parents[1]
if str(DECIDER_DIR) not in sys.path:
    sys.path.insert(0, str(DECIDER_DIR))

from logic.command_filter import CommandFilter
from user_entry import (
    AdvancedDribbler,
    DeciderFSM,
    PushToGoalController,
    resolve_field_size,
)


def _load_decider_module():
    spec = importlib.util.spec_from_file_location(
        "decider_main_for_test", DECIDER_DIR / "decider.py"
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class FakeLogger:
    def get_child(self, _suffix):
        return self

    def debug(self, _message):
        pass

    def info(self, _message):
        pass

    def warning(self, _message):
        pass

    def error(self, _message):
        pass


class FakeGameController:
    game_state = "STATE_PLAYING"


class FakeAgent:
    def __init__(self, active_field_size=(9.0, 6.0), color="red"):
        self.color = color
        self.is_simulation = True
        self.current_cmd = [0.0, 0.0, 0.0]
        self.robot_w = np.array([0.0, 0.0], dtype=float)
        self.ball_w = np.array([1.0, 0.0], dtype=float)
        self.robot_yaw_deg = 0.0
        self.ball_local_override = None
        self.ball_visible = True
        self.gamecontroller = FakeGameController()
        self._logger = FakeLogger()
        self._config = {
            "league": "M",
            "active_field_size": list(active_field_size),
            "field_size": {
                "S": [9.0, 6.0],
                "M": [14.0, 9.0],
                "L": [22.0, 14.0],
            },
            "decider_fsm": {
                "ball_seen_confirm_frames": 1,
                "ball_lost_confirm_frames": 1,
                "approach_to_align_dist": 0.85,
                "approach_behind_setup_dist": 0.75,
                "approach_waypoint_tolerance": 0.18,
                "approach_far_ball_dist": 1.50,
                "approach_far_vmax": 0.65,
                "approach_near_vmax": 0.35,
                "align_to_dribble_ball_x_min": 0.15,
                "align_to_dribble_ball_x_max": 0.35,
                "align_to_dribble_ball_y_max": 0.08,
                "align_to_dribble_facing_error_deg": 12,
                "align_to_dribble_lateral_tol": 0.05,
                "align_to_dribble_min_behind_depth": 0.22,
                "align_turn_only_yaw_deg": 35,
                "align_direct_vx_max": 0.45,
                "navigation_min_vx": 0.24,
                "align_escape_enter_dist": 0.70,
                "align_escape_exit_dist": 0.78,
                "align_escape_vx": 0.30,
                "align_escape_face_enter_deg": 10,
                "align_escape_face_exit_deg": 18,
                "brake_translation_tolerance": 0.02,
                "orbit_path_clearance": 0.28,
                "orbit_side_lateral_offset": 0.50,
                "orbit_behind_lateral_offset": 0.35,
                "orbit_waypoint_tolerance": 0.12,
                "orbit_side_deadband": 0.05,
                "dribble_to_approach_dist": 0.9,
                "sideline_ball_enter_margin": 0.25,
                "sideline_ball_exit_margin": 0.55,
                "sideline_exit_stable_frames": 5,
                "sideline_robot_outside_margin": 0.30,
                "sideline_robot_inside_margin": 0.15,
                "side_recovery_setup_dist": 0.33,
                "side_recovery_forward_weight": 0.60,
                "side_recovery_max_linear": 0.35,
                "side_recovery_push_vx": 0.30,
                "side_recovery_safe_ball_dist": 0.75,
                "side_recovery_retreat_step": 0.40,
                "side_recovery_bypass_x_offset": 0.50,
                "side_recovery_bypass_infield_offset": 0.55,
                "side_recovery_cross_outside_offset": 0.12,
                "side_recovery_stage_outside_margin": 0.28,
                "side_recovery_cross_speed": 0.28,
                "side_recovery_stage_tolerance": 0.10,
                "side_recovery_face_enter_deg": 10,
                "side_recovery_face_exit_deg": 18,
                "kick_ball_x_min": 0.12,
                "kick_ball_x_max": 0.30,
                "kick_ball_y_max": 0.06,
                "kick_facing_error_deg": 5,
                "kick_max_dist_to_goal": 10.0,
                "kick_ball_dist_max": 0.31,
                "behind_ball_setup_dist": 0.35,
                "behind_ball_lateral_tol": 0.18,
                "behind_ball_depth_tol": 0.08,
                "kick_behind_ball_lateral_tol": 0.08,
                "sim_kick_push_vx": 0.35,
                "dribble_fallback_push_vx": 0.30,
                "kick_cooldown_sec": 0.0,
                "kick_action_duration_sec": 0.35,
                "kick_settle_duration_sec": 0.3,
                "recover_duration_sec": 0.5,
                "search_timeout_sec": 15.0,
            },
            "cmd_filter": {
                "vx_max": 0.90,
                "vx_min": 0.0,
                "vy_max": 0.35,
                "w_max": 1.5,
                "vx_accel": 0.04,
                "vy_accel": 0.02,
                "w_accel": 0.08,
                "smooth_alpha": 0.3,
            },
        }
        self.adv_dribbler = AdvancedDribbler(self)
        self.push_to_goal = PushToGoalController(self)

    def get_config(self):
        return self._config

    def get_logger(self):
        return self._logger

    def get_if_ball(self):
        return self.ball_visible

    def get_self_pos(self):
        return self.robot_w.copy()

    def get_self_yaw(self):
        return self.robot_yaw_deg

    def get_ball_pos_in_map(self):
        return self.ball_w.copy()

    def get_ball_pos(self):
        if self.ball_local_override is not None:
            return np.array(self.ball_local_override, dtype=float)
        delta = self.ball_w - self.robot_w
        yaw = math.radians(self.robot_yaw_deg)
        c = math.cos(yaw)
        s = math.sin(yaw)
        return np.array(
            [delta[0] * c + delta[1] * s, -delta[0] * s + delta[1] * c],
            dtype=float,
        )

    def get_ball_distance(self):
        ball = self.get_ball_pos()
        return float(np.linalg.norm(ball))

    def get_ball_angle(self):
        ball = self.get_ball_pos()
        return math.atan2(float(ball[1]), float(ball[0]))

    def angle_normalize(self, angle):
        return (angle + math.pi) % (2.0 * math.pi) - math.pi

    def cmd_vel(self, vx, vy, w):
        self.current_cmd = [float(vx), float(vy), float(w)]

    def move_head(self, _pitch, _yaw):
        pass

    def kick(self, foot=0, death=0):
        self.last_kick = (foot, death)


class FieldSizeTests(unittest.TestCase):
    def test_active_field_size_has_priority(self):
        config = {
            "active_field_size": [9.0, 6.0],
            "league": "M",
            "field_size": {"M": [14.0, 9.0]},
        }
        self.assertEqual(resolve_field_size(config), (9.0, 6.0))

    def test_field_size_falls_back_to_league(self):
        config = {"league": "M", "field_size": {"M": [14.0, 9.0]}}
        self.assertEqual(resolve_field_size(config), (14.0, 9.0))

    def test_match_config_exposes_real_field_size(self):
        module = _load_decider_module()
        path, match_config = module._load_sim_match_config()
        self.assertEqual(path.name, "match_config.json")
        self.assertEqual(
            module._active_field_size_from_match_config(match_config),
            [9.0, 6.0],
        )

    def test_goal_point_is_positive_x_for_both_teams(self):
        for color in ("red", "blue"):
            with self.subTest(color=color):
                fsm = DeciderFSM(FakeAgent(color=color))
                np.testing.assert_allclose(fsm._goal_point(), [4.5, 0.0])


class DirectionGeometryTests(unittest.TestCase):
    def test_dribbler_boundary_repulsion_uses_ball_position(self):
        agent = FakeAgent()
        agent.robot_w = np.array([0.0, 0.0])
        agent.ball_w = np.array([1.0, 2.8])

        target_vector, is_safe_zone = agent.adv_dribbler.get_target_vector()
        goal_only = np.array([4.5, 0.0]) - agent.ball_w
        goal_only /= np.linalg.norm(goal_only)

        self.assertEqual(is_safe_zone, 1)
        self.assertLess(target_vector[1], goal_only[1])

    def test_dribbler_turn_sign_matches_target_side(self):
        cases = [
            ("left", np.array([1.0, -1.0]), 1),
            ("right", np.array([1.0, 1.0]), -1),
            ("straight", np.array([1.0, 0.0]), 0),
        ]
        for name, ball_w, expected_sign in cases:
            with self.subTest(name=name):
                agent = FakeAgent()
                agent.ball_w = ball_w
                agent.ball_local_override = [0.20, 0.0]
                cmd_w = agent.adv_dribbler.compute_command()[2]
                if expected_sign == 0:
                    self.assertAlmostEqual(cmd_w, 0.0, places=7)
                else:
                    self.assertEqual(math.copysign(1.0, cmd_w), expected_sign)

    def test_turn_damping_reduces_magnitude_without_flipping_sign(self):
        agent = FakeAgent()
        agent.ball_w = np.array([1.0, -1.0])
        agent.ball_local_override = [0.20, 0.0]
        close_w = agent.adv_dribbler.compute_command()[2]
        agent.ball_local_override = [0.80, 0.0]
        far_w = agent.adv_dribbler.compute_command()[2]
        self.assertGreater(close_w, 0.0)
        self.assertGreater(far_w, 0.0)
        self.assertLess(abs(close_w), abs(far_w))

    def test_behind_ball_metrics_keep_signed_lateral_error(self):
        agent = FakeAgent()
        fsm = DeciderFSM(agent)
        agent.ball_w = np.array([3.0, 2.5])
        unit_to_goal = fsm._ball_to_goal_unit(agent.ball_w)
        lateral_unit = np.array([-unit_to_goal[1], unit_to_goal[0]])

        agent.robot_w = agent.ball_w - unit_to_goal * 0.35 + lateral_unit * 0.10
        positive = fsm._behind_ball_metrics()
        self.assertAlmostEqual(positive["behind_depth"], 0.35, places=6)
        self.assertAlmostEqual(positive["lateral_err"], 0.10, places=6)

        agent.robot_w = agent.ball_w - unit_to_goal * 0.35 - lateral_unit * 0.10
        negative = fsm._behind_ball_metrics()
        self.assertAlmostEqual(negative["behind_depth"], 0.35, places=6)
        self.assertAlmostEqual(negative["lateral_err"], -0.10, places=6)

    def test_facing_error_uses_ball_to_goal_not_robot_to_goal(self):
        agent = FakeAgent()
        agent.robot_w = np.array([0.0, 3.0])
        agent.ball_w = np.array([3.0, 2.0])
        agent.robot_yaw_deg = 0.0
        fsm = DeciderFSM(agent)

        expected = math.atan2(-2.0, 1.5)
        robot_to_goal = math.atan2(-3.0, 4.5)
        actual = fsm._goal_facing_error()
        self.assertAlmostEqual(actual, expected, places=7)
        self.assertNotAlmostEqual(actual, robot_to_goal, places=3)


class AlignControllerTests(unittest.TestCase):
    def test_waypoint_navigation_turns_then_moves_forward(self):
        agent = FakeAgent()
        controller = agent.push_to_goal
        agent.robot_w = np.array([0.0, 0.5])
        agent.ball_w = np.array([1.0, 0.0])
        target = np.array([0.65, 0.0])
        travel_yaw = math.degrees(math.atan2(
            target[1] - agent.robot_w[1],
            target[0] - agent.robot_w[0],
        ))

        agent.robot_yaw_deg = travel_yaw + 90.0
        turning = controller.compute_command(aligned=False)
        self.assertEqual(turning.mode, "TURN_ONLY")
        self.assertEqual(controller.last_mode, "TURN")
        self.assertEqual(turning.vx, 0.0)

        agent.robot_yaw_deg = travel_yaw
        moving = controller.compute_command(aligned=False)
        self.assertEqual(moving.mode, "NORMAL")
        self.assertEqual(controller.last_mode, "DIRECT_POSITION")
        self.assertGreaterEqual(moving.vx, controller.navigation_min_vx)
        self.assertEqual(moving.vy, 0.0)

    def test_path_through_ball_uses_locked_orbit_waypoints(self):
        agent = FakeAgent()
        controller = agent.push_to_goal
        agent.ball_w = np.array([1.0, 0.0])
        agent.robot_w = np.array([1.4, 0.0])

        controller.compute_command(aligned=False)
        first_side = controller.orbit_side
        first_target = controller.last_target.copy()

        self.assertEqual(controller.orbit_phase, "ORBIT_SIDE")
        self.assertIsNotNone(first_side)
        self.assertGreater(first_target[1], 0.0)
        distance, _ = controller._point_to_segment_distance(
            agent.ball_w, agent.robot_w, first_target
        )
        self.assertGreaterEqual(distance, controller.orbit_clearance)

        agent.robot_yaw_deg = math.degrees(math.atan2(
            first_target[1] - agent.robot_w[1],
            first_target[0] - agent.robot_w[0],
        ))
        moving = controller.compute_command(aligned=False)
        self.assertEqual(controller.last_mode, "ORBIT_SIDE")
        self.assertGreaterEqual(moving.vx, controller.navigation_min_vx)

        agent.robot_w = np.array([1.4, -0.20])
        controller.compute_command(aligned=False)
        self.assertEqual(controller.orbit_side, first_side)
        self.assertGreater(controller.last_target[1], 0.0)

        side_target = controller._orbit_target(
            "ORBIT_SIDE",
            agent.ball_w,
            np.array([0.65, 0.0]),
            np.array([0.0, 1.0]),
        )
        agent.robot_w = side_target
        controller.compute_command(aligned=False)
        self.assertEqual(controller.orbit_phase, "ORBIT_BEHIND")

        behind_side_target = controller.last_target.copy()
        distance, _ = controller._point_to_segment_distance(
            agent.ball_w, side_target, behind_side_target
        )
        self.assertGreaterEqual(distance, controller.orbit_clearance)

        agent.robot_w = behind_side_target
        controller.compute_command(aligned=False)
        self.assertIsNone(controller.orbit_phase)

        agent.robot_w = np.array([0.65, 0.0])
        controller.compute_command(aligned=False)
        self.assertEqual(controller.last_mode, "DIRECT_FACE")

    def test_safe_path_goes_directly_to_behind_target(self):
        agent = FakeAgent()
        agent.ball_w = np.array([1.0, 0.0])
        agent.robot_w = np.array([0.0, 0.5])

        agent.push_to_goal.compute_command(aligned=False)

        self.assertIsNone(agent.push_to_goal.orbit_side)
        np.testing.assert_allclose(
            agent.push_to_goal.last_target,
            [0.65, 0.0],
            atol=1e-7,
        )

    def test_align_never_emits_negative_forward_velocity(self):
        agent = FakeAgent()
        fsm = DeciderFSM(agent)
        fsm.state = "ALIGN_BEHIND_BALL"
        agent.ball_w = np.array([1.0, 0.0])
        agent.robot_w = np.array([1.4, 0.4])

        intent = agent.push_to_goal.compute_command(aligned=False)
        filtered = fsm._emit_intent(intent)

        self.assertGreaterEqual(intent.vx, 0.0)
        self.assertGreaterEqual(filtered[0], 0.0)

    def test_near_ball_large_yaw_escapes_with_positive_forward_motion(self):
        agent = FakeAgent()
        controller = agent.push_to_goal
        agent.robot_w = np.array([0.4, 0.0])
        agent.ball_w = np.array([1.0, 0.0])
        agent.robot_yaw_deg = -90.0

        face = controller.compute_command(aligned=False)
        self.assertEqual(controller.last_mode, "ESCAPE_FACE")
        self.assertEqual(face.mode, "TURN_ONLY")
        self.assertEqual(face.vx, 0.0)

        agent.robot_yaw_deg = math.degrees(controller.escape_yaw)
        forward = controller.compute_command(aligned=False)
        self.assertEqual(controller.last_mode, "ESCAPE_FORWARD")
        self.assertEqual(forward.mode, "NORMAL")
        self.assertAlmostEqual(forward.vx, 0.30)

        agent.robot_w = np.array([0.20, 0.0])
        controller.compute_command(aligned=False)
        self.assertIsNone(controller.escape_phase)

    def test_turn_request_brakes_before_strict_turn_only(self):
        agent = FakeAgent()
        fsm = DeciderFSM(agent)
        fsm.state = "ALIGN_BEHIND_BALL"
        agent.robot_w = np.array([0.4, 0.6])
        agent.ball_w = np.array([1.0, 0.0])
        agent.robot_yaw_deg = 90.0
        fsm.cmd_filter.last_cmd = (0.8, -0.2, 0.4)

        first = fsm._do_align_behind_ball()
        self.assertEqual(agent.push_to_goal.last_mode, "BRAKE")
        self.assertGreater(first[0], 0.0)
        self.assertLess(first[0], 0.8)

        for _ in range(100):
            cmd = fsm._do_align_behind_ball()
            if agent.push_to_goal.last_mode == "TURN":
                break
        self.assertEqual(agent.push_to_goal.last_mode, "TURN")
        self.assertEqual(cmd[0], 0.0)
        self.assertEqual(cmd[1], 0.0)
        self.assertEqual(fsm.cmd_filter.last_cmd[:2], (0.0, 0.0))


class ApproachControllerTests(unittest.TestCase):
    def test_approach_target_is_behind_ball(self):
        agent = FakeAgent()
        fsm = DeciderFSM(agent)
        agent.ball_w = np.array([1.0, 0.5])

        target = fsm._approach_target(agent.ball_w)
        unit_to_goal = fsm._ball_to_goal_unit(agent.ball_w)

        np.testing.assert_allclose(
            target,
            agent.ball_w - unit_to_goal * 0.75,
        )
        self.assertAlmostEqual(
            float(np.linalg.norm(agent.ball_w - target)),
            0.75,
        )

    def test_approach_to_align_brakes_without_resetting_filter_state(self):
        agent = FakeAgent()
        fsm = DeciderFSM(agent)
        fsm.state = "APPROACH_BALL"
        agent.robot_w = np.array([0.2, 0.0])
        agent.ball_w = np.array([1.0, 0.0])
        fsm.cmd_filter.last_cmd = (0.8, 0.2, -0.4)

        cmd = fsm._do_approach_ball()

        self.assertEqual(fsm.state, "ALIGN_BEHIND_BALL")
        self.assertEqual(agent.push_to_goal.last_mode, "BRAKE")
        self.assertGreater(cmd[0], 0.0)
        self.assertLess(cmd[0], 0.8)
        self.assertLess(abs(cmd[1]), 0.2)
        self.assertLess(abs(cmd[2]), 0.4)


class SideRecoveryTests(unittest.TestCase):
    def test_upper_and_lower_sidelines_use_mirrored_recovery_geometry(self):
        agent = FakeAgent()
        fsm = DeciderFSM(agent)

        upper = fsm._side_recovery_geometry([1.0, 2.9])
        lower = fsm._side_recovery_geometry([1.0, -2.9])
        upper_dir = upper["recovery_dir"]
        lower_dir = lower["recovery_dir"]
        upper_stage = upper["staging_target"]
        lower_stage = lower["staging_target"]

        self.assertGreater(upper_dir[0], 0.0)
        self.assertLess(upper_dir[1], 0.0)
        self.assertGreater(lower_dir[0], 0.0)
        self.assertGreater(lower_dir[1], 0.0)
        self.assertAlmostEqual(upper_dir[0], lower_dir[0])
        self.assertAlmostEqual(upper_dir[1], -lower_dir[1])
        self.assertAlmostEqual(upper_stage[0], lower_stage[0])
        self.assertAlmostEqual(upper_stage[1], -lower_stage[1])
        self.assertLessEqual(abs(upper_stage[1]), 3.28)
        np.testing.assert_allclose(
            upper["bypass_target"],
            [0.5, 2.45],
        )
        np.testing.assert_allclose(
            upper["cross_target"],
            [0.5, 3.12],
        )

    def test_sideline_ball_enters_recovery_state(self):
        agent = FakeAgent()
        fsm = DeciderFSM(agent)
        fsm.state = "APPROACH_BALL"
        agent.robot_w = np.array([0.5, 2.4])
        agent.ball_w = np.array([1.0, 2.80])

        fsm.tick()

        self.assertEqual(fsm.state, "SIDE_RECOVERY")
        self.assertEqual(fsm._side_recovery_phase, "RETREAT_INFIELD")

    def test_robot_outside_margin_returns_before_chasing_ball(self):
        agent = FakeAgent()
        fsm = DeciderFSM(agent)
        fsm.state = "APPROACH_BALL"
        agent.robot_w = np.array([0.0, 3.31])
        agent.ball_w = np.array([2.0, 0.0])

        cmd = fsm.tick()

        self.assertEqual(fsm.state, "SIDE_RECOVERY")
        self.assertEqual(fsm._side_recovery_phase, "RETURN_FIELD")
        self.assertEqual(cmd[0], 0.0)
        self.assertEqual(cmd[1], 0.0)
        self.assertLess(cmd[2], 0.0)

    def test_robot_outside_margin_interrupts_search(self):
        agent = FakeAgent()
        fsm = DeciderFSM(agent)
        fsm.state = "SEARCH_BALL"
        agent.robot_w = np.array([0.0, -3.31])
        agent.ball_visible = False

        fsm.tick()

        self.assertEqual(fsm.state, "SIDE_RECOVERY")
        self.assertEqual(fsm._side_recovery_phase, "RETURN_FIELD")

    def test_push_pauses_when_recovery_heading_drifts(self):
        agent = FakeAgent()
        fsm = DeciderFSM(agent)
        fsm.state = "SIDE_RECOVERY"
        fsm._side_recovery_phase = "PUSH"
        agent.robot_w = np.array([0.8, 3.1])
        agent.ball_w = np.array([1.0, 2.9])
        agent.robot_yaw_deg = 90.0

        cmd = fsm._do_side_recovery()

        self.assertEqual(fsm._side_recovery_phase, "BRAKE_FACE_IN")
        self.assertGreaterEqual(cmd[0], 0.0)

    def test_recovery_exits_after_five_clear_frames(self):
        agent = FakeAgent()
        fsm = DeciderFSM(agent)
        fsm.state = "SIDE_RECOVERY"
        fsm._side_recovery_phase = "PUSH"
        agent.robot_w = np.array([0.7, 2.7])
        agent.ball_w = np.array([1.0, 2.40])
        agent.ball_local_override = [0.30, 0.0]

        for _ in range(4):
            fsm._do_side_recovery()
            self.assertEqual(fsm.state, "SIDE_RECOVERY")
        fsm._do_side_recovery()

        self.assertEqual(fsm.state, "ALIGN_BEHIND_BALL")

    def test_lost_ball_leaves_recovery_for_search(self):
        agent = FakeAgent()
        fsm = DeciderFSM(agent)
        fsm.state = "SIDE_RECOVERY"
        agent.ball_visible = False

        fsm._do_side_recovery()

        self.assertEqual(fsm.state, "SEARCH_BALL")

    def test_recovery_advances_through_safe_waypoints(self):
        agent = FakeAgent()
        fsm = DeciderFSM(agent)
        fsm.state = "SIDE_RECOVERY"
        agent.ball_w = np.array([1.0, -2.9])
        geometry = fsm._side_recovery_geometry(agent.ball_w)

        fsm._side_recovery_phase = "BYPASS_INFIELD"
        agent.robot_w = geometry["bypass_target"].copy()
        fsm._do_side_recovery()
        self.assertEqual(fsm._side_recovery_phase, "CROSS_OUTSIDE")

        agent.robot_w = geometry["cross_target"].copy()
        fsm._do_side_recovery()
        self.assertEqual(fsm._side_recovery_phase, "STAGE_OUTSIDE")

        agent.robot_w = geometry["staging_target"].copy()
        fsm._do_side_recovery()
        self.assertEqual(fsm._side_recovery_phase, "BRAKE_FACE_IN")

        fsm._do_side_recovery()
        self.assertEqual(fsm._side_recovery_phase, "FACE_IN")

        recovery_yaw = math.degrees(math.atan2(
            geometry["recovery_dir"][1],
            geometry["recovery_dir"][0],
        ))
        agent.robot_yaw_deg = recovery_yaw
        fsm._do_side_recovery()
        self.assertEqual(fsm._side_recovery_phase, "PUSH")

        cmd = fsm._do_side_recovery()
        self.assertGreater(cmd[0], 0.0)
        self.assertGreater(fsm._side_recovery_push_vx, 0.20)

    def test_return_field_transition_stops_before_new_phase(self):
        agent = FakeAgent()
        fsm = DeciderFSM(agent)
        fsm.state = "SIDE_RECOVERY"
        fsm._side_recovery_phase = "RETURN_FIELD"
        agent.robot_w = np.array([0.0, 2.9])
        agent.ball_w = np.array([1.0, 2.9])
        fsm.cmd_filter.last_cmd = (0.5, 0.0, 0.0)

        cmd = fsm._do_side_recovery()

        self.assertEqual(fsm._side_recovery_phase, "BYPASS_INFIELD")
        self.assertGreater(cmd[0], 0.0)
        self.assertLess(cmd[0], 0.5)

    def test_unknown_phase_resets_without_pushing(self):
        agent = FakeAgent()
        fsm = DeciderFSM(agent)
        fsm.state = "SIDE_RECOVERY"
        fsm._side_recovery_phase = "UNKNOWN"
        agent.robot_w = np.array([0.0, 2.4])
        agent.ball_w = np.array([1.0, 2.9])

        cmd = fsm._do_side_recovery()

        self.assertIn(
            fsm._side_recovery_phase,
            ("RETREAT_INFIELD", "BYPASS_INFIELD"),
        )
        self.assertEqual(cmd, (0.0, 0.0, 0.0))

    def test_face_push_uses_heading_hysteresis(self):
        agent = FakeAgent()
        fsm = DeciderFSM(agent)
        fsm.state = "SIDE_RECOVERY"
        agent.robot_w = np.array([0.8, 3.1])
        agent.ball_w = np.array([1.0, 2.9])
        geometry = fsm._side_recovery_geometry(agent.ball_w)
        recovery_yaw = math.atan2(
            geometry["recovery_dir"][1],
            geometry["recovery_dir"][0],
        )

        agent.robot_yaw_deg = math.degrees(recovery_yaw + math.radians(15))
        fsm._side_recovery_phase = "FACE_IN"
        fsm._do_side_recovery()
        self.assertEqual(fsm._side_recovery_phase, "FACE_IN")

        fsm._side_recovery_phase = "PUSH"
        fsm._do_side_recovery()
        self.assertEqual(fsm._side_recovery_phase, "PUSH")

    def test_recovery_waypoint_segments_clear_the_ball(self):
        agent = FakeAgent()
        fsm = DeciderFSM(agent)
        ball = np.array([1.0, 2.9])
        geometry = fsm._side_recovery_geometry(ball)
        segments = [
            (geometry["bypass_target"], geometry["cross_target"]),
            (geometry["cross_target"], geometry["staging_target"]),
        ]
        for start, end in segments:
            with self.subTest(start=start, end=end):
                distance, _ = agent.push_to_goal._point_to_segment_distance(
                    ball, start, end
                )
                self.assertGreaterEqual(
                    distance, agent.push_to_goal.orbit_clearance
                )


class DribbleTransitionTests(unittest.TestCase):
    def setUp(self):
        self.agent = FakeAgent()
        self.fsm = DeciderFSM(self.agent)
        self.agent.ball_w = np.array([3.0, 0.0])
        self.agent.robot_w = np.array([2.72, 0.0])
        self.agent.ball_local_override = [0.28, 0.0]

    def test_nineteen_degree_yaw_no_longer_enters_dribble(self):
        self.agent.robot_yaw_deg = 19.3
        self.assertFalse(self.fsm._ready_to_dribble())

    def test_strict_geometry_enters_dribble(self):
        self.agent.robot_yaw_deg = 0.0
        self.assertTrue(self.fsm._ready_to_dribble())

        self.agent.robot_w = np.array([2.72, 0.051])
        self.assertFalse(self.fsm._ready_to_dribble())

    def test_dribble_exits_when_ball_is_farther_than_point_nine(self):
        self.fsm.state = "DRIBBLE"
        self.agent.ball_local_override = [0.91, 0.0]

        self.fsm._do_dribble()

        self.assertEqual(self.fsm.state, "APPROACH_BALL")


class KickStateTests(unittest.TestCase):
    def setUp(self):
        self.agent = FakeAgent()
        self.fsm = DeciderFSM(self.agent)
        self.place_kick_pose()

    def place_kick_pose(self, depth=0.28, lateral=0.0, yaw_deg=0.0):
        self.agent.ball_w = np.array([3.0, 0.0])
        self.agent.robot_w = np.array([3.0 - depth, lateral])
        self.agent.robot_yaw_deg = yaw_deg
        self.agent.ball_local_override = [depth, 0.0]

    def test_kick_rejects_each_strict_alignment_violation(self):
        self.assertTrue(self.fsm._can_kick())

        cases = [
            ("ball_x", {"ball_local_override": [0.31, 0.0]}),
            ("ball_y", {"ball_local_override": [0.28, 0.07]}),
            ("yaw", {"robot_yaw_deg": 6.0}),
            ("depth", {"robot_w": np.array([2.80, 0.0])}),
            ("lateral", {"robot_w": np.array([2.72, 0.09])}),
        ]
        for name, changes in cases:
            with self.subTest(name=name):
                self.place_kick_pose()
                for key, value in changes.items():
                    setattr(self.agent, key, value)
                self.assertFalse(self.fsm._can_kick())

    def test_settle_failure_returns_to_align_and_can_retry(self):
        self.fsm.switch("KICK")
        self.agent.ball_local_override = [0.28, 0.07]
        self.fsm._do_kick()
        self.assertEqual(self.fsm.state, "ALIGN_BEHIND_BALL")

        self.place_kick_pose()
        self.fsm._do_align_behind_ball()
        self.assertEqual(self.fsm.state, "KICK")

    def test_clip_only_kick_does_not_pollute_normal_filter_state(self):
        self.fsm.switch("KICK")
        self.fsm.state_enter_time = time.time() - self.fsm._kick_settle_duration_sec - 0.01
        kick_cmd = self.fsm._do_kick()
        self.assertEqual(kick_cmd, (0.35, 0.0, 0.0))
        self.assertEqual(self.fsm.cmd_filter.last_cmd, (0.0, 0.0, 0.0))

        self.fsm.kick_triggered = True
        self.fsm.last_kick_time = time.time() - self.fsm._kick_action_duration_sec - 0.01
        self.fsm._do_kick()
        self.assertEqual(self.fsm.state, "RECOVER")
        self.assertEqual(self.fsm.cmd_filter.last_cmd, (0.0, 0.0, 0.0))

        self.fsm.switch("DRIBBLE")
        normal_cmd = self.fsm._emit(0.20, 0.0, 0.0)
        self.assertGreater(normal_cmd[0], 0.0)
        self.assertLess(normal_cmd[0], 0.20)


class CommandFilterTests(unittest.TestCase):
    def test_clip_only_limits_without_updating_state(self):
        command_filter = CommandFilter(
            {
                "cmd_filter": {
                    "vx_max": 0.9,
                    "vx_min": 0.0,
                    "vy_max": 0.35,
                    "w_max": 1.5,
                }
            }
        )
        command_filter.last_cmd = (0.2, -0.1, 0.3)
        result = command_filter.apply_clip_only(2.0, -2.0, 2.0)
        self.assertEqual(result, (0.9, -0.35, 1.5))
        self.assertEqual(command_filter.last_cmd, (0.2, -0.1, 0.3))

    def test_turn_only_clears_translation_and_filters_rotation(self):
        command_filter = CommandFilter(
            {
                "cmd_filter": {
                    "vx_max": 0.9,
                    "vx_min": 0.0,
                    "vy_max": 0.35,
                    "w_max": 1.5,
                    "w_accel": 0.08,
                    "smooth_alpha": 0.3,
                }
            }
        )
        command_filter.last_cmd = (0.8, -0.3, 0.4)

        result = command_filter.apply_turn_only(1.5)

        self.assertEqual(result[0], 0.0)
        self.assertEqual(result[1], 0.0)
        self.assertGreater(result[2], 0.4)
        self.assertLess(result[2], 1.5)
        self.assertEqual(command_filter.last_cmd, result)

    def test_brake_smoothly_reduces_translation(self):
        command_filter = CommandFilter(
            {
                "cmd_filter": {
                    "vx_max": 0.9,
                    "vx_min": 0.0,
                    "vy_max": 0.35,
                    "w_max": 1.5,
                    "vx_accel": 0.04,
                    "vy_accel": 0.02,
                    "w_accel": 0.08,
                    "smooth_alpha": 0.3,
                }
            }
        )
        command_filter.last_cmd = (0.8, -0.3, 0.4)
        first = command_filter.apply_brake()
        second = command_filter.apply_brake()

        self.assertGreater(first[0], second[0])
        self.assertGreater(second[0], 0.0)
        self.assertGreater(abs(first[1]), abs(second[1]))
        self.assertFalse(command_filter.is_translation_stopped())

    def test_negative_forward_commands_are_clipped_to_zero(self):
        command_filter = CommandFilter(
            {"cmd_filter": {"vx_min": -0.25, "vx_max": 0.9}}
        )
        result = command_filter.apply_clip_only(-0.5, 0.0, 0.0)
        self.assertEqual(result[0], 0.0)


if __name__ == "__main__":
    unittest.main()
