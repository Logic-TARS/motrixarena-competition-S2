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

from logic.command_filter import CommandFilter, forward_speed_limit
from user_entry import (
    AdvancedDribbler,
    ContinuousPushToGoal,
    DeciderFSM,
    DirectChaseConfig,
    DirectChaseController,
    PushToGoalConfig,
    PushToGoalController,
    game,
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
            "push_to_goal": {
                "behind_offset": 0.12,
                "far_distance": 0.60,
                "push_vx_min": 0.35,
                "vx_gain": 2.5,
                "vx_max": 1.0,
                "vy_gain": 2.5,
                "vy_max": 0.20,
                "w_gain": 2.5,
                "w_max": 0.45,
                "turn_only_yaw_deg": 60.0,
                "search_w": 0.6,
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
        self.assertEqual(turning.mode, "NORMAL")
        self.assertGreaterEqual(turning.vx, controller.navigation_min_vx)

        fsm = DeciderFSM(agent)
        filtered = fsm.cmd_filter.apply_clip_only(
            turning.vx, turning.vy, turning.w
        )
        self.assertEqual(filtered[0], 0.0)

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
        # Continuous blending: vx is positive (never pure rotation)
        self.assertEqual(face.mode, "NORMAL")
        self.assertGreater(face.vx, 0.0)
        self.assertNotEqual(face.w, 0.0)
        self.assertLess(face.vx, 0.10)  # heavily scaled during escape face

        agent.robot_yaw_deg = math.degrees(controller.escape_yaw)
        forward = controller.compute_command(aligned=False)
        self.assertEqual(controller.last_mode, "ESCAPE_FORWARD")
        self.assertEqual(forward.mode, "NORMAL")
        self.assertAlmostEqual(forward.vx, 0.30)

        agent.robot_w = np.array([0.20, 0.0])
        controller.compute_command(aligned=False)
        self.assertIsNone(controller.escape_phase)

    def test_turn_request_uses_filter_envelope_without_negative_velocity(self):
        agent = FakeAgent()
        fsm = DeciderFSM(agent)
        fsm.state = "ALIGN_BEHIND_BALL"
        agent.robot_w = np.array([0.4, 0.6])
        agent.ball_w = np.array([1.0, 0.0])
        agent.robot_yaw_deg = 90.0
        fsm.cmd_filter.last_cmd = (0.8, -0.2, 0.4)

        all_vx_nonnegative = True
        any_significant_turn = False
        for _ in range(200):
            vx, vy, w = fsm._do_align_behind_ball()
            if vx < 0.0:
                all_vx_nonnegative = False
            if abs(w) > 0.5:
                any_significant_turn = True
        self.assertTrue(all_vx_nonnegative)
        self.assertTrue(any_significant_turn)


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
        self.assertEqual(result, (0.0, -0.35, 1.5))
        self.assertEqual(command_filter.last_cmd, (0.2, -0.1, 0.3))

    def test_forward_limit_decreases_continuously_with_yaw(self):
        command_filter = CommandFilter(
            {
                "cmd_filter": {
                    "vx_max": 0.8,
                    "w_max": 1.5,
                    "yaw_full_speed": 0.25,
                    "yaw_zero_speed": 1.5,
                }
            }
        )
        low_turn = command_filter.apply_clip_only(0.8, 0.0, 0.25)
        medium_turn = command_filter.apply_clip_only(0.8, 0.0, 0.875)
        hard_turn = command_filter.apply_clip_only(0.8, 0.0, 1.5)
        self.assertEqual(low_turn[0], 0.8)
        self.assertAlmostEqual(medium_turn[0], 0.4)
        self.assertEqual(hard_turn[0], 0.0)

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


class DirectChaseTests(unittest.TestCase):
    def setUp(self):
        self.agent = FakeAgent()
        self.ctrl = DirectChaseController()

    # --- Game state gating ---

    def test_only_moves_in_state_playing(self):
        for gc_state, expect_zero in [
            ("STATE_INITIAL", True),
            ("STATE_READY", True),
            ("STATE_SET", True),
            ("STATE_FINISHED", True),
            ("STATE_STANDBY", True),
            ("STATE_PLAYING", False),
        ]:
            with self.subTest(gc_state=gc_state):
                self.agent.gamecontroller.game_state = gc_state
                self.agent.current_cmd = [0.5, 0.0, 0.0]  # non-zero before
                self.ctrl.reset()
                self.ctrl.tick(self.agent)
                if expect_zero:
                    self.assertEqual(self.agent.current_cmd, [0.0, 0.0, 0.0])
                else:
                    self.assertNotEqual(self.agent.current_cmd, [0.0, 0.0, 0.0])

    def test_non_playing_resets_filter(self):
        self.agent.gamecontroller.game_state = "STATE_PLAYING"
        self.agent.ball_w = np.array([2.0, 0.0])
        self.ctrl.tick(self.agent)
        self.assertGreater(self.ctrl._last_vx, 0.0)

        self.agent.gamecontroller.game_state = "STATE_FINISHED"
        self.ctrl.tick(self.agent)
        self.assertEqual(self.ctrl._last_vx, 0.0)
        self.assertEqual(self.agent.current_cmd, [0.0, 0.0, 0.0])

    # --- Stop distance ---

    def test_stops_immediately_within_stop_distance(self):
        self.agent.gamecontroller.game_state = "STATE_PLAYING"
        self.ctrl._last_vx = 0.5  # residual speed
        self.agent.ball_local_override = [0.30, 0.0]  # <= 0.35
        self.ctrl.tick(self.agent)
        self.assertEqual(self.agent.current_cmd, [0.0, 0.0, 0.0])
        self.assertEqual(self.ctrl._last_vx, 0.0)

    def test_moves_when_beyond_stop_distance(self):
        self.agent.gamecontroller.game_state = "STATE_PLAYING"
        self.agent.ball_local_override = [1.0, 0.0]
        self.ctrl.tick(self.agent)
        self.assertGreater(self.agent.current_cmd[0], 0.0)

    # --- Ball loss ---

    def test_ball_loss_no_crash(self):
        self.agent.gamecontroller.game_state = "STATE_PLAYING"
        self.agent.ball_visible = False
        self.ctrl.tick(self.agent)
        self.assertAlmostEqual(self.agent.current_cmd[0], 0.0)
        self.assertNotEqual(self.agent.current_cmd[2], 0.0)

    def test_position_invalid_fallback_search_direction(self):
        self.agent.gamecontroller.game_state = "STATE_PLAYING"
        self.agent.ball_visible = False
        self.agent.robot_w = np.array([0.0, None])
        self.ctrl.tick(self.agent)
        self.assertGreater(self.agent.current_cmd[2], 0.0)

    # --- Agent isolation ---

    def test_no_shared_state_between_agents(self):
        agent_a = FakeAgent()
        agent_b = FakeAgent()
        ctrl_a = DirectChaseController()
        ctrl_b = DirectChaseController()
        agent_a.gamecontroller.game_state = "STATE_PLAYING"
        agent_b.gamecontroller.game_state = "STATE_PLAYING"
        agent_a.ball_w = np.array([1.0, 0.0])
        agent_b.ball_w = np.array([3.0, 0.0])

        # After different numbers of ticks, filter states should differ
        for _ in range(5):
            ctrl_a.tick(agent_a)
        ctrl_b.tick(agent_b)
        # Agent A has accumulated more filter history — its _last_vx should differ
        self.assertNotEqual(
            ctrl_a._last_vx, ctrl_b._last_vx,
            "Separate controllers must not share filter state",
        )

    # --- Forward-yaw envelope ---

    def test_turn_envelope_reduces_vx(self):
        cfg = DirectChaseConfig(
            stop_distance=0.35, vx_gain=1.0, vx_max=1.0,
            w_gain=10.0, w_max=1.2,
        )
        ctrl = DirectChaseController(cfg)
        agent = FakeAgent()
        agent.gamecontroller.game_state = "STATE_PLAYING"
        # Ball at 45° — large heading → large w, should trigger envelope
        agent.ball_local_override = [1.0, 1.0]
        ctrl.tick(agent)
        # With the envelope active at high |w|, vx should be reduced
        # below the raw gain * distance
        self.assertLess(agent.current_cmd[0], 1.0)
        self.assertGreater(abs(agent.current_cmd[2]), 0.0)

    def test_forward_speed_limit_allows_zero_vx(self):
        """At large |w|, forward speed is allowed to reach zero."""
        vx = forward_speed_limit(1.0, 1.5, yaw_full_speed=0.25,
                                 yaw_zero_speed=1.5, vx_max=1.0)
        self.assertEqual(vx, 0.0)

    def test_config_uses_command_filter_envelope_thresholds(self):
        self.agent._config["cmd_filter"]["yaw_full_speed"] = 0.10
        self.agent._config["cmd_filter"]["yaw_zero_speed"] = 0.80

        cfg = DirectChaseConfig.from_agent(self.agent)
        cfg.vx_gain = 2.0
        cfg.w_gain = 1.0
        cfg.w_max = 1.0
        cfg.vx_accel = 2.0
        cfg.w_accel = 2.0
        cfg.smooth_alpha = 1.0
        ctrl = DirectChaseController(cfg)
        self.agent.gamecontroller.game_state = "STATE_PLAYING"
        self.agent.ball_local_override = [
            2.0 * math.cos(0.45),
            2.0 * math.sin(0.45),
        ]
        ctrl.tick(self.agent)

        self.assertEqual(cfg.yaw_full_speed, 0.10)
        self.assertEqual(cfg.yaw_zero_speed, 0.80)
        self.assertAlmostEqual(self.agent.current_cmd[0], 0.5)
        self.assertAlmostEqual(self.agent.current_cmd[2], 0.45)

    # --- Search rotation ---

    def test_search_rotation_direction(self):
        # Use separate controllers to avoid filter state bleed between subtests
        for y_pos, expected_sign in [(1.0, -1.0), (-1.0, 1.0)]:
            with self.subTest(y_pos=y_pos):
                ctrl = DirectChaseController()
                agent = FakeAgent()
                agent.gamecontroller.game_state = "STATE_PLAYING"
                agent.ball_visible = False
                agent.robot_w = np.array([0.0, y_pos])
                ctrl.tick(agent)
                if expected_sign > 0:
                    self.assertGreater(agent.current_cmd[2], 0.0)
                else:
                    self.assertLess(agent.current_cmd[2], 0.0)

    # --- Acceleration ---

    def test_straight_acceleration_limited(self):
        self.agent.gamecontroller.game_state = "STATE_PLAYING"
        self.agent.ball_local_override = [3.0, 0.0]  # far away
        ctrl = DirectChaseController(DirectChaseConfig(
            vx_accel=0.08, smooth_alpha=1.0,  # alpha=1 → no extra smoothing
        ))
        ctrl.tick(self.agent)
        first_vx = self.agent.current_cmd[0]
        self.assertGreater(first_vx, 0.0)
        self.assertLessEqual(first_vx, 0.08)  # first step limited by accel

    # --- CLI and strategy routing ---

    def test_real_parser_exposes_strategy_choices(self):
        module = _load_decider_module()
        parser = module.build_arg_parser()

        self.assertEqual(parser.parse_args([]).sim_strategy, "push_to_goal")
        self.assertEqual(
            parser.parse_args(["--sim-strategy", "direct_chase"]).sim_strategy,
            "direct_chase",
        )

    def test_game_creates_and_reuses_direct_chase_controller(self):
        self.agent._sim_strategy = "direct_chase"

        game(self.agent)
        first = self.agent._direct_chase_controller
        game(self.agent)

        self.assertIs(self.agent._direct_chase_controller, first)
        self.assertFalse(hasattr(self.agent, "_decider_fsm"))

class ContinuousPushToGoalTests(unittest.TestCase):
    """Tests for the continuous PushToGoal ball-chasing controller."""

    def setUp(self):
        self.agent = FakeAgent()
        self.agent._config["push_to_goal"] = {
            "behind_offset": 0.12,
            "far_distance": 0.60,
            "push_vx_min": 0.35,
            "vx_gain": 2.5,
            "vx_max": 1.0,
            "vy_gain": 2.5,
            "vy_max": 0.20,
            "w_gain": 2.5,
            "w_max": 0.45,
            "turn_only_yaw_deg": 60.0,
            "search_w": 0.6,
        }
        self.cfg = PushToGoalConfig.from_config(self.agent._config)
        self.ctrl = ContinuousPushToGoal(self.cfg)

    # --- Parser ---

    def test_real_parser_default_is_push_to_goal(self):
        module = _load_decider_module()
        parser = module.build_arg_parser()
        self.assertEqual(
            parser.parse_args([]).sim_strategy, "push_to_goal"
        )

    def test_real_parser_rejects_fsm(self):
        module = _load_decider_module()
        parser = module.build_arg_parser()
        with self.assertRaises(SystemExit):
            parser.parse_args(["--sim-strategy", "fsm"])

    def test_real_parser_accepts_direct_chase(self):
        module = _load_decider_module()
        parser = module.build_arg_parser()
        self.assertEqual(
            parser.parse_args(["--sim-strategy", "direct_chase"]).sim_strategy,
            "direct_chase",
        )

    # --- game() dispatch ---

    def test_game_creates_and_reuses_push_to_goal_controller(self):
        self.agent._sim_strategy = "push_to_goal"

        game(self.agent)
        first = self.agent._push_to_goal_controller
        game(self.agent)

        self.assertIs(self.agent._push_to_goal_controller, first)

    def test_game_does_not_create_decider_fsm_for_push_to_goal(self):
        self.agent._sim_strategy = "push_to_goal"
        game(self.agent)
        self.assertFalse(hasattr(self.agent, "_decider_fsm"))

    def test_game_rejects_unknown_strategy(self):
        self.agent._sim_strategy = "bogus"
        with self.assertRaisesRegex(ValueError, "Unsupported simulation strategy"):
            game(self.agent)

    # --- State gating ---

    def test_stops_when_not_state_playing(self):
        for state in (
            "STATE_INITIAL", "STATE_READY", "STATE_SET",
            "STATE_FINISHED", "STATE_STANDBY",
        ):
            with self.subTest(state=state):
                self.agent.gamecontroller.game_state = state
                self.agent.robot_w = np.array([0.0, 0.0])
                self.agent.ball_w = np.array([3.0, 0.0])
                self.agent.ball_visible = True
                self.ctrl.tick(self.agent)
                self.assertEqual(
                    self.agent.current_cmd, [0.0, 0.0, 0.0],
                    f"Expected zero command in {state}",
                )

    def test_stops_when_position_unknown(self):
        # get_self_pos returns None → zero commands
        saved = self.agent.get_self_pos
        try:
            self.agent.get_self_pos = lambda: None
            self.ctrl.tick(self.agent)
            self.assertEqual(self.agent.current_cmd, [0.0, 0.0, 0.0])
        finally:
            self.agent.get_self_pos = saved

    # --- Ball loss ---

    def test_ball_loss_search_rotation(self):
        self.agent.ball_visible = False
        self.agent.robot_w = np.array([0.0, 0.0])
        self.ctrl.tick(self.agent)
        vx, vy, w = self.agent.current_cmd
        self.assertEqual(vx, 0.0)
        self.assertEqual(vy, 0.0)
        self.assertNotEqual(w, 0.0, "Expected search rotation")

    def test_ball_loss_default_search_direction_positive(self):
        self.agent.ball_visible = False
        self.agent.robot_w = np.array([0.0, 0.0])
        # _last_ball_side defaults to 0 → positive direction
        self.ctrl._last_ball_side = 0.0
        self.ctrl.tick(self.agent)
        self.assertGreater(self.agent.current_cmd[2], 0.0)

    def test_ball_loss_remembers_last_lateral_side_left(self):
        # Ball on left side → search left on loss
        self.agent.robot_w = np.array([0.0, 0.0])
        self.agent.ball_w = np.array([1.0, 0.5])  # ball to the left
        self.agent.robot_yaw_deg = 0.0
        self.agent.ball_visible = True
        self.ctrl.tick(self.agent)  # records _last_ball_side > 0

        self.assertGreater(self.ctrl._last_ball_side, 0.0)

        # Now lose ball
        self.agent.ball_visible = False
        self.ctrl.tick(self.agent)
        self.assertGreater(self.agent.current_cmd[2], 0.0,
                           "Expected positive search rotation after ball on left")

    def test_ball_loss_remembers_last_lateral_side_right(self):
        # Ball on right side → search right on loss
        self.agent.robot_w = np.array([0.0, 0.0])
        self.agent.ball_w = np.array([1.0, -0.5])  # ball to the right
        self.agent.robot_yaw_deg = 0.0
        self.agent.ball_visible = True
        self.ctrl.tick(self.agent)  # records _last_ball_side < 0

        self.assertLess(self.ctrl._last_ball_side, 0.0)

        # Now lose ball
        self.agent.ball_visible = False
        self.ctrl.tick(self.agent)
        self.assertLess(self.agent.current_cmd[2], 0.0,
                        "Expected negative search rotation after ball on right")

    # --- Motion: far ball ---

    def test_far_ball_straight_ahead_outputs_high_vx(self):
        self.agent.robot_w = np.array([0.0, 0.0])
        self.agent.ball_w = np.array([3.0, 0.0])  # 3m ahead
        self.agent.robot_yaw_deg = 0.0
        self.agent.ball_visible = True
        self.ctrl.tick(self.agent)

        vx, vy, w = self.agent.current_cmd
        # Ball is at (3,0), target behind ball at ~(2.88, 0)
        # target_distance ≈ 2.88, heading_error ≈ 0 → cos ≈ 1
        # vx = clip(2.5 * 2.88 * 1.0, 0, 1.0) = 1.0
        self.assertGreater(vx, 0.5, f"Expected high vx from far ball, got {vx}")

    # --- Motion: lateral ball ---

    def test_lateral_ball_strafes_and_turns_toward_ball(self):
        self.agent.robot_w = np.array([0.0, 0.0])
        # Ball 2m ahead, 1m left → behind-ball target also offset left
        self.agent.ball_w = np.array([2.0, 1.0])
        self.agent.robot_yaw_deg = 0.0
        self.agent.ball_visible = True
        self.ctrl.tick(self.agent)

        vx, vy, w = self.agent.current_cmd
        # Far from the ball, both lateral motion and yaw point toward it.
        self.assertGreater(vy, 0.0, f"Expected vy > 0 for left-side ball, got {vy}")
        self.assertGreater(w, 0.0, f"Expected w > 0 for left-side ball, got {w}")
        self.assertGreater(vx, 0.0, f"Expected vx > 0, got {vx}")

    # --- Motion: near ball (continuous push) ---

    def test_near_ball_does_not_stop(self):
        """Ball within 0.2m directly ahead still produces a push command."""
        self.agent.robot_w = np.array([0.0, 0.0])
        self.agent.ball_w = np.array([0.2, 0.0])  # very close
        self.agent.robot_yaw_deg = 0.0
        self.agent.ball_visible = True
        self.ctrl.tick(self.agent)

        vx, vy, w = self.agent.current_cmd
        self.assertGreater(vx, 0.0)

    def test_at_behind_target_keeps_push_speed(self):
        """Reaching the behind-ball point must not stop continuous pushing."""
        self.agent.robot_w = np.array([0.0, 0.0])
        # Ball at (0.12, 0) → behind-ball target at (0, 0) = robot position
        self.agent.ball_w = np.array([0.12, 0.0])
        self.agent.robot_yaw_deg = 0.0
        self.agent.ball_visible = True
        self.ctrl.tick(self.agent)

        vx = self.agent.current_cmd[0]
        self.assertGreaterEqual(vx, self.cfg.push_vx_min)

    def test_near_ball_from_side_targets_behind_point_before_goal(self):
        """A side approach turns toward the behind-ball target first."""
        self.agent.robot_w = np.array([0.0, 0.4])
        self.agent.ball_w = np.array([0.0, 0.0])
        self.agent.robot_yaw_deg = 0.0
        self.agent.ball_visible = True
        self.ctrl.tick(self.agent)

        vx, vy, w = self.agent.current_cmd
        self.assertEqual(vx, 0.0)
        self.assertEqual(vy, 0.0)
        self.assertLess(w, 0.0)

    def test_target_behind_robot_uses_turn_only_command(self):
        """Large heading error does not combine saturated strafe and turn."""
        self.agent.robot_w = np.array([1.0, 0.0])
        self.agent.ball_w = np.array([0.2, 0.3])
        self.agent.robot_yaw_deg = 0.0
        self.agent.ball_visible = True
        self.ctrl.tick(self.agent)

        vx, vy, w = self.agent.current_cmd
        self.assertEqual(vx, 0.0)
        self.assertEqual(vy, 0.0)
        self.assertNotEqual(w, 0.0)

    # --- Goal direction ---

    def test_red_goal_is_positive_x(self):
        self.agent.color = "red"
        self.agent.robot_w = np.array([0.0, 0.0])
        self.agent.ball_w = np.array([2.0, 0.0])
        self.agent.robot_yaw_deg = 0.0
        self.agent.ball_visible = True
        self.ctrl.tick(self.agent)

        # Red team: goal at +field_length/2 = +4.5 (M league)
        # Ball at (2, 0) → behind-ball target should be at (~1.88, 0)
        # Robot at (0, 0) → delta ≈ (+1.88, 0) → travel_yaw ≈ 0 → w ≈ 0
        self.assertAlmostEqual(self.agent.current_cmd[2], 0.0, delta=0.01)

    def test_blue_goal_is_negative_x(self):
        self.agent.color = "blue"
        self.agent.robot_w = np.array([0.0, 0.0])
        self.agent.ball_w = np.array([-2.0, 0.0])
        self.agent.robot_yaw_deg = 0.0
        self.agent.ball_visible = True
        self.ctrl.tick(self.agent)

        # Blue team: goal at -field_length/2 = -4.5
        # Ball at (-2, 0) → behind-ball target = ball - unit(ball→goal) * 0.12
        # ball→goal = (-4.5) - (-2) = -2.5 → unit = (-1, 0)
        # target_w = (-2, 0) - (-1, 0) * 0.12 = (-1.88, 0)
        # Robot at (0, 0) → delta = (-1.88, 0) → travel_yaw = π
        # heading_error = π - 0 = π → w should be constrained by w_max
        # Robot is behind the ball relative to its own goal → should turn around
        self.assertLess(self.agent.current_cmd[2], 0.0,
                        f"Blue team should turn toward negative goal, got w={self.agent.current_cmd[2]}")

    def test_red_team_ball_beyond_goal_stops(self):
        self.agent.color = "red"
        self.agent.robot_w = np.array([5.0, 0.0])
        self.agent.ball_w = np.array([5.2, 0.0])  # past goal at +4.5
        self.agent.robot_yaw_deg = 0.0
        self.agent.ball_visible = True
        self.ctrl.tick(self.agent)

        self.assertEqual(self.agent.current_cmd, [0.0, 0.0, 0.0])

    def test_blue_team_ball_beyond_goal_stops(self):
        self.agent.color = "blue"
        self.agent.robot_w = np.array([-5.0, 0.0])
        self.agent.ball_w = np.array([-5.2, 0.0])
        self.agent.robot_yaw_deg = 180.0
        self.agent.ball_visible = True
        self.ctrl.tick(self.agent)
        self.assertEqual(self.agent.current_cmd, [0.0, 0.0, 0.0])

    # --- Behind offset ---

    def test_behind_offset_creates_target_behind_ball(self):
        """The target point should be behind the ball relative to goal."""
        self.agent.color = "red"
        self.agent.robot_w = np.array([0.0, 0.0])
        self.agent.ball_w = np.array([2.0, 0.0])
        self.agent.robot_yaw_deg = 0.0
        self.agent.ball_visible = True
        self.ctrl.tick(self.agent)

        # Ball at (2, 0), goal at (4.5, 0)
        # to_goal = (2.5, 0) → unit = (1, 0)
        # target_w = (2, 0) - (1, 0)*0.12 = (1.88, 0)
        # Robot at (0, 0) → delta = (1.88, 0)
        # heading_error should be ~0 (target is straight ahead)
        vx, vy, w = self.agent.current_cmd
        self.assertAlmostEqual(w, 0.0, delta=0.01,
                               msg="Behind-ball target should be straight ahead")
        self.assertGreater(vx, 0.0, "Should move toward behind-ball target")

    # --- Field size from config ---

    def test_active_field_size_has_priority_in_controller(self):
        self.agent._config["active_field_size"] = [9.0, 6.0]
        self.agent._config["field_size"]["M"] = [14.0, 9.0]
        self.agent.robot_w = np.array([4.4, 0.0])
        self.agent.ball_w = np.array([4.6, 0.0])
        self.agent.robot_yaw_deg = 0.0
        self.agent.ball_visible = True
        self.ctrl.tick(self.agent)
        self.assertEqual(self.ctrl._field_length, 9.0)
        self.assertEqual(self.agent.current_cmd, [0.0, 0.0, 0.0])

    def test_far_distance_changes_approach_target(self):
        self.agent.robot_w = np.array([0.0, 0.0])
        self.agent.ball_w = np.array([0.4, 0.2])
        self.agent.robot_yaw_deg = 0.0
        self.agent.ball_visible = True

        far_ctrl = ContinuousPushToGoal(PushToGoalConfig(far_distance=0.30))
        far_ctrl.tick(self.agent)
        far_cmd = tuple(self.agent.current_cmd)

        near_ctrl = ContinuousPushToGoal(PushToGoalConfig(far_distance=1.00))
        near_ctrl.tick(self.agent)
        near_cmd = tuple(self.agent.current_cmd)

        self.assertNotEqual(far_cmd, near_cmd)

    # --- Config round-trip ---

    def test_push_to_goal_config_from_config(self):
        cfg = PushToGoalConfig.from_config(self.agent._config)
        self.assertEqual(cfg.behind_offset, 0.12)
        self.assertEqual(cfg.push_vx_min, 0.35)
        self.assertEqual(cfg.vx_gain, 2.5)
        self.assertEqual(cfg.w_max, 0.45)
        self.assertEqual(cfg.search_w, 0.6)

    def test_controller_default_config(self):
        ctrl = ContinuousPushToGoal()
        self.assertEqual(ctrl.cfg.behind_offset, 0.12)
        self.assertEqual(ctrl.cfg.vx_max, 1.0)


if __name__ == "__main__":
    unittest.main()
