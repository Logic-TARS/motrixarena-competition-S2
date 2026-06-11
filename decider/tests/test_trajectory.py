import csv
import json
import subprocess
import tempfile
import time
import unittest
from pathlib import Path

from scripts.analyze_trajectory import analyze_trajectory, build_summary
from trajectory import TRAJECTORY_FIELDS, TrajectoryRecorder

from test_directional_kick import DECIDER_DIR, DeciderFSM, FakeAgent, _load_decider_module


class TrajectoryRecorderTests(unittest.TestCase):
    def test_streams_rows_and_flushes_every_twenty_frames(self):
        with tempfile.TemporaryDirectory() as tmp:
            recorder = TrajectoryRecorder(tmp, flush_interval=20)
            for frame in range(19):
                recorder.write({"frame": frame, "run_mode": "fixed_command"})
            self.assertEqual(recorder._rows_since_flush, 19)
            recorder.write({"frame": 19, "run_mode": "fixed_command"})
            self.assertEqual(recorder._rows_since_flush, 0)
            recorder.close()

            with recorder.csv_path.open(newline="", encoding="utf-8") as handle:
                rows = list(csv.DictReader(handle))
            self.assertEqual(len(rows), 20)
            self.assertEqual(list(rows[0].keys()), TRAJECTORY_FIELDS)
            self.assertEqual(rows[-1]["frame"], "19")

    def test_sim_agent_records_sent_command_with_returned_pose(self):
        module = _load_decider_module()
        agent = module.SimAgent.__new__(module.SimAgent)

        class CapturingRecorder:
            def __init__(self):
                self.rows = []

            def write(self, row):
                self.rows.append(row)

        class GameController:
            game_state = "STATE_PLAYING"

        recorder = CapturingRecorder()
        agent._trajectory_recorder = recorder
        agent._trajectory_frame = 7
        agent._trajectory_started_perf = time.perf_counter() - 1.0
        agent._trajectory_last_perf = None
        agent._config = {"id": 0, "active_field_size": [9.0, 6.0]}
        agent._active_field_length = 9.0
        agent._active_field_width = 6.0
        agent.sim_fixed_cmd = [0.5, 0.0, 0.0]
        agent.color = "red"
        agent.gamecontroller = GameController()
        agent.get_self_pos = lambda: [1.25, -0.5]
        agent.get_self_yaw = lambda: 12.0
        agent.get_ball_pos_in_map = lambda: [2.0, 0.25]
        agent.get_ball_pos = lambda: [0.75, 0.1]
        agent.get_ball_distance = lambda: 0.76

        response = {
            "sim_timestamp": 123.0,
            "state": {"ball": {"x": 2.0, "y": 0.25, "z": 0.09}},
        }
        original_resolver = module.user_entry.resolve_field_size
        module.user_entry.resolve_field_size = lambda _config: self.fail(
            "field size must be cached before trajectory recording"
        )
        try:
            agent._record_trajectory(response, [0.5, -0.1, 0.2])
        finally:
            module.user_entry.resolve_field_size = original_resolver

        row = recorder.rows[0]
        self.assertEqual(row["frame"], 7)
        self.assertEqual(row["cmd_vx"], 0.5)
        self.assertEqual(row["cmd_vy"], -0.1)
        self.assertEqual(row["cmd_w"], 0.2)
        self.assertEqual(row["robot_x"], 1.25)
        self.assertEqual(row["ball_x"], 2.0)
        self.assertEqual(row["run_mode"], "fixed_command")
        self.assertNotIn("fsm_state", row)

    def test_sim_agent_records_selected_direct_chase_mode(self):
        module = _load_decider_module()
        agent = module.SimAgent.__new__(module.SimAgent)

        class CapturingRecorder:
            def __init__(self):
                self.rows = []

            def write(self, row):
                self.rows.append(row)

        class GameController:
            game_state = "STATE_PLAYING"

        recorder = CapturingRecorder()
        agent._trajectory_recorder = recorder
        agent._trajectory_frame = 0
        agent._trajectory_started_perf = time.perf_counter()
        agent._trajectory_last_perf = None
        agent._config = {"id": 0}
        agent._active_field_length = 9.0
        agent._active_field_width = 6.0
        agent.sim_fixed_cmd = None
        agent._sim_strategy = "direct_chase"
        agent.color = "red"
        agent.gamecontroller = GameController()
        agent.get_self_pos = lambda: [0.0, 0.0]
        agent.get_self_yaw = lambda: 0.0
        agent.get_ball_pos_in_map = lambda: [2.0, 0.0]
        agent.get_ball_pos = lambda: [2.0, 0.0]
        agent.get_ball_distance = lambda: 2.0

        response = {
            "sim_timestamp": 123.0,
            "state": {"ball": {"x": 2.0, "y": 0.0, "z": 0.09}},
        }
        agent._record_trajectory(response, [0.8, 0.0, 0.0])

        self.assertEqual(recorder.rows[0]["run_mode"], "direct_chase")
        self.assertNotIn("fsm_state", recorder.rows[0])


class SnapshotTests(unittest.TestCase):
    def test_snapshot_is_serializable_and_has_no_side_effects(self):
        agent = FakeAgent()
        fsm = DeciderFSM(agent)
        agent.ball_w[:] = [3.0, 0.0]
        agent.robot_w[:] = [2.72, 0.0]
        agent.ball_local_override = [0.28, 0.0]
        fsm._last_can_kick_reason = "sentinel"
        before_filter = fsm.cmd_filter.last_cmd
        before_state = fsm.state

        snapshot = fsm.get_snapshot()
        json.dumps(snapshot)

        self.assertEqual(fsm.state, before_state)
        self.assertEqual(fsm._last_can_kick_reason, "sentinel")
        self.assertEqual(fsm.cmd_filter.last_cmd, before_filter)
        self.assertTrue(snapshot["can_kick"])
        self.assertEqual(snapshot["can_kick_reason"], "ok")
        self.assertAlmostEqual(snapshot["behind_depth"], 0.28)

    def test_snapshot_keeps_geometry_during_kick_cooldown(self):
        agent = FakeAgent()
        fsm = DeciderFSM(agent)
        agent.ball_w[:] = [3.0, 0.0]
        agent.robot_w[:] = [2.72, 0.0]
        agent.ball_local_override = [0.28, 0.0]
        fsm._kick_cooldown_sec = 2.0
        fsm.last_kick_time = time.time()

        snapshot = fsm.get_snapshot()

        self.assertFalse(snapshot["can_kick"])
        self.assertEqual(snapshot["can_kick_reason"], "cooldown")
        self.assertAlmostEqual(snapshot["behind_depth"], 0.28)
        self.assertAlmostEqual(snapshot["lateral_err"], 0.0)

    def test_snapshot_exposes_align_submode(self):
        agent = FakeAgent()
        fsm = DeciderFSM(agent)
        fsm.state = "ALIGN_BEHIND_BALL"
        agent.push_to_goal.last_mode = "ORBIT_SIDE"

        snapshot = fsm.get_snapshot()

        self.assertEqual(snapshot["align_mode"], "ORBIT_SIDE")

class TrajectoryAnalysisTests(unittest.TestCase):
    def _rows(self):
        return [
            {
                "frame": "0",
                "elapsed_s": "0.0",
                "run_mode": "fsm",
                "fsm_state": "ALIGN_BEHIND_BALL",
                "align_mode": "TURN",
                "ball_x": "0.0",
                "ball_y": "0.0",
                "field_length": "9.0",
                "can_kick": "true",
                "can_kick_reason": "ok",
                "ball_local_x": "0.28",
                "ball_local_y": "0.0",
                "behind_depth": "0.28",
                "lateral_err": "0.0",
                "ball_to_goal_yaw_err_deg": "0.0",
                "distance_to_goal": "4.5",
            },
            {
                "frame": "1",
                "elapsed_s": "0.1",
                "run_mode": "fsm",
                "fsm_state": "KICK",
                "align_mode": "",
                "kick_push": "false",
                "ball_x": "0.0",
                "ball_y": "0.0",
                "field_length": "9.0",
            },
            {
                "frame": "2",
                "elapsed_s": "0.2",
                "run_mode": "fsm",
                "fsm_state": "KICK",
                "align_mode": "",
                "kick_push": "true",
                "ball_x": "0.0",
                "ball_y": "0.0",
                "field_length": "9.0",
            },
            {
                "frame": "3",
                "elapsed_s": "0.3",
                "run_mode": "fsm",
                "fsm_state": "KICK",
                "align_mode": "",
                "kick_push": "true",
                "ball_x": "0.2",
                "ball_y": "0.0",
                "field_length": "9.0",
            },
            {
                "frame": "4",
                "elapsed_s": "0.4",
                "run_mode": "fsm",
                "fsm_state": "RECOVER",
                "align_mode": "",
                "ball_x": "0.2",
                "ball_y": "0.0",
                "field_length": "9.0",
            },
        ]

    def test_summary_uses_pre_kick_row_and_goal_projection(self):
        summary = build_summary(self._rows())
        self.assertTrue(summary["fsm_enabled"])
        self.assertEqual(summary["kick_count"], 1)
        kick = summary["kicks"][0]
        self.assertEqual(kick["pre_kick"]["behind_depth"], 0.28)
        displacement = kick["ball_displacement"]
        self.assertAlmostEqual(displacement["distance"], 0.2)
        self.assertAlmostEqual(displacement["toward_goal_projection"], 0.2)
        self.assertAlmostEqual(displacement["direction_error_deg"], 0.0)
        self.assertAlmostEqual(
            summary["align_mode_durations_s"]["TURN"],
            0.1,
        )

    def test_summary_counts_side_recovery_phase_durations(self):
        rows = [
            {
                "elapsed_s": "0.0",
                "run_mode": "fsm",
                "fsm_state": "SIDE_RECOVERY",
                "side_recovery_phase": "BYPASS_INFIELD",
            },
            {
                "elapsed_s": "0.2",
                "run_mode": "fsm",
                "fsm_state": "SIDE_RECOVERY",
                "side_recovery_phase": "CROSS_OUTSIDE",
            },
            {
                "elapsed_s": "0.5",
                "run_mode": "fsm",
                "fsm_state": "SIDE_RECOVERY",
                "side_recovery_phase": "PUSH",
            },
            {
                "elapsed_s": "0.8",
                "run_mode": "fsm",
                "fsm_state": "APPROACH_BALL",
                "side_recovery_phase": "",
            },
        ]

        summary = build_summary(rows)

        self.assertAlmostEqual(
            summary["side_recovery_phase_durations_s"]["BYPASS_INFIELD"],
            0.2,
        )
        self.assertAlmostEqual(
            summary["side_recovery_phase_durations_s"]["CROSS_OUTSIDE"],
            0.3,
        )
        self.assertAlmostEqual(
            summary["side_recovery_phase_durations_s"]["PUSH"],
            0.3,
        )

    def test_fixed_command_summary_is_non_fsm(self):
        rows = [
            {
                "elapsed_s": "0.0",
                "run_mode": "fixed_command",
                "fsm_state": "",
            },
            {
                "elapsed_s": "1.0",
                "run_mode": "fixed_command",
                "fsm_state": "",
            },
        ]
        summary = build_summary(rows)
        self.assertEqual(summary["run_mode"], "fixed_command")
        self.assertFalse(summary["fsm_enabled"])
        self.assertEqual(summary["kick_count"], 0)

    def test_push_to_goal_summary_includes_new_metrics(self):
        rows = [
            {
                "elapsed_s": "0.0",
                "run_mode": "push_to_goal",
                "team": "red",
                "robot_x": "0.0",
                "robot_y": "0.0",
                "ball_x": "2.0",
                "ball_y": "0.0",
                "ball_distance": "2.0",
                "cmd_vx": "1.0",
                "cmd_vy": "0.0",
                "cmd_w": "0.0",
                "field_length": "9.0",
                "field_width": "6.0",
                "fsm_state": "",
                "game_state": "STATE_PLAYING",
            },
            {
                "elapsed_s": "0.5",
                "run_mode": "push_to_goal",
                "team": "red",
                "robot_x": "0.5",
                "robot_y": "0.0",
                "ball_x": "2.5",
                "ball_y": "0.0",
                "ball_distance": "0.30",
                "cmd_vx": "0.6",
                "cmd_vy": "0.0",
                "cmd_w": "0.0",
                "field_length": "9.0",
                "field_width": "6.0",
                "fsm_state": "",
                "game_state": "STATE_PLAYING",
            },
        ]
        summary = build_summary(rows)
        self.assertEqual(summary["run_mode"], "push_to_goal")
        self.assertFalse(summary["fsm_enabled"])

        ptg = summary.get("push_to_goal", {})
        self.assertIsNotNone(ptg)
        self.assertAlmostEqual(ptg["ball_dist_initial"], 2.0)
        self.assertAlmostEqual(ptg["ball_dist_min"], 0.30)
        self.assertAlmostEqual(ptg["ball_dist_final"], 0.30)
        self.assertAlmostEqual(ptg["time_to_enter_0_5m"], 0.5)
        self.assertAlmostEqual(ptg["robot_displacement"], 0.5)
        self.assertAlmostEqual(ptg["goal_direction_ball_progress"], 0.5)

        cmd_stats = ptg["cmd_statistics"]
        self.assertAlmostEqual(cmd_stats["vx_mean"], 0.8)
        self.assertAlmostEqual(ptg["near_ball_vx_mean"], 0.6)
        self.assertIsNone(ptg["time_to_cross_goal_line"])

    def test_blue_push_to_goal_progress_is_positive_toward_negative_x(self):
        rows = [
            {
                "elapsed_s": "0.0",
                "run_mode": "push_to_goal",
                "team": "blue",
                "robot_x": "0.0",
                "robot_y": "0.0",
                "ball_x": "-1.0",
                "ball_y": "0.0",
                "ball_distance": "1.0",
                "cmd_vx": "0.8",
                "cmd_vy": "-0.2",
                "cmd_w": "-0.4",
                "field_length": "9.0",
                "fsm_state": "",
            },
            {
                "elapsed_s": "1.0",
                "run_mode": "push_to_goal",
                "team": "blue",
                "robot_x": "-0.5",
                "robot_y": "0.0",
                "ball_x": "-4.6",
                "ball_y": "0.0",
                "ball_distance": "0.4",
                "cmd_vx": "0.4",
                "cmd_vy": "0.1",
                "cmd_w": "0.2",
                "field_length": "9.0",
                "fsm_state": "",
            },
        ]
        ptg = build_summary(rows)["push_to_goal"]
        self.assertGreater(ptg["goal_direction_ball_progress"], 0.0)
        self.assertEqual(ptg["time_to_cross_goal_line"], 1.0)
        self.assertEqual(ptg["cmd_statistics"]["vy_abs_max"], 0.2)
        self.assertEqual(ptg["cmd_statistics"]["w_abs_max"], 0.4)

    def test_analyzer_writes_summary_and_plots(self):
        with tempfile.TemporaryDirectory() as tmp:
            output_dir = Path(tmp)
            csv_path = output_dir / "trajectory.csv"
            rows = self._rows()
            with csv_path.open("w", newline="", encoding="utf-8") as handle:
                writer = csv.DictWriter(handle, fieldnames=TRAJECTORY_FIELDS)
                writer.writeheader()
                for row in rows:
                    writer.writerow(row)

            analyze_trajectory(csv_path)

            self.assertTrue((output_dir / "summary.json").is_file())
            self.assertTrue((output_dir / "trajectory_xy.png").is_file())
            self.assertTrue((output_dir / "trajectory_timeseries.png").is_file())


class ShellInterfaceTests(unittest.TestCase):
    def test_match_config_defaults_to_playing_push_to_goal(self):
        command = """
set -u
REPO_ROOT=/tmp/repo
source scripts/match_config.sh
printf '%s\\n%s\\n%s\\n' "$REFEREE_ARG" "$REFEREE_STATE_ARG" "$FIXED_CMD"
"""
        result = subprocess.run(
            ["bash", "-c", command],
            cwd=DECIDER_DIR.parent,
            check=True,
            text=True,
            capture_output=True,
        )
        self.assertEqual(
            result.stdout.splitlines(),
            ["--use-referee", "--referee-state playing", ""],
        )

    def test_fixed_command_preserves_playing_defaults(self):
        command = """
set -u
REPO_ROOT=/tmp/repo
source scripts/match_config.sh
parse_args --sim-fixed-cmd 0.5,0,0
printf '%s\\n%s\\n%s\\n' "$REFEREE_ARG" "$REFEREE_STATE_ARG" "$FIXED_CMD"
"""
        result = subprocess.run(
            ["bash", "-c", command],
            cwd=DECIDER_DIR.parent,
            check=True,
            text=True,
            capture_output=True,
        )
        self.assertEqual(
            result.stdout.splitlines(),
            [
                "--use-referee",
                "--referee-state playing",
                "--sim-fixed-cmd 0.5,0,0",
            ],
        )

    def test_match_config_parses_trajectory_directory(self):
        command = """
set -u
REPO_ROOT=/tmp/repo
source scripts/match_config.sh
parse_args --trajectory-dir '/tmp/trajectory path' --play
printf '%s\\n%s\\n%s\\n' "$TRAJECTORY_ENABLED" "$TRAJECTORY_DIR" "$FIXED_CMD"
"""
        result = subprocess.run(
            ["bash", "-c", command],
            cwd=DECIDER_DIR.parent,
            check=True,
            text=True,
            capture_output=True,
        )
        self.assertEqual(
            result.stdout.splitlines(),
            ["1", "/tmp/trajectory path", ""],
        )

    def test_match_config_accepts_trajectory_long_flag(self):
        command = """
set -u
REPO_ROOT=/tmp/repo
source scripts/match_config.sh
parse_args --trajectory
printf '%s\\n' "$TRAJECTORY_ENABLED"
"""
        result = subprocess.run(
            ["bash", "-c", command],
            cwd=DECIDER_DIR.parent,
            check=True,
            text=True,
            capture_output=True,
        )
        self.assertEqual(result.stdout.strip(), "1")


if __name__ == "__main__":
    unittest.main()
