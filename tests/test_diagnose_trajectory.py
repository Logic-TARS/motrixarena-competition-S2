#!/usr/bin/env python3
"""Unit tests for decider.scripts.diagnose_trajectory."""

from __future__ import annotations

import json
import os
import sys
import unittest
from pathlib import Path

# Add decider/ to sys.path so we can import scripts.* (same pattern as decider.py).
_PROJECT = Path(__file__).resolve().parent.parent
_decider_dir = _PROJECT / "decider"
if str(_decider_dir) not in sys.path:
    sys.path.insert(0, str(_decider_dir))

from scripts.diagnose_trajectory import (
    _analyze_state_layer,
    _analyze_condition_layer,
    _build_suggestions,
    _identify_primary_blocker,
    _load_thresholds,
    diagnose_trajectory,
)
from scripts.analyze_trajectory import load_rows

FIXTURES = Path(__file__).resolve().parent / "fixtures"


class StateLayerTests(unittest.TestCase):
    """Layer 1 — state diagnosis."""

    def test_chase_only_never_05m(self):
        rows = load_rows(str(FIXTURES / "traj_never_05m.csv"))
        state = _analyze_state_layer(rows)
        self.assertEqual(state["states_entered"], ["SEARCH", "CHASE"])
        self.assertTrue(state["did_reach_chase"])
        self.assertFalse(state["did_reach_align_behind"])
        self.assertFalse(state["did_reach_push_forward"])
        self.assertIsNone(state["time_to_enter_0_5m"])
        self.assertGreater(state["ball_dist_min"], 0.5)

    def test_alignment_no_push_forward(self):
        rows = load_rows(str(FIXTURES / "traj_no_push_forward.csv"))
        state = _analyze_state_layer(rows)
        self.assertIn("ALIGN_BEHIND", state["states_entered"])
        self.assertTrue(state["did_reach_chase"])
        self.assertTrue(state["did_reach_align_behind"])
        self.assertFalse(state["did_reach_push_forward"])
        self.assertIsNotNone(state["time_to_enter_0_5m"])
        self.assertIn("ALIGN_FACE_GOAL", state["align_mode_durations_s"])

    def test_full_push_forward(self):
        rows = load_rows(str(FIXTURES / "traj_push_no_progress.csv"))
        state = _analyze_state_layer(rows)
        self.assertIn("PUSH_FORWARD", state["states_entered"])
        self.assertTrue(state["did_reach_push_forward"])
        self.assertIsNotNone(state["time_to_enter_0_5m"])


class ConditionLayerTests(unittest.TestCase):
    """Layer 2 — condition diagnosis."""

    def setUp(self):
        self.thresholds, _ = _load_thresholds(use_external_config=False)

    def test_yaw_zero_pass_rate_on_yaw_blocked_trajectory(self):
        rows = load_rows(str(FIXTURES / "traj_no_push_forward.csv"))
        state = _analyze_state_layer(rows)
        cond = _analyze_condition_layer(rows, state, self.thresholds)
        yaw = cond["conditions"]["yaw_error_small"]
        self.assertEqual(yaw["pass_rate"], 0.0)
        self.assertGreaterEqual(yaw["worst_value"], 30.0)

    def test_all_conditions_pass_on_push_trajectory(self):
        rows = load_rows(str(FIXTURES / "traj_push_no_progress.csv"))
        state = _analyze_state_layer(rows)
        # Find PUSH_FORWARD start, analyze ALIGN_BEHIND before it
        cond = _analyze_condition_layer(rows, state, self.thresholds)
        yaw = cond["conditions"]["yaw_error_small"]
        # yaw should pass at least sometimes before PUSH_FORWARD
        self.assertGreater(yaw["pass_rate"], 0.5)

    def test_ball_x_window_statistics(self):
        rows = load_rows(str(FIXTURES / "traj_no_push_forward.csv"))
        state = _analyze_state_layer(rows)
        cond = _analyze_condition_layer(rows, state, self.thresholds)
        bx = cond["conditions"]["ball_local_x_in_window"]
        self.assertIsNotNone(bx["first_pass_time"])
        self.assertIsNotNone(bx["worst_value"])
        self.assertIsNotNone(bx["closest_value"])
        # failing_seconds may be 0 if all ball_local_x values are in window
        self.assertGreaterEqual(bx["failing_seconds"], 0.0)
        self.assertIsNotNone(bx["latest_value"])


class PrimaryBlockerTests(unittest.TestCase):
    """Primary blocker identification."""

    def setUp(self):
        self.thresholds, _ = _load_thresholds(use_external_config=False)

    def test_chase_too_slow_when_never_align_behind(self):
        rows = load_rows(str(FIXTURES / "traj_never_05m.csv"))
        state = _analyze_state_layer(rows)
        cond = _analyze_condition_layer(rows, state, self.thresholds)
        blocker = _identify_primary_blocker(cond, state, self.thresholds)
        self.assertEqual(blocker["condition"], "chase_too_slow_or_bad_heading")

    def test_yaw_blocker_on_yaw_blocked_trajectory(self):
        rows = load_rows(str(FIXTURES / "traj_no_push_forward.csv"))
        state = _analyze_state_layer(rows)
        cond = _analyze_condition_layer(rows, state, self.thresholds)
        blocker = _identify_primary_blocker(cond, state, self.thresholds)
        # align_face ran <2s AND yaw 0% → align_face_goal_too_late
        # But our fixture has ALIGN_FACE_GOAL < 2s, so this should fire
        self.assertIn(
            blocker["condition"],
            {"yaw_error_small", "align_face_goal_too_late"},
        )

    def test_no_progress_blocker_when_push_forward_no_goal_progress(self):
        rows = load_rows(str(FIXTURES / "traj_push_no_progress.csv"))
        state = _analyze_state_layer(rows)
        self.assertTrue(state["did_reach_push_forward"])
        # When PUSH_FORWARD is reached, primary_blocker is set by
        # diagnose_trajectory, not by _identify_primary_blocker.
        # Test the full pipeline instead.
        result = diagnose_trajectory(
            str(FIXTURES / "traj_push_no_progress.csv"),
            use_external_config=False,
        )
        push_exec = result["push_execution"]
        self.assertTrue(push_exec["entered_push_forward"])
        # Ball in fixture stays at y=0.28 throughout PUSH,
        # no progress toward goal at x=4.5
        self.assertFalse(push_exec["has_goal_progress"])


class SuggestionTests(unittest.TestCase):
    """Suggestion generation."""

    def test_each_blocker_has_suggestions(self):
        blockers = [
            {"condition": "chase_too_slow_or_bad_heading", "pass_rate": None},
            {"condition": "align_face_goal_too_late", "pass_rate": 0.0},
            {"condition": "yaw_error_small", "pass_rate": 0.0},
            {"condition": "ball_local_x_in_window", "pass_rate": 0.0},
            {"condition": "ball_local_y_centered", "pass_rate": 0.0},
            {"condition": "lateral_err_small", "pass_rate": 0.0},
            {"condition": "behind_depth_sufficient", "pass_rate": 0.0},
        ]
        empty_state = {"align_mode_durations_s": {}}
        empty_exec = {"entered_push_forward": False}
        for pb in blockers:
            suggestions = _build_suggestions(pb, empty_state, empty_exec)
            self.assertIsInstance(suggestions, list)
            self.assertGreater(len(suggestions), 0, msg=f"No suggestions for {pb['condition']}")

    def test_empty_suggestions_when_healthy(self):
        suggestions = _build_suggestions(
            {}, {"align_mode_durations_s": {}}, {"entered_push_forward": False}
        )
        self.assertTrue(any("正常" in s for s in suggestions))


class EndToEndTests(unittest.TestCase):
    """Full pipeline tests through diagnose_trajectory()."""

    def test_e2e_never_05m(self):
        result = diagnose_trajectory(
            str(FIXTURES / "traj_never_05m.csv"),
            use_external_config=False,
        )
        self.assertEqual(result["meta"]["run_mode"], "fsm_mvp")
        self.assertFalse(result["state_summary"]["did_reach_push_forward"])
        self.assertEqual(
            result["primary_blocker"]["condition"],
            "chase_too_slow_or_bad_heading",
        )
        self.assertIn("condition_analysis", result)
        self.assertIn("suggestions", result)

    def test_e2e_no_push_forward(self):
        result = diagnose_trajectory(
            str(FIXTURES / "traj_no_push_forward.csv"),
            use_external_config=False,
        )
        self.assertFalse(result["state_summary"]["did_reach_push_forward"])
        self.assertIn("condition_analysis", result)
        pb = result["primary_blocker"]
        self.assertIn(
            pb["condition"],
            {"yaw_error_small", "align_face_goal_too_late"},
        )

    def test_e2e_push_no_progress(self):
        result = diagnose_trajectory(
            str(FIXTURES / "traj_push_no_progress.csv"),
            use_external_config=False,
        )
        self.assertTrue(result["state_summary"]["did_reach_push_forward"])
        pe = result["push_execution"]
        self.assertTrue(pe["entered_push_forward"])
        self.assertFalse(pe["has_goal_progress"])

    def test_e2e_writes_output_files(self):
        import tempfile
        with tempfile.TemporaryDirectory() as tmpdir:
            result = diagnose_trajectory(
                str(FIXTURES / "traj_no_push_forward.csv"),
                output_dir=tmpdir,
                use_external_config=False,
            )
            json_path = Path(tmpdir) / "diagnosis.json"
            txt_path = Path(tmpdir) / "diagnosis.txt"
            self.assertTrue(json_path.exists())
            self.assertTrue(txt_path.exists())
            # Verify JSON is valid
            with json_path.open() as f:
                data = json.load(f)
            self.assertEqual(data["meta"]["run_mode"], "fsm_mvp")
            # Verify TXT contains expected sections
            txt = txt_path.read_text(encoding="utf-8")
            self.assertIn("结论:", txt)
            self.assertIn("关键证据:", txt)
            self.assertIn("建议:", txt)


class AcceptanceTests(unittest.TestCase):
    """Acceptance test against the real fsm_mvp_30e trajectory."""

    TRAJ_PATH = "/tmp/fsm_mvp_30e_traj/trajectory.csv"

    def test_acceptance_fsm_mvp_30e(self):
        if not Path(self.TRAJ_PATH).exists():
            self.skipTest(f"Trajectory not found: {self.TRAJ_PATH}")
        result = diagnose_trajectory(self.TRAJ_PATH, use_external_config=False)
        # State
        self.assertEqual(result["meta"]["run_mode"], "fsm_mvp")
        self.assertFalse(result["state_summary"]["did_reach_push_forward"])
        self.assertAlmostEqual(
            result["state_summary"]["align_mode_durations_s"].get("ALIGN_FACE_GOAL", 0),
            0.18,
            delta=0.05,
        )
        self.assertAlmostEqual(
            result["state_summary"]["time_to_enter_0_5m"] or 0,
            28.93,
            delta=0.15,
        )
        # Conditions
        conds = result["condition_analysis"]["conditions"]
        self.assertEqual(conds["yaw_error_small"]["pass_rate"], 0.0)
        self.assertGreater(conds["yaw_error_small"]["worst_value"], 100.0)
        self.assertAlmostEqual(conds["yaw_error_small"]["latest_value"], 115.0, delta=2.0)
        # Primary blocker
        pb = result["primary_blocker"]
        self.assertIn(
            pb["condition"],
            {"yaw_error_small", "align_face_goal_too_late"},
        )
        # Suggestions exist
        self.assertGreater(len(result["suggestions"]), 0)

    def test_json_only_mode(self):
        if not Path(self.TRAJ_PATH).exists():
            self.skipTest(f"Trajectory not found: {self.TRAJ_PATH}")
        # Simulate --json-only via diagnose_trajectory internals
        from scripts.analyze_trajectory import load_rows as _load_csv_rows
        from scripts.diagnose_trajectory import (
            _analyze_state_layer as _state,
            _analyze_condition_layer as _cond,
            _identify_primary_blocker as _blocker,
        )

        rows = _load_csv_rows(str(Path(self.TRAJ_PATH)))
        thresholds, _ = _load_thresholds(use_external_config=False)
        state = _state(rows)
        con = _cond(rows, state, thresholds)
        pb = _blocker(con, state, thresholds)
        self.assertIsNotNone(pb)
        self.assertIn(
            pb["condition"],
            {"yaw_error_small", "align_face_goal_too_late"},
        )


class EdgeCaseTests(unittest.TestCase):
    """Graceful degradation on malformed / incomplete data."""

    def test_empty_csv(self):
        import tempfile
        with tempfile.TemporaryDirectory() as tmpdir:
            empty_csv = Path(tmpdir) / "empty.csv"
            empty_csv.write_text("elapsed_s,run_mode,fsm_state\n")
            result = diagnose_trajectory(
                str(empty_csv), use_external_config=False
            )
            self.assertIn("error", result)

    def test_missing_fsm_columns(self):
        import tempfile
        with tempfile.TemporaryDirectory() as tmpdir:
            csv_path = Path(tmpdir) / "no_fsm.csv"
            csv_path.write_text(
                "elapsed_s,run_mode,fsm_state,align_mode,ball_distance,"
                "ball_x,ball_y,ball_local_x,ball_local_y,"
                "ball_to_goal_yaw_err_deg,lateral_err,behind_depth,"
                "cmd_vy,field_length,team,robot_x,robot_y\n"
                "0.0,fsm_mvp,,,,3.0,0.0,3.0,3.0,0.0,0.0,0.0,0.0,0.0,9.0,red,0.0,3.0\n"
                "1.0,fsm_mvp,,,,3.0,0.0,3.0,3.0,0.0,0.0,0.0,0.0,0.0,9.0,red,0.0,3.0\n"
            )
            result = diagnose_trajectory(
                str(csv_path), use_external_config=False
            )
            # Should succeed but with empty state analysis
            self.assertFalse(result["state_summary"]["did_reach_push_forward"])
            self.assertEqual(result["state_summary"]["states_entered"], [])

    def test_compile(self):
        """Verify the script compiles cleanly (py_compile)."""
        import py_compile
        import io

        script = (
            Path(__file__).resolve().parent.parent
            / "decider"
            / "scripts"
            / "diagnose_trajectory.py"
        )
        self.assertTrue(script.exists(), f"Script not found: {script}")
        try:
            py_compile.compile(str(script), doraise=True)
        except py_compile.PyCompileError as e:
            self.fail(f"Compile failed: {e}")


if __name__ == "__main__":
    unittest.main()
