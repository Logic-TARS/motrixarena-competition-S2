import ast
from pathlib import Path
import unittest


SIM_PATH = Path(__file__).resolve().parents[1] / "app" / "multi_robot_sim.py"


class NoAutomaticFallResetTest(unittest.TestCase):
    def test_fallen_robots_are_not_teleported_upright(self):
        source = SIM_PATH.read_text(encoding="utf-8")
        tree = ast.parse(source)
        function_names = {
            node.name for node in ast.walk(tree) if isinstance(node, ast.FunctionDef)
        }

        self.assertNotIn("_recover_fallen_robots", function_names)
        self.assertNotIn("_enable_fall_recovery", source)
        self.assertNotIn("FALL_RESET_PROTECT_SEC", source)

    def test_explicit_drag_reset_remains_available(self):
        source = SIM_PATH.read_text(encoding="utf-8")

        self.assertIn("DRAG_RESET_PROTECT_SEC", source)
        self.assertIn("def _hold_robot_at_reset_pose", source)

    def test_policy_recovery_does_not_write_base_state(self):
        source = SIM_PATH.read_text(encoding="utf-8")
        tree = ast.parse(source)
        recovery_functions = {
            node.name: node
            for node in ast.walk(tree)
            if isinstance(node, ast.FunctionDef)
            and node.name in {
                "_update_k1_recovery_states",
                "_apply_k1_recovery_overrides",
            }
        }
        self.assertEqual(
            set(recovery_functions),
            {"_update_k1_recovery_states", "_apply_k1_recovery_overrides"},
        )
        for name, node in recovery_functions.items():
            text = ast.get_source_segment(source, node) or ""
            writes = [
                line.strip()
                for line in text.splitlines()
                if line.strip().startswith(("self.data.dof_pos[", "self.data.dof_vel["))
                and "=" in line
            ]
            self.assertEqual(writes, [], name)
            self.assertNotIn("set_dof_pos", text, name)
            self.assertNotIn("set_dof_vel", text, name)


if __name__ == "__main__":
    unittest.main()
