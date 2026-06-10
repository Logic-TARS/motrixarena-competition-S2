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


if __name__ == "__main__":
    unittest.main()
