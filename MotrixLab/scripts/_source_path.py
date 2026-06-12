from pathlib import Path
import sys


def ensure_source_path() -> None:
    root_dir = Path(__file__).resolve().parents[1]
    for src_dir in (root_dir / "motrix_envs" / "src", root_dir / "motrix_rl" / "src"):
        src = str(src_dir)
        if src not in sys.path:
            sys.path.insert(0, src)
