#!/usr/bin/env python3
"""Create a reproducible manifest for a candidate policy artifact."""

import argparse
import hashlib
import json
import subprocess
from datetime import datetime, timezone
from pathlib import Path


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("policy", type=Path)
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--iteration", type=int, required=True)
    parser.add_argument("--env", required=True)
    parser.add_argument("--evaluation", type=Path)
    parser.add_argument("--config", type=Path)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    commit = subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip()
    manifest = {
        "policy": str(args.policy.resolve()),
        "sha256": sha256(args.policy),
        "git_commit": commit,
        "environment": args.env,
        "seed": args.seed,
        "iteration": args.iteration,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "evaluation": str(args.evaluation.resolve()) if args.evaluation else None,
        "config_snapshot": str(args.config.resolve()) if args.config else None,
    }
    output = args.output or args.policy.with_suffix(args.policy.suffix + ".manifest.json")
    output.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(output)


if __name__ == "__main__":
    main()
