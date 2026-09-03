#!/usr/bin/env python3
"""Run or resume a manifest of teleportation-estimator experiments."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import subprocess
import sys


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--stop-after", type=int, default=0)
    return parser.parse_args()


def _complete(path: Path) -> bool:
    if not path.exists():
        return False
    try:
        return bool(json.loads(path.read_text(encoding="utf-8")).get("completed"))
    except (OSError, ValueError, TypeError):
        return False


def main() -> None:
    args = parse_args()
    manifest = json.loads(args.manifest.read_text(encoding="utf-8"))
    tasks = manifest["tasks"]
    args.output_dir.mkdir(parents=True, exist_ok=True)
    launched = 0
    for index, task in enumerate(tasks, start=1):
        output = args.output_dir / f"{task['slug']}.json"
        if _complete(output):
            print(
                f"sweep checkpoint task={index}/{len(tasks)} slug={task['slug']} "
                "status=skip-complete",
                flush=True,
            )
            continue
        command = [
            sys.executable,
            "-m",
            "n22_k2_d6_shortened_golay.teleportation.run_estimator",
            "--output",
            str(output),
        ]
        for key, value in task["arguments"].items():
            command.extend(("--" + key.replace("_", "-"), str(value)))
        if output.exists():
            command.append("--resume")
        print(
            f"sweep checkpoint task={index}/{len(tasks)} slug={task['slug']} status=start",
            flush=True,
        )
        subprocess.run(command, check=True)
        launched += 1
        print(
            f"sweep checkpoint task={index}/{len(tasks)} slug={task['slug']} status=done",
            flush=True,
        )
        if args.stop_after and launched >= args.stop_after:
            print(f"sweep checkpoint status=stopped-after launched={launched}", flush=True)
            return


if __name__ == "__main__":
    main()
