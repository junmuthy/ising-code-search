#!/usr/bin/env python3
"""Run or resume a manifest of bicycle-chain postselection experiments."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import subprocess
import sys
from typing import Any


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--stop-after", type=int, default=0)
    return parser.parse_args()


def load_json(path: Path) -> Any:
    with path.open(encoding="utf-8") as source:
        return json.load(source)


def completed(path: Path) -> bool:
    if not path.exists():
        return False
    try:
        return bool(load_json(path).get("completed"))
    except (OSError, ValueError, TypeError):
        return False


def main() -> None:
    args = parse_args()
    manifest = load_json(args.manifest)
    tasks = manifest["tasks"]
    args.output_dir.mkdir(parents=True, exist_ok=True)
    launched = 0
    for index, task in enumerate(tasks, start=1):
        slug = str(task["slug"])
        output = args.output_dir / f"{slug}.json"
        if completed(output):
            print(f"sweep checkpoint task={index}/{len(tasks)} slug={slug} status=skip-complete", flush=True)
            continue
        command = [
            sys.executable,
            "-m",
            "bicycle_chain_l2_paper.tmr_postselection.run_postselection",
            "--output",
            str(output),
        ]
        for key, value in task["arguments"].items():
            flag = "--" + key.replace("_", "-")
            if isinstance(value, bool):
                if value:
                    command.append(flag)
            else:
                command.extend((flag, str(value)))
        if output.exists():
            command.append("--resume")
        print(
            f"sweep checkpoint task={index}/{len(tasks)} slug={slug} status=start",
            flush=True,
        )
        subprocess.run(command, check=True)
        launched += 1
        print(
            f"sweep checkpoint task={index}/{len(tasks)} slug={slug} status=done",
            flush=True,
        )
        if args.stop_after and launched >= args.stop_after:
            print(f"sweep checkpoint status=stopped-after launched={launched}", flush=True)
            break


if __name__ == "__main__":
    main()
