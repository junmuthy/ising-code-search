#!/usr/bin/env python3
"""Exactly synthesize a systematic `[[16,2,6]]` CSS code or prove UNSAT."""

from __future__ import annotations

import argparse
import json
import tempfile
import time
from pathlib import Path
from typing import Any

import z3

from searches.distance_first_c2.search import analyze_css
from searches.distance_first_c2.systematic import SystematicCSSSolver


def atomic_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(
        "w", dir=path.parent, delete=False, encoding="utf-8"
    ) as output:
        temporary = Path(output.name)
        output.write(json.dumps(value, indent=2, sort_keys=True) + "\n")
    temporary.replace(path)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--run-name", required=True)
    parser.add_argument("--length", type=int, choices=(14, 16, 18, 20), default=16)
    parser.add_argument("--target-distance", type=int, default=6)
    parser.add_argument("--timeout-seconds", type=float, default=600)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    output = args.output_root / args.run_name
    if output.exists():
        raise SystemExit(f"refusing to overwrite existing run: {output}")
    output.mkdir(parents=True)
    started = time.perf_counter()
    solver = SystematicCSSSolver(
        args.length, timeout_ms=max(1, int(1000 * args.timeout_seconds))
    )
    stages: list[dict[str, Any]] = []

    def checkpoint(status: str, **extra: Any) -> None:
        record = {
            "status": status,
            "n": args.length,
            "k": 2,
            "rank_x": solver.rank,
            "rank_z": solver.rank,
            "target_distance": args.target_distance,
            "constraints_by_weight": solver.constraints_by_weight,
            "stages": stages,
            "seconds": round(time.perf_counter() - started, 6),
            **extra,
        }
        atomic_json(output / "progress.json", record)
        print(json.dumps(record, sort_keys=True), flush=True)

    atomic_json(
        output / "arguments.json",
        {**vars(args), "output_root": str(args.output_root)},
    )
    checkpoint("starting")
    final_status = "unknown"
    survivor: dict[str, Any] | None = None
    for weight in range(1, args.target_distance):
        build_started = time.perf_counter()
        count = solver.add_weight(weight)
        checkpoint(
            "weight-built",
            active_weight=weight,
            operators_at_weight=count,
            build_seconds=round(time.perf_counter() - build_started, 6),
        )
        check_started = time.perf_counter()
        status = solver.check()
        stage: dict[str, Any] = {
            "weight": weight,
            "operators": count,
            "status": str(status),
            "solve_seconds": round(time.perf_counter() - check_started, 6),
        }
        if status == z3.sat:
            analysis = analyze_css(solver.state())
            stage.update(
                {
                    "model_distance": analysis["distance"],
                    "model_distance_x": analysis["distance_x"],
                    "model_distance_z": analysis["distance_z"],
                    "model_minimum_maximum_check_weight": analysis[
                        "minimum_maximum_check_weight"
                    ],
                }
            )
            atomic_json(output / f"model-through-weight-{weight}.json", analysis)
            if weight == args.target_distance - 1:
                survivor = analysis
                final_status = "sat"
        elif status == z3.unsat:
            final_status = "unsat"
        else:
            final_status = "unknown"
            stage["reason_unknown"] = solver.solver.reason_unknown()
        stages.append(stage)
        checkpoint("stage-complete", active_weight=weight)
        if status != z3.sat:
            break
    summary = {
        "target": f"systematic [[{args.length},2,>={args.target_distance}]] CSS synthesis",
        "status": final_status,
        "complete_parameterization_up_to_qubit_permutation": True,
        "stages": stages,
        "survivor": survivor is not None,
        "seconds": round(time.perf_counter() - started, 6),
    }
    if survivor is not None:
        atomic_json(output / "survivor.json", survivor)
    atomic_json(output / "summary.json", summary)
    checkpoint(final_status)
    print(json.dumps(summary, indent=2, sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
