#!/usr/bin/env python3
"""Build and archive the BB64 Stage-D3 categorical factor graph."""

from __future__ import annotations

import argparse
from collections import Counter
from datetime import datetime, timezone
import json
import math
from pathlib import Path
import time

from bb64_tmr_postselection.circuit import NoiseModel

from .circuit_decoder.factor_graph import build_d3_factor_graph, save_d3_factor_graph
from .model import load_hybrid_model
from .syndrome_history import build_syndrome_history_circuit


PACKAGE_DIR = Path(__file__).resolve().parent
DEFAULT_OUTPUT = PACKAGE_DIR / "results" / "d3_factor_graph_p1e3.npz"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--theta", type=float, default=math.pi / 32)
    parser.add_argument("--probability", type=float, default=0.001)
    parser.add_argument("--syndrome-rounds", type=int, default=3)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--summary", type=Path)
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def _atomic_json(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    temporary.replace(path)


def main() -> None:
    args = parse_args()
    summary = args.summary or args.output.with_suffix(".json")
    for path in (args.output, summary):
        if path.exists() and not args.overwrite:
            raise SystemExit(f"refusing to overwrite {path}")
    print("checkpoint phase=construct_circuit status=start", flush=True)
    model = load_hybrid_model()
    history = build_syndrome_history_circuit(
        model,
        theta=args.theta,
        noise=NoiseModel(probability=args.probability),
        syndrome_rounds=args.syndrome_rounds,
    )
    started = time.monotonic()
    print("checkpoint phase=build_factor_graph status=start", flush=True)
    graph = build_d3_factor_graph(
        model,
        history,
        theta=args.theta,
        progress=lambda message: print(f"checkpoint {message}", flush=True),
    )
    build_seconds = time.monotonic() - started
    print(
        f"checkpoint phase=build_factor_graph status=done seconds={build_seconds:.3f} "+
        f"variables={graph.num_variables} components={graph.num_components}",
        flush=True,
    )
    save_started = time.monotonic()
    save_d3_factor_graph(args.output, graph)
    save_seconds = time.monotonic() - save_started
    kinds = Counter(variable.kind for variable in graph.variables)
    payload = {
        "schema": "bb64-d3-factor-graph-summary-v1",
        "completed": True,
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "theta": args.theta,
        "physical_probability": args.probability,
        "syndrome_rounds": args.syndrome_rounds,
        "detectors": graph.parsed_circuit.num_detectors,
        "physical_variables": graph.physical_variable_count,
        "tmr_branch_variables": len(graph.branch_variable_indices),
        "variables": graph.num_variables,
        "binary_components": graph.num_components,
        "variables_by_kind": dict(sorted(kinds.items())),
        "archive": str(args.output),
        "archive_bytes": args.output.stat().st_size,
        "timing_seconds": {"build": build_seconds, "save": save_seconds},
    }
    _atomic_json(summary, payload)
    print(
        f"checkpoint phase=complete archive={args.output} summary={summary} "+
        f"save_seconds={save_seconds:.3f}",
        flush=True,
    )


if __name__ == "__main__":
    main()
