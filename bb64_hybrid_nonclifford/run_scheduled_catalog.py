#!/usr/bin/env python3
"""Build the exact BB64 single-fault catalog and selected pair extension."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
import math
from pathlib import Path
import time

from bb64_tmr_postselection.circuit import NoiseModel

from .circuit_decoder.archive import save_fault_catalog
from .circuit_decoder.catalog import add_selected_pairs, build_fault_catalog
from .model import load_hybrid_model
from .syndrome_history import build_syndrome_history_circuit


PACKAGE_DIR = Path(__file__).resolve().parent
DEFAULT_OUTPUT = PACKAGE_DIR / "results" / "scheduled_fault_catalog_p1e3.npz"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--theta", type=float, default=math.pi / 32)
    parser.add_argument("--probability", type=float, default=0.001)
    parser.add_argument("--syndrome-rounds", type=int, default=3)
    parser.add_argument("--pair-top-groups", type=int, default=64)
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
    model = load_hybrid_model()
    history = build_syndrome_history_circuit(
        model,
        theta=args.theta,
        noise=NoiseModel(probability=args.probability),
        syndrome_rounds=args.syndrome_rounds,
    )
    started = time.monotonic()
    print("checkpoint phase=single_catalog status=start", flush=True)
    catalog = build_fault_catalog(
        history,
        model.code,
        progress=lambda message: print(f"checkpoint {message}", flush=True),
    )
    single_seconds = time.monotonic() - started
    print(
        f"checkpoint phase=single_catalog status=done seconds={single_seconds:.3f} "
        f"groups={len(catalog.single_groups)}",
        flush=True,
    )
    pair_started = time.monotonic()
    catalog = add_selected_pairs(
        catalog,
        model.code,
        top_groups=args.pair_top_groups,
        progress=lambda message: print(f"checkpoint {message}", flush=True),
    )
    pair_seconds = time.monotonic() - pair_started
    print(
        f"checkpoint phase=pair_catalog status=done seconds={pair_seconds:.3f} "
        f"groups={len(catalog.pair_groups)}",
        flush=True,
    )
    save_fault_catalog(args.output, catalog)
    ledger = catalog.ledger
    payload = {
        "schema": "bb64-scheduled-fault-catalog-summary-v1",
        "completed": True,
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "theta": args.theta,
        "physical_probability": args.probability,
        "syndrome_rounds": args.syndrome_rounds,
        "locations": len(ledger.locations),
        "elementary_mechanisms": len(ledger.mechanisms),
        "unique_single_signatures": len(catalog.single_groups),
        "pair_top_groups": args.pair_top_groups,
        "unique_selected_pair_signatures": len(catalog.pair_groups),
        "probability_mass": {
            "zero_faults": ledger.no_fault_probability,
            "exactly_one_fault": ledger.single_fault_probability,
            "exactly_two_faults_all_pairs": ledger.exactly_two_fault_probability,
            "through_two_faults_all_pairs": ledger.through_two_fault_probability,
            "modeled_zero_single_selected_pairs": catalog.modeled_absolute_probability,
            "conservative_omitted_tail": catalog.conservative_tail_probability,
        },
        "timing_seconds": {"singles": single_seconds, "selected_pairs": pair_seconds},
        "archive": str(args.output),
    }
    _atomic_json(summary, payload)
    print(f"checkpoint phase=complete archive={args.output} summary={summary}", flush=True)


if __name__ == "__main__":
    main()
