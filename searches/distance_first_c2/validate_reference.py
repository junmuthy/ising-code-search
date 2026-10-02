#!/usr/bin/env python3
"""Regression-check the distance engine on the saved `[[32,4,6]]` code."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from searches.distance_first_c2.search import CSSState, analyze_css, canonical_basis


ROOT = Path(__file__).resolve().parents[2]
DEFAULT_REFERENCE = (
    ROOT
    / "codes/n32_k4_d6_reference_code"
    / "results"
    / "folded-local-20x250-260828-v1"
    / "start-results"
    / "start-010.json"
)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--reference", type=Path, default=DEFAULT_REFERENCE)
    args = parser.parse_args()
    record = json.loads(args.reference.read_text(encoding="utf-8"))
    saved = record["result"]["best"]
    state = CSSState(
        32,
        canonical_basis(saved["stabilizer_masks_x"], 32),
        canonical_basis(saved["stabilizer_masks_z"], 32),
    )
    result = analyze_css(state)
    checks = {
        "n": result["n"] == 32,
        "k": result["k"] == 4,
        "distance_x": result["distance_x"] == 6,
        "distance_z": result["distance_z"] == 6,
        "css_orthogonal": result["css_orthogonal"],
        "minimum_check_weight": result["minimum_maximum_check_weight"] == 8,
    }
    output = {
        "source": str(args.reference),
        "checks": checks,
        "all_checks_pass": all(checks.values()),
        "analysis": result,
    }
    print(json.dumps(output, indent=2, sort_keys=True), flush=True)
    if not output["all_checks_pass"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
