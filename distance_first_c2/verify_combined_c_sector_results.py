#!/usr/bin/env python3
"""Combine staged and direct fixed-C runs into a complete d>=6 certificate."""

from __future__ import annotations

import argparse
import json
import tempfile
from collections import Counter
from pathlib import Path
from typing import Any

from distance_first_c2.systematic import canonical_c_sectors


def atomic_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(
        "w", dir=path.parent, delete=False, encoding="utf-8"
    ) as output:
        temporary = Path(output.name)
        output.write(json.dumps(value, indent=2, sort_keys=True) + "\n")
    temporary.replace(path)


def load_records(root: Path) -> dict[int, dict[str, Any]]:
    records = [
        json.loads(path.read_text(encoding="utf-8"))
        for path in sorted((root / "sectors").glob("sector-*.json"))
    ]
    return {int(record["index"]): record for record in records}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--primary", type=Path, required=True)
    parser.add_argument("--supplement", type=Path, action="append", default=[])
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--length", type=int, choices=(14, 16, 18, 20), default=18)
    args = parser.parse_args()
    n = args.length
    rank = (n - 2) // 2
    sectors = canonical_c_sectors(rank)
    primary = load_records(args.primary)
    supplements = [load_records(root) for root in args.supplement]
    final_status: dict[int, str] = {}
    source: dict[int, str] = {}
    for index in range(len(sectors)):
        record = primary.get(index)
        if record is None:
            final_status[index] = "missing"
            continue
        if record.get("d5_status") == "unsat":
            final_status[index] = "unsat"
            source[index] = "primary-d5-unsat"
            continue
        if record.get("d6_status") in ("sat", "unsat"):
            final_status[index] = str(record["d6_status"])
            source[index] = "primary-d6"
            continue
        for supplement_index, supplement in enumerate(supplements):
            extra = supplement.get(index)
            if extra is not None and extra.get("d6_status") in ("sat", "unsat"):
                final_status[index] = str(extra["d6_status"])
                source[index] = f"supplement-{supplement_index}-d6"
                break
        else:
            final_status[index] = "unknown"
    primary_d5 = Counter(
        record.get("d5_status", "missing") for record in primary.values()
    )
    final_counts = Counter(final_status.values())
    checks = {
        "complete_primary_index_set": set(primary) == set(range(len(sectors))),
        "catalog_raw_multiplicity_complete": sum(
            sector.raw_multiplicity for sector in sectors
        )
        == 2 ** (2 * rank),
        "all_sectors_resolved_for_d6": set(final_status.values()) <= {"sat", "unsat"},
        "all_sectors_d6_unsat": set(final_status.values()) == {"unsat"},
        "no_primary_worker_errors": all("error" not in record for record in primary.values()),
        "no_supplement_worker_errors": all(
            "error" not in record
            for supplement in supplements
            for record in supplement.values()
        ),
    }
    result = {
        "target": f"balanced binary CSS [[{n},2,d>=6]]",
        "parameterization": {
            "rank_x": rank,
            "rank_z": rank,
            "canonical_c_sectors": len(sectors),
            "raw_c_matrices": 2 ** (2 * rank),
        },
        "primary": str(args.primary),
        "supplements": [str(path) for path in args.supplement],
        "primary_d5_status_counts": dict(sorted(primary_d5.items())),
        "combined_d6_status_counts": dict(sorted(final_counts.items())),
        "resolution_sources": dict(sorted(Counter(source.values()).items())),
        "checks": checks,
        "all_checks_pass": all(checks.values()),
        "conclusion": (
            f"no balanced binary CSS [[{n},2,d>=6]] code with "
            f"rank_x=rank_z={rank} exists"
            if all(checks.values())
            else "combined d>=6 coverage is incomplete or contains a survivor"
        ),
    }
    atomic_json(args.output, result)
    print(json.dumps(result, indent=2, sort_keys=True), flush=True)
    if not result["all_checks_pass"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
