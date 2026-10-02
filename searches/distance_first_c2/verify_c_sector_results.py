#!/usr/bin/env python3
"""Verify coverage and statuses of a fixed-C systematic search."""

from __future__ import annotations

import argparse
import json
import tempfile
from pathlib import Path
from typing import Any

from searches.distance_first_c2.systematic import canonical_c_sectors


def atomic_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(
        "w", dir=path.parent, delete=False, encoding="utf-8"
    ) as output:
        temporary = Path(output.name)
        output.write(json.dumps(value, indent=2, sort_keys=True) + "\n")
    temporary.replace(path)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--length", type=int, choices=(14, 16, 18, 20))
    args = parser.parse_args()
    arguments_path = args.run / "arguments.json"
    arguments = json.loads(arguments_path.read_text(encoding="utf-8"))
    n = int(args.length if args.length is not None else arguments["length"])
    rank = (n - 2) // 2
    expected = canonical_c_sectors(rank)
    raw_total = 2 ** (2 * rank)
    paths = sorted((args.run / "sectors").glob("sector-*.json"))
    records = [json.loads(path.read_text(encoding="utf-8")) for path in paths]
    records_by_index = {int(record["index"]): record for record in records}
    coverage_checks = {
        "expected_sector_count": len(expected)
        == int(arguments["canonical_sectors_total"]),
        "expected_result_files": len(paths) == len(expected),
        "unique_result_indices": len(records_by_index) == len(records),
        "all_indices_present": set(records_by_index) == set(range(len(expected))),
        "raw_multiplicity_is_complete": sum(
            sector.raw_multiplicity for sector in expected
        )
        == raw_total,
        "counts_match_catalog": all(
            tuple(records_by_index[index]["counts"]) == sector.counts
            and int(records_by_index[index]["raw_multiplicity"])
            == sector.raw_multiplicity
            for index, sector in enumerate(expected)
        ),
        "all_d5_unsat": all(
            record.get("d5_status") == "unsat" for record in records
        ),
        "no_unknowns": all(
            record.get("d5_status") != "unknown"
            and record.get("d6_status") != "unknown"
            for record in records
        ),
        "no_worker_errors": all("error" not in record for record in records),
    }
    result = {
        "run": str(args.run),
        "parameterization": {
            "n": n,
            "k": 2,
            "rank_x": rank,
            "rank_z": rank,
            "form": f"H_X=[I_{rank}|A], B=[I_{rank}|C], H_Z=[B A^T|B]",
            "canonical_c_sectors": len(expected),
            "raw_c_matrices": raw_total,
        },
        "coverage_checks": coverage_checks,
        "all_checks_pass": all(coverage_checks.values()),
        "conclusion": (
            f"no balanced binary CSS [[{n},2,d>=5]] code with "
            f"rank_x=rank_z={rank} exists"
            if all(coverage_checks.values())
            else "coverage or solver-status validation failed"
        ),
    }
    atomic_json(args.output, result)
    print(json.dumps(result, indent=2, sort_keys=True), flush=True)
    if not result["all_checks_pass"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
