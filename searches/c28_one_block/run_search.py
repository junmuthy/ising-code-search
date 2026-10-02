#!/usr/bin/env python3
"""Run the exhaustive one-block ``C28`` search without overwriting results."""

from __future__ import annotations

import argparse
import json
import time
from collections import Counter
from pathlib import Path

from searches.c28_one_block.search import (
    analyze_mask,
    enumerate_structural_masks,
    enumeration_summary,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-name", required=True)
    parser.add_argument(
        "--maximum-certifications",
        type=int,
        default=0,
        help="zero certifies every structural survivor",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    output_dir = Path("results/c28-one-block") / args.run_name
    if output_dir.exists():
        raise FileExistsError(f"refusing to overwrite existing run: {output_dir}")
    output_dir.mkdir(parents=True)
    started = time.perf_counter()
    masks, considered, self_orthogonal, rank_twelve, canonical = (
        enumerate_structural_masks()
    )
    masks = [int(mask) for mask in masks]
    enumeration = enumeration_summary(
        considered, self_orthogonal, rank_twelve, canonical
    )
    limit = args.maximum_certifications or len(masks)
    low_weight_counts: Counter[int] = Counter()
    survivors = []
    with (output_dir / "candidates.jsonl").open("w") as stream:
        for index, mask in enumerate(masks):
            analysis = analyze_mask(mask, certify_distance=index < limit)
            record = {"candidate_index": index, **analysis}
            stream.write(json.dumps(record) + "\n")
            if analysis["low_logical_below_six"] is not None:
                low_weight_counts[analysis["low_logical_below_six"]["weight"]] += 1
            if analysis["certified_distance_at_least_six"]:
                survivors.append(record)
    result = {
        "target": "[[28,4,>=6]] one-block weakly self-dual C28 CSS",
        "method": "exhaustive-even-sector-weight-6-8-10-12",
        "enumeration": enumeration,
        "canonical_structural_candidates": len(masks),
        "distance_certifications": min(limit, len(masks)),
        "low_logical_weight_counts": {
            str(weight): count for weight, count in sorted(low_weight_counts.items())
        },
        "distance_at_least_six_survivors": len(survivors),
        "survivors": survivors,
        "seconds": round(time.perf_counter() - started, 6),
        "output": str(output_dir),
    }
    (output_dir / "summary.json").write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
