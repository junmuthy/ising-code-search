"""CLI for the exhaustive automorphism-dual ``C4 x C5`` search."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from c4xc5_search.automorphism_dual import search_automorphism_dual


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-name", required=True)
    parser.add_argument("--seed-restarts", type=int, default=4)
    parser.add_argument("--random-seed", type=int, default=40504)
    parser.add_argument("--maximum-certifications", type=int, default=0)
    args = parser.parse_args()
    output = Path("results/c4xc5-search") / args.run_name
    if output.exists():
        raise FileExistsError(f"refusing to overwrite existing run: {output}")
    output.mkdir(parents=True)
    structural_path = output / "structural-candidates.jsonl"
    progress_path = output / "progress.json"

    def checkpoint(progress: dict[str, Any]) -> None:
        temporary = output / "progress.tmp.json"
        temporary.write_text(json.dumps(progress, indent=2, sort_keys=True) + "\n")
        temporary.replace(progress_path)
        print(json.dumps(progress, sort_keys=True), flush=True)

    with structural_path.open("w") as structural_stream:
        def save_candidate(record: dict[str, Any]) -> None:
            structural_stream.write(json.dumps(record, sort_keys=True) + "\n")
            structural_stream.flush()

        checkpoint(
            {
                "status": "starting",
                "processed": 0,
                "structural_candidates": 0,
            }
        )
        result = search_automorphism_dual(
            seed_restarts=args.seed_restarts,
            random_seed=args.random_seed,
            maximum_certifications=args.maximum_certifications,
            progress_callback=checkpoint,
            candidate_callback=save_candidate,
        )
    structural = result.pop("structural_candidates")
    survivors = result.pop("survivors")
    near_misses = result.pop("near_misses")
    survivors_path = output / "survivors.jsonl"
    with survivors_path.open("w") as handle:
        for record in survivors:
            handle.write(json.dumps(record, sort_keys=True) + "\n")
    near_misses_path = output / "near-misses.jsonl"
    with near_misses_path.open("w") as handle:
        for record in near_misses:
            handle.write(json.dumps(record, sort_keys=True) + "\n")
    summary = {
        **result,
        "structural_candidates": len(structural),
        "survivors": len(survivors),
        "near_misses": len(near_misses),
        "structural_candidates_path": str(structural_path),
        "survivors_path": str(survivors_path),
        "near_misses_path": str(near_misses_path),
        "output": str(output),
    }
    (output / "summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n"
    )
    checkpoint(
        {
            "status": "complete",
            "processed": result["counters"].get("polynomials", 0),
            "structural_candidates": len(structural),
            "survivors": len(survivors),
            "seconds": result["seconds"],
            "summary_path": str(output / "summary.json"),
        }
    )
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
