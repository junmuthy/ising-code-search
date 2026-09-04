#!/usr/bin/env python3
"""Write summaries of bicycle-chain postselection result JSON files."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any


FIELDS = (
    "file",
    "mode",
    "protocol",
    "postselection_policy",
    "N",
    "M",
    "theta_over_pi",
    "p",
    "attempted",
    "accepted",
    "acceptance",
    "standard_error",
    "wilson_low",
    "wilson_high",
    "ideal_acceptance",
    "noise_survival_ratio",
    "candidate_yield",
    "peak_active_width",
    "physical_cnots",
    "cnot_layers",
    "shots_per_second",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("result_dir", type=Path)
    parser.add_argument("--csv", type=Path, required=True)
    parser.add_argument("--markdown", type=Path, required=True)
    return parser.parse_args()


def row(path: Path) -> dict[str, Any]:
    with path.open(encoding="utf-8") as source:
        record = json.load(source)
    config = record["configuration"]
    counts = record["counts"]
    estimates = record["estimates"]
    circuit = record["circuit"]
    return {
        "file": path.name,
        "mode": config["mode"],
        "protocol": config["protocol"],
        "postselection_policy": config.get("postselection_policy", "strict-xz"),
        "N": config["logical_count"],
        "M": config["partition_count"],
        "theta_over_pi": config["theta"] / __import__("math").pi,
        "p": config["probability"],
        "attempted": counts["attempted"],
        "accepted": counts["accepted"],
        "acceptance": estimates["block_acceptance"],
        "standard_error": estimates["block_acceptance_standard_error"],
        "wilson_low": estimates["block_acceptance_wilson_95"][0],
        "wilson_high": estimates["block_acceptance_wilson_95"][1],
        "ideal_acceptance": estimates["ideal_tmr_acceptance"],
        "noise_survival_ratio": estimates["noise_survival_ratio"],
        "candidate_yield": estimates["accepted_candidate_resources_per_attempt"],
        "peak_active_width": circuit["peak_active_width"],
        "physical_cnots": circuit["operation_counts"]["physical_cnots"],
        "cnot_layers": circuit["operation_counts"]["cnot_layers"],
        "shots_per_second": record["timing"]["attempted_shots_per_second"],
    }


def display(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, float):
        return f"{value:.8g}"
    return str(value)


def main() -> None:
    args = parse_args()
    rows = [row(path) for path in sorted(args.result_dir.glob("*.json"))]
    args.csv.parent.mkdir(parents=True, exist_ok=True)
    with args.csv.open("w", newline="", encoding="utf-8") as destination:
        writer = csv.DictWriter(destination, fieldnames=FIELDS, lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)

    headers = (
        "file", "mode", "protocol", "policy", "N", "M", "theta/pi", "p",
        "acceptance", "s.e.", "95% Wilson", "ideal", "yield", "shots/s",
    )
    lines = [
        "| " + " | ".join(headers) + " |",
        "|" + "|".join("---" for _ in headers) + "|",
    ]
    for item in rows:
        values = (
            f"`{item['file']}`",
            f"`{item['mode']}`",
            f"`{item['protocol']}`",
            f"`{item['postselection_policy']}`",
            item["N"],
            item["M"],
            item["theta_over_pi"],
            item["p"],
            item["acceptance"],
            item["standard_error"],
            f"[{item['wilson_low']:.8g}, {item['wilson_high']:.8g}]",
            item["ideal_acceptance"],
            item["candidate_yield"],
            item["shots_per_second"],
        )
        lines.append("| " + " | ".join(display(value) for value in values) + " |")
    args.markdown.parent.mkdir(parents=True, exist_ok=True)
    args.markdown.write_text("# Postselection run summary\n\n" + "\n".join(lines) + "\n", encoding="utf-8")
    print(f"wrote {args.csv} and {args.markdown}", flush=True)


if __name__ == "__main__":
    main()
