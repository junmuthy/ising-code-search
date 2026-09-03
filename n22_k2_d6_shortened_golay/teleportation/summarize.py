#!/usr/bin/env python3
"""Create compact tables and an SVG summary from ignored raw result JSON."""

from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("result_dir", type=Path)
    parser.add_argument("--calibration-csv", type=Path, required=True)
    parser.add_argument("--rus-csv", type=Path, required=True)
    parser.add_argument("--svg", type=Path, required=True)
    return parser.parse_args()


def _calibration_rows(result_dir: Path) -> list[dict[str, object]]:
    rows = []
    for path in result_dir.glob("pi*-n*-m*.json"):
        record = json.loads(path.read_text(encoding="utf-8"))
        if record.get("schema") != "n22-teleportation-estimator-v1":
            continue
        config = record["configuration"]
        estimates = record["estimates"]
        logicals = estimates["per_logical"]
        rows.append(
            {
                "file": path.name,
                "theta": float(config["theta"]),
                "theta_over_pi": float(config["theta"]) / math.pi,
                "logical_count": int(config["logical_count"]),
                "partition_count": int(config["partition_count"]),
                "attempted": int(record["counts"]["attempted"]),
                "accepted": int(record["counts"]["accepted"]),
                "acceptance": float(estimates["preparation_acceptance"]),
                "acceptance_se": float(estimates["preparation_acceptance_standard_error"]),
                "logical_0_infidelity": float(logicals[0]["conditional_infidelity"]),
                "logical_0_se": float(logicals[0]["standard_error"]),
                "logical_1_infidelity": (
                    float(logicals[1]["conditional_infidelity"]) if len(logicals) == 2 else ""
                ),
                "logical_1_se": float(logicals[1]["standard_error"]) if len(logicals) == 2 else "",
                "all_cancel_any_infidelity": float(
                    estimates["all_cancel_any_logical_infidelity"]
                ),
                "all_cancel_samples": int(record["counts"]["all_cancel"]),
            }
        )
    return sorted(rows, key=lambda row: (float(row["theta"]), int(row["logical_count"])))


def _histogram_se(histogram: dict[str, int]) -> float:
    total = sum(histogram.values())
    if total < 2:
        return 0.0
    mean = sum(int(value) * count for value, count in histogram.items()) / total
    variance = (
        sum((int(value) - mean) ** 2 * count for value, count in histogram.items())
        / (total - 1)
    )
    return math.sqrt(variance / total)


def _rus_rows(result_dir: Path) -> list[dict[str, object]]:
    rows = []
    for path in result_dir.glob("rus-*.json"):
        record = json.loads(path.read_text(encoding="utf-8"))
        if record.get("schema") != "n22-rus-v1":
            continue
        result = record["result"]
        shots = int(result["shots"])
        any_error = float(result["any_logical_error_rate"])
        attempt_se = _histogram_se(result["raw_attempt_histogram"])
        rows.append(
            {
                "file": path.name,
                "theta": float(result["theta"]),
                "theta_over_pi": float(result["theta"]) / math.pi,
                "shots": shots,
                "mean_parallel_levels": float(result["mean_parallel_levels"]),
                "mean_raw_preparation_attempts": float(
                    result["mean_raw_preparation_attempts"]
                ),
                "mean_raw_preparation_attempts_se": attempt_se,
                "mean_syndrome_cycles": float(result["mean_syndrome_cycles"]),
                "mean_syndrome_cycles_se": 2 * attempt_se,
                "logical_0_error": float(result["per_logical_error_rates"][0]),
                "logical_1_error": float(result["per_logical_error_rates"][1]),
                "any_logical_error": any_error,
                "any_logical_error_se": math.sqrt(any_error * (1 - any_error) / shots),
                "truncated": int(result["truncated"]),
            }
        )
    return sorted(rows, key=lambda row: float(row["theta"]))


def _write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as destination:
        writer = csv.DictWriter(
            destination, fieldnames=list(rows[0]), lineterminator="\n"
        )
        writer.writeheader()
        writer.writerows(rows)


def _write_svg(path: Path, rows: list[dict[str, object]]) -> None:
    width, height = 1000, 650
    left, right, top, bottom = 110, 100, 70, 90
    plot_w, plot_h = width - left - right, height - top - bottom
    minimum_theta = min(float(row["theta"]) for row in rows)
    maximum_theta = max(float(row["theta"]) for row in rows)

    def xx(theta: float) -> float:
        return left + math.log(theta / minimum_theta, 2) / math.log(
            maximum_theta / minimum_theta, 2
        ) * plot_w

    def yy_error(value: float) -> float:
        low, high = 1e-4, 3e-3
        return top + (math.log10(high) - math.log10(value)) / (
            math.log10(high) - math.log10(low)
        ) * plot_h

    def yy_cycles(value: float) -> float:
        low, high = 0.0, 24.0
        return top + (high - value) / (high - low) * plot_h

    error_points = " ".join(
        f"{xx(float(row['theta'])):.2f},{yy_error(float(row['any_logical_error'])):.2f}"
        for row in rows
    )
    cycle_points = " ".join(
        f"{xx(float(row['theta'])):.2f},{yy_cycles(float(row['mean_syndrome_cycles'])):.2f}"
        for row in rows
    )
    elements = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">',
        '<rect width="100%" height="100%" fill="white"/>',
        '<style>text{font-family:DejaVu Sans,Arial,sans-serif;fill:#1f2937}.axis{stroke:#374151;stroke-width:1.5}.grid{stroke:#d1d5db;stroke-width:1}.err{fill:none;stroke:#b91c1c;stroke-width:3}.cyc{fill:none;stroke:#1d4ed8;stroke-width:3;stroke-dasharray:9 6}</style>',
        '<text x="500" y="31" font-size="22" text-anchor="middle">[[22,2,6]] full two-logical RUS teleportation</text>',
        '<text x="500" y="54" font-size="15" text-anchor="middle">p=10^-3, adaptive M in {1,3}, 1,000,000 trajectories per point</text>',
    ]
    for value in (1e-4, 3e-4, 1e-3, 3e-3):
        y = yy_error(value)
        elements.extend(
            (
                f'<line class="grid" x1="{left}" y1="{y:.2f}" x2="{left + plot_w}" y2="{y:.2f}"/>',
                f'<text x="{left - 12}" y="{y + 5:.2f}" font-size="14" text-anchor="end">{value:.0e}</text>',
            )
        )
    for value in (0, 6, 12, 18, 24):
        y = yy_cycles(value)
        elements.append(
            f'<text x="{left + plot_w + 12}" y="{y + 5:.2f}" font-size="14">{value}</text>'
        )
    for row in rows:
        x = xx(float(row["theta"]))
        denominator = round(math.pi / float(row["theta"]))
        elements.extend(
            (
                f'<line class="grid" x1="{x:.2f}" y1="{top}" x2="{x:.2f}" y2="{top + plot_h}"/>',
                f'<text x="{x:.2f}" y="{top + plot_h + 27}" font-size="14" text-anchor="middle">pi/{denominator}</text>',
            )
        )
    elements.extend(
        (
            f'<line class="axis" x1="{left}" y1="{top + plot_h}" x2="{left + plot_w}" y2="{top + plot_h}"/>',
            f'<line class="axis" x1="{left}" y1="{top}" x2="{left}" y2="{top + plot_h}"/>',
            f'<line class="axis" x1="{left + plot_w}" y1="{top}" x2="{left + plot_w}" y2="{top + plot_h}"/>',
            f'<text x="{left + plot_w / 2:.2f}" y="{height - 27}" font-size="17" text-anchor="middle">Initial target angle</text>',
            f'<text x="28" y="{top + plot_h / 2:.2f}" font-size="17" text-anchor="middle" transform="rotate(-90 28 {top + plot_h / 2:.2f})">Any-logical error rate</text>',
            f'<text x="976" y="{top + plot_h / 2:.2f}" font-size="17" text-anchor="middle" transform="rotate(90 976 {top + plot_h / 2:.2f})">Expected syndrome cycles</text>',
            f'<polyline class="err" points="{error_points}"/>',
            f'<polyline class="cyc" points="{cycle_points}"/>',
        )
    )
    for row in rows:
        x = xx(float(row["theta"]))
        y = yy_error(float(row["any_logical_error"]))
        low = max(1e-4, float(row["any_logical_error"]) - float(row["any_logical_error_se"]))
        high = min(3e-3, float(row["any_logical_error"]) + float(row["any_logical_error_se"]))
        elements.extend(
            (
                f'<line x1="{x:.2f}" y1="{yy_error(low):.2f}" x2="{x:.2f}" y2="{yy_error(high):.2f}" stroke="#b91c1c" stroke-width="1.5"/>',
                f'<circle cx="{x:.2f}" cy="{y:.2f}" r="4.5" fill="#b91c1c"/>',
                f'<circle cx="{x:.2f}" cy="{yy_cycles(float(row["mean_syndrome_cycles"])):.2f}" r="4.5" fill="#1d4ed8"/>',
            )
        )
    elements.extend(
        (
            f'<line x1="{left + 20}" y1="{top + 25}" x2="{left + 65}" y2="{top + 25}" class="err"/>',
            f'<text x="{left + 75}" y="{top + 30}" font-size="15">Any-logical error (+/-1 s.e.)</text>',
            f'<line x1="{left + 20}" y1="{top + 50}" x2="{left + 65}" y2="{top + 50}" class="cyc"/>',
            f'<text x="{left + 75}" y="{top + 55}" font-size="15">Expected syndrome cycles</text>',
            "</svg>",
        )
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(elements) + "\n", encoding="utf-8")


def main() -> None:
    args = parse_args()
    calibration = _calibration_rows(args.result_dir)
    rus = _rus_rows(args.result_dir)
    if not calibration or not rus:
        raise SystemExit("calibration or RUS result rows are missing")
    _write_csv(args.calibration_csv, calibration)
    _write_csv(args.rus_csv, rus)
    _write_svg(args.svg, rus)
    print(
        f"wrote {args.calibration_csv}, {args.rus_csv}, and {args.svg}", flush=True
    )


if __name__ == "__main__":
    main()
