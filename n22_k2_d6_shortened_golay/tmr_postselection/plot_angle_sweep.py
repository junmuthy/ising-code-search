#!/usr/bin/env python3
"""Create CSV, Markdown, and dependency-free SVG angle-sweep summaries."""

from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("result_dir", type=Path)
    parser.add_argument("--svg", type=Path, required=True)
    parser.add_argument("--csv", type=Path, required=True)
    parser.add_argument("--markdown", type=Path, required=True)
    return parser.parse_args()


def load_rows(result_dir: Path) -> list[dict[str, float | int | str]]:
    rows = []
    for path in result_dir.glob("*.json"):
        with path.open(encoding="utf-8") as source:
            record = json.load(source)
        if not record.get("completed"):
            continue
        config = record["configuration"]
        counts = record["counts"]
        estimates = record["estimates"]
        rows.append(
            {
                "file": path.name,
                "theta": float(config["theta"]),
                "theta_star": float(record["circuit"]["theta_star"]),
                "attempted": int(counts["attempted"]),
                "accepted": int(counts["accepted"]),
                "acceptance": float(estimates["block_acceptance"]),
                "standard_error": float(estimates["block_acceptance_standard_error"]),
                "wilson_low": float(estimates["block_acceptance_wilson_95"][0]),
                "wilson_high": float(estimates["block_acceptance_wilson_95"][1]),
                "ideal_acceptance": float(estimates["ideal_tmr_acceptance"]),
                "candidate_yield": float(estimates["accepted_candidate_resources_per_attempt"]),
            }
        )
    return sorted(rows, key=lambda row: float(row["theta"]))


def write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as destination:
        writer = csv.DictWriter(destination, fieldnames=tuple(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def write_markdown(path: Path, rows: list[dict[str, object]]) -> None:
    lines = [
        "# `[[22,2,6]]` post-selection angle sweep",
        "",
        "Full two-logical circuit, `M=3`, one final check, `p=10^-3`, and one million attempts per angle.",
        "",
        "| Target angle `theta` (rad) | Physical `theta_star` (rad) | Acceptance | Standard error | Ideal TMR acceptance | Candidate yield |",
        "|---:|---:|---:|---:|---:|---:|",
    ]
    for row in reversed(rows):
        lines.append(
            "| `{theta:.9g}` | `{theta_star:.9g}` | `{acceptance:.7f}` | "
            "`{standard_error:.3g}` | `{ideal_acceptance:.7f}` | `{candidate_yield:.7f}` |".format(**row)
        )
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def write_svg(path: Path, rows: list[dict[str, object]]) -> None:
    width, height = 1000, 650
    left, right, top, bottom = 105, 35, 65, 90
    plot_w, plot_h = width - left - right, height - top - bottom

    def xx(theta: float) -> float:
        return left + (math.log10(theta) + 3) / 3 * plot_w

    def yy(rate: float) -> float:
        return top + (1 - rate) * plot_h

    measured = " ".join(
        f"{xx(float(row['theta'])):.2f},{yy(float(row['acceptance'])):.2f}"
        for row in rows
    )
    ideal = " ".join(
        f"{xx(float(row['theta'])):.2f},{yy(float(row['ideal_acceptance'])):.2f}"
        for row in rows
    )
    elements = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">',
        '<rect width="100%" height="100%" fill="white"/>',
        '<style>text{font-family:DejaVu Sans,Arial,sans-serif;fill:#1f2937}.axis{stroke:#374151;stroke-width:1.5}.grid{stroke:#d1d5db;stroke-width:1}.minor{stroke:#e5e7eb;stroke-width:1}.measured{fill:none;stroke:#1d4ed8;stroke-width:3}.ideal{fill:none;stroke:#dc2626;stroke-width:2.5;stroke-dasharray:9 7}</style>',
        '<text x="500" y="31" font-size="22" text-anchor="middle">[[22,2,6]] two-logical post-selection acceptance</text>',
        '<text x="500" y="54" font-size="15" text-anchor="middle">M=3, full circuit, p=10^-3, 1,000,000 attempts per angle</text>',
    ]
    for tick in (0, 0.2, 0.4, 0.6, 0.8, 1.0):
        y = yy(tick)
        elements.extend(
            (
                f'<line class="grid" x1="{left}" y1="{y:.2f}" x2="{left + plot_w}" y2="{y:.2f}"/>',
                f'<text x="{left - 14}" y="{y + 5:.2f}" font-size="14" text-anchor="end">{tick:.1f}</text>',
            )
        )
    for exponent in (-3, -2, -1, 0):
        x = xx(10**exponent)
        elements.extend(
            (
                f'<line class="grid" x1="{x:.2f}" y1="{top}" x2="{x:.2f}" y2="{top + plot_h}"/>',
                f'<text x="{x:.2f}" y="{top + plot_h + 27}" font-size="15" text-anchor="middle">10^{exponent}</text>',
            )
        )
        if exponent < 0:
            for multiplier in (2, 5):
                theta = multiplier * 10**exponent
                if theta <= 1:
                    xm = xx(theta)
                    elements.append(f'<line class="minor" x1="{xm:.2f}" y1="{top}" x2="{xm:.2f}" y2="{top + plot_h}"/>')
    elements.extend(
        (
            f'<line class="axis" x1="{left}" y1="{top + plot_h}" x2="{left + plot_w}" y2="{top + plot_h}"/>',
            f'<line class="axis" x1="{left}" y1="{top}" x2="{left}" y2="{top + plot_h}"/>',
            f'<text x="{left + plot_w / 2:.2f}" y="{height - 27}" font-size="17" text-anchor="middle">Target logical angle theta (radians, log scale)</text>',
            f'<text x="27" y="{top + plot_h / 2:.2f}" font-size="17" text-anchor="middle" transform="rotate(-90 27 {top + plot_h / 2:.2f})">Block acceptance probability</text>',
            f'<polyline class="ideal" points="{ideal}"/>',
            f'<polyline class="measured" points="{measured}"/>',
        )
    )
    for row in rows:
        x = xx(float(row["theta"]))
        y = yy(float(row["acceptance"]))
        low = yy(float(row["acceptance"]) - float(row["standard_error"]))
        high = yy(float(row["acceptance"]) + float(row["standard_error"]))
        elements.extend(
            (
                f'<line x1="{x:.2f}" y1="{low:.2f}" x2="{x:.2f}" y2="{high:.2f}" stroke="#1d4ed8" stroke-width="1.5"/>',
                f'<circle cx="{x:.2f}" cy="{y:.2f}" r="4.5" fill="#1d4ed8"/>',
            )
        )
    legend_x, legend_y = left + 25, top + 30
    elements.extend(
        (
            f'<line x1="{legend_x}" y1="{legend_y}" x2="{legend_x + 45}" y2="{legend_y}" class="measured"/>',
            f'<circle cx="{legend_x + 22.5}" cy="{legend_y}" r="4.5" fill="#1d4ed8"/>',
            f'<text x="{legend_x + 55}" y="{legend_y + 5}" font-size="15">Measured full-circuit acceptance (+/-1 s.e.)</text>',
            f'<line x1="{legend_x}" y1="{legend_y + 26}" x2="{legend_x + 45}" y2="{legend_y + 26}" class="ideal"/>',
            f'<text x="{legend_x + 55}" y="{legend_y + 31}" font-size="15">Ideal TMR-only acceptance</text>',
            "</svg>",
        )
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(elements) + "\n", encoding="utf-8")


def main() -> None:
    args = parse_args()
    rows = load_rows(args.result_dir)
    if not rows:
        raise SystemExit(f"no completed JSON results found under {args.result_dir}")
    write_csv(args.csv, rows)
    write_markdown(args.markdown, rows)
    write_svg(args.svg, rows)
    print(f"wrote {args.csv}, {args.markdown}, and {args.svg}", flush=True)


if __name__ == "__main__":
    main()
