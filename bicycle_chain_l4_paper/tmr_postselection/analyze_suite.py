#!/usr/bin/env python3
"""Build durable tables and a dependency-free SVG for the BB56 acceptance suite."""

from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path
from typing import Any, Iterable


PACKAGE_DIR = Path(__file__).resolve().parent
DEFAULT_RESULT_DIRS = (
    PACKAGE_DIR / "results" / "pilot_small_angles",
    PACKAGE_DIR / "results" / "noisy_trotter_n8",
    PACKAGE_DIR / "results" / "noisy_n_scaling",
    PACKAGE_DIR / "results" / "ideal_bb64_match",
)
DEFAULT_BB64 = (
    PACKAGE_DIR.parents[1]
    / "bb64_tmr_postselection"
    / "results"
    / "noisy_pi32_initial"
    / "full-single-n8-pi32-p1e-3-target400.json"
)


def load_record(path: Path, code: str = "BB56") -> dict[str, Any]:
    with path.open(encoding="utf-8") as source:
        payload = json.load(source)
    config = payload["configuration"]
    estimates = payload["estimates"]
    counts = payload["counts"]
    operations = payload["circuit"]["operation_counts"]
    return {
        "code": code,
        "file": path.name,
        "completed": bool(payload.get("completed")),
        "mode": config["mode"],
        "protocol": config["protocol"],
        "policy": config.get("postselection_policy", "strict-xz"),
        "N": int(config["logical_count"]),
        "M": int(config["partition_count"]),
        "theta": float(config["theta"]),
        "p": float(config["probability"]),
        "attempted": int(counts["attempted"]),
        "accepted": int(counts["accepted"]),
        "acceptance": float(estimates["block_acceptance"]),
        "standard_error": estimates["block_acceptance_standard_error"],
        "wilson_low": float(estimates["block_acceptance_wilson_95"][0]),
        "wilson_high": float(estimates["block_acceptance_wilson_95"][1]),
        "ideal": float(estimates["ideal_tmr_acceptance"]),
        "noise_survival": estimates["noise_survival_ratio"],
        "candidate_yield": float(estimates["accepted_candidate_resources_per_attempt"]),
        "physical_cnots": int(operations["physical_cnots"]),
        "cnot_layers": int(operations["cnot_layers"]),
        "physical_rotations": int(operations["physical_rotations"]),
        "peak_active_width": int(payload["circuit"]["peak_active_width"]),
        "shots_per_second": float(payload["timing"]["attempted_shots_per_second"]),
    }


def load_records(paths: Iterable[Path]) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    seen: set[tuple[Any, ...]] = set()
    for directory in paths:
        for path in sorted(directory.glob("*.json")):
            record = load_record(path)
            if not record["completed"]:
                continue
            key = (
                record["mode"], record["policy"], record["N"], record["M"],
                record["theta"], record["p"],
            )
            if key in seen:
                continue
            seen.add(key)
            records.append(record)
    return records


def expected_geometric_maximum(probability: float, blocks: int) -> float:
    """Expected maximum of ``blocks`` geometric variables supported on 1,2,... ."""
    if not 0 < probability <= 1 or blocks <= 0:
        return math.inf
    survival = 1 - probability
    total = 0.0
    round_index = 0
    while True:
        term = 1 - (1 - survival**round_index) ** blocks
        total += term
        if term < 1e-14:
            return total
        round_index += 1


def geometric_maximum_quantile(probability: float, blocks: int, quantile: float) -> int:
    if not 0 < probability <= 1 or blocks <= 0 or not 0 < quantile < 1:
        raise ValueError("invalid geometric-maximum parameters")
    rounds = 1
    while (1 - (1 - probability) ** rounds) ** blocks < quantile:
        rounds += 1
    return rounds


def fmt(value: Any, digits: int = 6) -> str:
    if value is None:
        return ""
    if isinstance(value, float):
        if not math.isfinite(value):
            return "inf"
        return f"{value:.{digits}g}"
    return str(value)


def write_csv(path: Path, records: list[dict[str, Any]]) -> None:
    fields = tuple(records[0]) if records else ()
    with path.open("w", newline="", encoding="utf-8") as destination:
        writer = csv.DictWriter(destination, fieldnames=fields, lineterminator="\n")
        if fields:
            writer.writeheader()
            writer.writerows(records)


def table(headers: tuple[str, ...], rows: Iterable[tuple[Any, ...]]) -> str:
    lines = [
        "| " + " | ".join(headers) + " |",
        "|" + "|".join("---" for _ in headers) + "|",
    ]
    lines.extend("| " + " | ".join(fmt(value) for value in row) + " |" for row in rows)
    return "\n".join(lines)


def write_svg(path: Path, records: list[dict[str, Any]], bb64: dict[str, Any] | None) -> None:
    selected = sorted(
        (
            record for record in records
            if record["mode"] == "full" and record["N"] == 8 and record["theta"] > 0
        ),
        key=lambda record: (record["policy"], record["theta"]),
    )
    if not selected:
        return
    width, height = 820, 520
    left, right, top, bottom = 88, 28, 35, 72
    plot_w, plot_h = width - left - right, height - top - bottom
    min_x, max_x = -3.15, -0.85
    min_y, max_y = -2.4, -0.25

    def xmap(theta: float) -> float:
        return left + (math.log10(theta) - min_x) / (max_x - min_x) * plot_w

    def ymap(acceptance: float) -> float:
        return top + (max_y - math.log10(acceptance)) / (max_y - min_y) * plot_h

    colors = {"tmr-x": "#2468b4", "strict-xz": "#c44536"}
    parts = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">',
        '<rect width="100%" height="100%" fill="white"/>',
        '<style>text{font-family:DejaVu Sans,Arial,sans-serif;font-size:13px}.title{font-size:17px;font-weight:bold}.axis{stroke:#222;stroke-width:1.3}.grid{stroke:#ddd;stroke-width:1}</style>',
        f'<text class="title" x="{width/2}" y="22" text-anchor="middle">BB56 N=8 post-selection acceptance</text>',
    ]
    for exponent in (-3, -2, -1):
        x = xmap(10**exponent)
        parts.extend((
            f'<line class="grid" x1="{x}" y1="{top}" x2="{x}" y2="{top+plot_h}"/>',
            f'<text x="{x}" y="{top+plot_h+24}" text-anchor="middle">10^{exponent}</text>',
        ))
    for exponent in (-2, -1):
        y = ymap(10**exponent)
        parts.extend((
            f'<line class="grid" x1="{left}" y1="{y}" x2="{left+plot_w}" y2="{y}"/>',
            f'<text x="{left-12}" y="{y+5}" text-anchor="end">10^{exponent}</text>',
        ))
    parts.extend((
        f'<line class="axis" x1="{left}" y1="{top+plot_h}" x2="{left+plot_w}" y2="{top+plot_h}"/>',
        f'<line class="axis" x1="{left}" y1="{top}" x2="{left}" y2="{top+plot_h}"/>',
        f'<text x="{left+plot_w/2}" y="{height-20}" text-anchor="middle">target angle theta (radians)</text>',
        f'<text transform="translate(22 {top+plot_h/2}) rotate(-90)" text-anchor="middle">accepted blocks / attempt</text>',
    ))
    for policy in ("tmr-x", "strict-xz"):
        points = [record for record in selected if record["policy"] == policy]
        coordinates = " ".join(
            f"{xmap(record['theta']):.2f},{ymap(record['acceptance']):.2f}"
            for record in points
        )
        if coordinates:
            color = colors[policy]
            parts.append(f'<polyline points="{coordinates}" fill="none" stroke="{color}" stroke-width="2"/>')
            for record in points:
                parts.append(
                    f'<circle cx="{xmap(record["theta"]):.2f}" cy="{ymap(record["acceptance"]):.2f}" r="4" fill="{color}"/>'
                )
    if bb64 is not None:
        strict_points = [record for record in selected if record["policy"] == "strict-xz"]
        extrapolated = " ".join(
            f"{xmap(record['theta']):.2f},{ymap(bb64['noise_survival'] * record['ideal']):.2f}"
            for record in strict_points
        )
        if extrapolated:
            parts.append(
                f'<polyline points="{extrapolated}" fill="none" stroke="#555" stroke-width="1.5" stroke-dasharray="6 5"/>'
            )
        parts.append(
            f'<rect x="{xmap(bb64["theta"])-4:.2f}" y="{ymap(bb64["acceptance"])-4:.2f}" width="8" height="8" fill="#222"/>'
        )
    legend_x, legend_y = left + 18, top + 20
    for offset, (label, color) in enumerate((
        ("BB56 paper TMR-X", colors["tmr-x"]),
        ("BB56 strict XZ", colors["strict-xz"]),
        ("BB64 strict XZ (fit; square measured)", "#222"),
    )):
        y = legend_y + 22 * offset
        parts.append(f'<rect x="{legend_x}" y="{y-8}" width="12" height="12" fill="{color}"/>')
        parts.append(f'<text x="{legend_x+20}" y="{y+2}">{label}</text>')
    parts.append("</svg>")
    path.write_text("\n".join(parts) + "\n", encoding="utf-8")


def write_markdown(
    path: Path, records: list[dict[str, Any]], bb64: dict[str, Any] | None
) -> None:
    noisy_n8 = sorted(
        (
            record for record in records
            if record["mode"] == "full" and record["N"] == 8
        ),
        key=lambda record: (record["theta"], record["policy"]),
    )
    scaling = sorted(
        (
            record for record in records
            if record["mode"] == "full"
            and record["theta"] in (0.01, math.pi / 32)
        ),
        key=lambda record: (record["theta"], record["policy"], record["N"]),
    )
    paired_policy_rows = []
    by_theta = {
        theta: {
            record["policy"]: record
            for record in noisy_n8
            if record["theta"] == theta
        }
        for theta in sorted({record["theta"] for record in noisy_n8})
    }
    for theta, pair in by_theta.items():
        if "tmr-x" not in pair or "strict-xz" not in pair:
            continue
        paper = pair["tmr-x"]["acceptance"]
        strict = pair["strict-xz"]["acceptance"]
        paired_policy_rows.append((theta, paper, strict, strict / paper if paper else None))
    ideal_records = sorted(
        (record for record in records if record["mode"] == "ideal-projection"),
        key=lambda record: (record["theta"], record["N"]),
    )
    strict_by_theta = {
        record["theta"]: record
        for record in noisy_n8
        if record["policy"] == "strict-xz"
    }
    paper_by_theta = {
        record["theta"]: record
        for record in noisy_n8
        if record["policy"] == "tmr-x"
    }
    lines = [
        "# BB56 post-selection suite",
        "",
        "All noisy runs use the frozen `[[56,8,6]]` presentation, the published depth-eight schedule, `M=3`, and `p=10^-3`. This report concerns candidate-state acceptance only; it does not include teleportation or conditional logical infidelity.",
        "",
        "## Key findings",
        "",
        f"- At `theta=10^-3`, `N=8` acceptance is `{paper_by_theta[0.001]['acceptance']:.5f}` under the paper policy and `{strict_by_theta[0.001]['acceptance']:.5f}` under strict XZ.",
        f"- At `theta=pi/32`, those acceptances fall to `{paper_by_theta[math.pi / 32]['acceptance']:.5f}` and `{strict_by_theta[math.pi / 32]['acceptance']:.5f}`.",
        f"- Sixteen independently retried strict-XZ blocks have expected slowest completion `{expected_geometric_maximum(strict_by_theta[0.001]['acceptance'], 16):.2f}` attempts at `theta=10^-3`, `{expected_geometric_maximum(strict_by_theta[0.01]['acceptance'], 16):.2f}` at `theta=10^-2`, and `{expected_geometric_maximum(strict_by_theta[math.pi / 32]['acceptance'], 16):.2f}` at `theta=pi/32`.",
        "- The measured noise/ideal ratio is approximately angle-independent within each acceptance convention, validating the factorized acceptance model over this sweep.",
        "",
        "## Circuit scaling",
        "",
        table(
            ("N", "physical CNOTs", "CNOT layers", "physical rotations", "ClifT active width"),
            (
                (
                    record["N"], record["physical_cnots"], record["cnot_layers"],
                    record["physical_rotations"], record["peak_active_width"],
                )
                for record in sorted(
                    {
                        record["N"]: record
                        for record in records
                        if record["mode"] == "full" and record["theta"] > 0
                    }.values(),
                    key=lambda record: record["N"],
                )
            ),
        ),
        "",
        "## Ideal-projection validation",
        "",
        table(
            ("theta", "N", "attempts", "measured +/- s.e.", "exact", "difference/s.e."),
            (
                (
                    record["theta"], record["N"], record["attempted"],
                    f"{record['acceptance']:.6g} +/- {record['standard_error']:.3g}",
                    record["ideal"],
                    (record["acceptance"] - record["ideal"]) / record["standard_error"],
                )
                for record in ideal_records
            ),
        ),
        "",
        "## N=8 angle sweep",
        "",
        table(
            ("theta", "policy", "attempts", "acceptance +/- s.e.", "ideal", "noise/ideal", "yield 8s", "1/s", "E[max 16]"),
            (
                (
                    record["theta"], f"`{record['policy']}`", record["attempted"],
                    f"{record['acceptance']:.6g} +/- {record['standard_error']:.3g}",
                    record["ideal"], record["noise_survival"], record["candidate_yield"],
                    1 / record["acceptance"],
                    expected_geometric_maximum(record["acceptance"], 16),
                )
                for record in noisy_n8
            ),
        ),
        "",
        "![BB56 acceptance angle sweep](BB56_POSTSELECTION_ANGLE_SWEEP.svg)",
        "",
        "## Acceptance-policy comparison",
        "",
        "The final column estimates the probability that all raw Z checks pass conditional on the paper-matching X projection passing.",
        "",
        table(
            ("theta", "TMR-X", "strict XZ", "strict/TMR-X"),
            paired_policy_rows,
        ),
        "",
        "## Number of simultaneous logicals",
        "",
        table(
            ("theta", "policy", "N", "attempts", "acceptance +/- s.e.", "ideal", "yield Ns"),
            (
                (
                    record["theta"], f"`{record['policy']}`", record["N"], record["attempted"],
                    f"{record['acceptance']:.6g} +/- {record['standard_error']:.3g}",
                    record["ideal"], record["candidate_yield"],
                )
                for record in scaling
            ),
        ),
        "",
        "## Direct BB64 comparison",
        "",
    ]
    if bb64 is None:
        lines.append("The frozen BB64 comparison result was not available.")
    else:
        match = [
            record for record in noisy_n8
            if record["policy"] == "strict-xz"
            and math.isclose(record["theta"], bb64["theta"], rel_tol=0, abs_tol=1e-14)
        ]
        rows = []
        if match:
            rows.append(("`BB56`", match[0]["attempted"], match[0]["acceptance"], match[0]["standard_error"], match[0]["candidate_yield"]))
        rows.append(("`BB64`", bb64["attempted"], bb64["acceptance"], bb64["standard_error"], bb64["candidate_yield"], bb64["physical_cnots"], bb64["cnot_layers"]))
        if match:
            rows[0] = (*rows[0], match[0]["physical_cnots"], match[0]["cnot_layers"])
        lines.append(table(("code", "attempts", "strict-XZ acceptance", "s.e.", "yield 8s", "physical CNOTs", "CNOT layers"), rows))
        lines.extend((
            "",
            "For context only, the following BB64 values extrapolate its measured `pi/32` noise-survival factor across angle. They are not additional BB64 simulations.",
            "",
            table(
                ("theta", "BB56 strict measured", "BB64 strict extrapolated", "BB56/BB64"),
                (
                    (
                        record["theta"], record["acceptance"],
                        bb64["noise_survival"] * record["ideal"],
                        record["acceptance"] / (bb64["noise_survival"] * record["ideal"]),
                    )
                    for record in noisy_n8
                    if record["policy"] == "strict-xz" and record["theta"] > 0
                ),
            ),
        ))
    lines.extend((
        "",
        "The `tmr-x` policy is retained to reproduce the paper's fitted acceptance convention. The `strict-xz` policy is the matched comparison with the original BB64 post-selection experiment.",
        "",
        "Raw checkpoint JSON files remain ignored; this report, its CSV table, the SVG, manifests, and source are version controlled.",
    ))
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--result-dir", type=Path, action="append")
    parser.add_argument("--bb64-result", type=Path, default=DEFAULT_BB64)
    parser.add_argument("--output-dir", type=Path, default=PACKAGE_DIR)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    records = load_records(args.result_dir or DEFAULT_RESULT_DIRS)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    bb64 = load_record(args.bb64_result, code="BB64") if args.bb64_result.exists() else None
    write_csv(args.output_dir / "BB56_POSTSELECTION_SUITE.csv", records)
    write_svg(args.output_dir / "BB56_POSTSELECTION_ANGLE_SWEEP.svg", records, bb64)
    write_markdown(args.output_dir / "BB56_POSTSELECTION_SUITE.md", records, bb64)
    print(f"wrote suite outputs for {len(records)} distinct result points", flush=True)


if __name__ == "__main__":
    main()
