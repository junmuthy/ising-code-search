#!/usr/bin/env python3
"""Compare measured noisy BB56 and BB64 acceptance as N varies."""

from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path
from typing import Any, Iterable


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_BB56_DIRS = (
    ROOT / "bicycle_chain_l4_paper" / "tmr_postselection" / "results" / "noisy_n_scaling",
    ROOT / "bicycle_chain_l4_paper" / "tmr_postselection" / "results" / "pilot_small_angles",
)
DEFAULT_BB64_DIR = ROOT / "bb64_tmr_postselection" / "results" / "noisy_n_scaling_comparison"
DEFAULT_BB64_STRICT_PI32 = (
    ROOT / "bb64_tmr_postselection" / "results" / "noisy_pi32_initial"
    / "full-single-n8-pi32-p1e-3-target400.json"
)
THETAS = (0.01, math.pi / 32)
LOGICAL_COUNTS = (1, 2, 4, 8)
POLICIES = ("tmr-x", "strict-xz")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--bb64-dir", type=Path, default=DEFAULT_BB64_DIR)
    parser.add_argument("--output-dir", type=Path, default=Path(__file__).resolve().parent)
    return parser.parse_args()


def load_record(path: Path, code: str) -> dict[str, Any]:
    with path.open(encoding="utf-8") as source:
        payload = json.load(source)
    config, estimates, counts = payload["configuration"], payload["estimates"], payload["counts"]
    return {
        "code": code,
        "policy": config.get("postselection_policy", "strict-xz"),
        "theta": float(config["theta"]),
        "N": int(config["logical_count"]),
        "M": int(config["partition_count"]),
        "p": float(config["probability"]),
        "mode": config["mode"],
        "protocol": config["protocol"],
        "attempted": int(counts["attempted"]),
        "accepted": int(counts["accepted"]),
        "acceptance": float(estimates["block_acceptance"]),
        "standard_error": float(estimates["block_acceptance_standard_error"]),
        "wilson_low": float(estimates["block_acceptance_wilson_95"][0]),
        "wilson_high": float(estimates["block_acceptance_wilson_95"][1]),
        "source": str(path.relative_to(ROOT)),
        "completed": bool(payload.get("completed")),
    }


def is_selected(record: dict[str, Any]) -> bool:
    return (
        record["completed"] and record["mode"] == "full"
        and record["protocol"] == "single-final-check" and record["M"] == 3
        and math.isclose(record["p"], 0.001, rel_tol=0, abs_tol=1e-15)
        and record["N"] in LOGICAL_COUNTS and record["policy"] in POLICIES
        and any(math.isclose(record["theta"], theta, rel_tol=0, abs_tol=1e-14) for theta in THETAS)
    )


def collect(directory: Path, code: str) -> Iterable[dict[str, Any]]:
    for path in sorted(directory.glob("*.json")):
        record = load_record(path, code)
        if is_selected(record):
            yield record


def canonical_key(record: dict[str, Any]) -> tuple[str, float, str, int]:
    theta = next(theta for theta in THETAS if math.isclose(record["theta"], theta, rel_tol=0, abs_tol=1e-14))
    return record["code"], theta, record["policy"], record["N"]


def load_all(bb64_dir: Path) -> list[dict[str, Any]]:
    records: dict[tuple[str, float, str, int], dict[str, Any]] = {}
    for directory in DEFAULT_BB56_DIRS:
        for record in collect(directory, "BB56"):
            records[canonical_key(record)] = record
    for record in collect(bb64_dir, "BB64"):
        records[canonical_key(record)] = record
    historical = load_record(DEFAULT_BB64_STRICT_PI32, "BB64")
    if is_selected(historical):
        records.setdefault(canonical_key(historical), historical)
    expected = {
        (code, theta, policy, logical_count)
        for code in ("BB56", "BB64") for theta in THETAS
        for policy in POLICIES for logical_count in LOGICAL_COUNTS
    }
    missing = sorted(expected - records.keys())
    if missing:
        raise RuntimeError(f"missing {len(missing)} measured points: {missing}")
    return [records[key] for key in sorted(records)]


def write_csv(path: Path, records: list[dict[str, Any]]) -> None:
    fields = (
        "code", "theta", "policy", "N", "M", "p", "attempted", "accepted",
        "acceptance", "standard_error", "wilson_low", "wilson_high", "source",
    )
    with path.open("w", newline="", encoding="utf-8") as destination:
        writer = csv.DictWriter(destination, fieldnames=fields, extrasaction="ignore", lineterminator="\n")
        writer.writeheader()
        writer.writerows(records)


def write_svg(path: Path, records: list[dict[str, Any]]) -> None:
    width, height = 1080, 635
    left, right, top, bottom, gap = 84, 30, 76, 135, 62
    panel_w = (width - left - right - gap) / 2
    panel_h, y_max = height - top - bottom, 0.42
    colors = {"BB56": "#2468b4", "BB64": "#d46a1f"}
    x_positions = {logical_count: index for index, logical_count in enumerate(LOGICAL_COUNTS)}

    def xmap(panel: int, logical_count: int) -> float:
        panel_left = left + panel * (panel_w + gap)
        return panel_left + 34 + x_positions[logical_count] * (panel_w - 68) / 3

    def ymap(value: float) -> float:
        return top + panel_h * (1 - value / y_max)

    parts = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">',
        '<rect width="100%" height="100%" fill="white"/>',
        '<style>text{font-family:DejaVu Sans,Arial,sans-serif;fill:#222;font-size:13px}.title{font-size:18px;font-weight:bold}.panel-title{font-size:16px;font-weight:bold}.axis{stroke:#222;stroke-width:1.3}.grid{stroke:#ddd;stroke-width:1}.err{stroke-width:1.25}</style>',
        f'<text class="title" x="{width/2}" y="27" text-anchor="middle">Noisy TMR post-selection acceptance versus simultaneous logicals</text>',
        f'<text x="{width/2}" y="49" text-anchor="middle">M=3, p=10^-3, single final syndrome check; error bars are +/-1 standard error</text>',
    ]
    for panel, theta in enumerate(THETAS):
        panel_left = left + panel * (panel_w + gap)
        panel_right = panel_left + panel_w
        title = "theta = 0.01" if theta == 0.01 else "theta = pi/32"
        parts.append(f'<text class="panel-title" x="{(panel_left+panel_right)/2}" y="{top-15}" text-anchor="middle">{title}</text>')
        for tick in (0, 0.1, 0.2, 0.3, 0.4):
            y = ymap(tick)
            parts.append(f'<line class="grid" x1="{panel_left}" y1="{y:.2f}" x2="{panel_right}" y2="{y:.2f}"/>')
            if panel == 0:
                parts.append(f'<text x="{panel_left-11}" y="{y+4:.2f}" text-anchor="end">{tick:.1f}</text>')
        parts.extend((
            f'<line class="axis" x1="{panel_left}" y1="{top+panel_h}" x2="{panel_right}" y2="{top+panel_h}"/>',
            f'<line class="axis" x1="{panel_left}" y1="{top}" x2="{panel_left}" y2="{top+panel_h}"/>',
        ))
        for logical_count in LOGICAL_COUNTS:
            x = xmap(panel, logical_count)
            parts.append(f'<text x="{x:.2f}" y="{top+panel_h+25}" text-anchor="middle">{logical_count}</text>')
        parts.append(f'<text x="{(panel_left+panel_right)/2}" y="{top+panel_h+58}" text-anchor="middle">simultaneous logical resources N (log2 spacing)</text>')
        for code in ("BB56", "BB64"):
            for policy in POLICIES:
                series = sorted(
                    (record for record in records if record["code"] == code and record["policy"] == policy and math.isclose(record["theta"], theta, rel_tol=0, abs_tol=1e-14)),
                    key=lambda record: record["N"],
                )
                color = colors[code]
                dash = "" if policy == "tmr-x" else ' stroke-dasharray="7 5"'
                coordinates = " ".join(f'{xmap(panel, record["N"]):.2f},{ymap(record["acceptance"]):.2f}' for record in series)
                parts.append(f'<polyline points="{coordinates}" fill="none" stroke="{color}" stroke-width="2.2"{dash}/>')
                for record in series:
                    x, y = xmap(panel, record["N"]), ymap(record["acceptance"])
                    y_low = ymap(record["acceptance"] + record["standard_error"])
                    y_high = ymap(record["acceptance"] - record["standard_error"])
                    parts.extend((
                        f'<line class="err" stroke="{color}" x1="{x:.2f}" y1="{y_low:.2f}" x2="{x:.2f}" y2="{y_high:.2f}"/>',
                        f'<line class="err" stroke="{color}" x1="{x-4:.2f}" y1="{y_low:.2f}" x2="{x+4:.2f}" y2="{y_low:.2f}"/>',
                        f'<line class="err" stroke="{color}" x1="{x-4:.2f}" y1="{y_high:.2f}" x2="{x+4:.2f}" y2="{y_high:.2f}"/>',
                    ))
                    tooltip = f'{code} {policy}, N={record["N"]}: {record["acceptance"]:.6f} +/- {record["standard_error"]:.6f}'
                    if policy == "tmr-x":
                        parts.append(f'<circle cx="{x:.2f}" cy="{y:.2f}" r="4.5" fill="{color}"><title>{tooltip}</title></circle>')
                    else:
                        parts.append(f'<rect x="{x-4:.2f}" y="{y-4:.2f}" width="8" height="8" fill="white" stroke="{color}" stroke-width="2"><title>{tooltip}</title></rect>')
    parts.append(f'<text transform="translate(22 {top+panel_h/2}) rotate(-90)" text-anchor="middle">accepted blocks / attempt</text>')
    legend = (("BB56", "tmr-x"), ("BB56", "strict-xz"), ("BB64", "tmr-x"), ("BB64", "strict-xz"))
    for index, (code, policy) in enumerate(legend):
        x, y = 218 + index * 182, 605
        dash = "" if policy == "tmr-x" else ' stroke-dasharray="7 5"'
        parts.append(f'<line x1="{x}" y1="{y}" x2="{x+28}" y2="{y}" stroke="{colors[code]}" stroke-width="2.2"{dash}/>')
        if policy == "tmr-x":
            parts.append(f'<circle cx="{x+14}" cy="{y}" r="4.5" fill="{colors[code]}"/>')
        else:
            parts.append(f'<rect x="{x+10}" y="{y-4}" width="8" height="8" fill="white" stroke="{colors[code]}" stroke-width="2"/>')
        parts.append(f'<text x="{x+35}" y="{y+4}">{code} {"TMR-X" if policy == "tmr-x" else "strict XZ"}</text>')
    parts.append("</svg>")
    path.write_text("\n".join(parts) + "\n", encoding="utf-8")


def fmt(value: float) -> str:
    return f"{value:.6g}"


def write_markdown(path: Path, records: list[dict[str, Any]]) -> None:
    lines = [
        "# BB56 versus BB64 acceptance as a function of N", "",
        "All points are direct noisy ClifT measurements with `M=3`, `p=10^-3`, and the single-final-check protocol. The plot does not use extrapolated points. Error bars are one binomial standard error.", "",
        "![BB56 and BB64 acceptance versus N](BB56_BB64_ACCEPTANCE_VS_N.svg)", "",
    ]
    for theta in THETAS:
        heading = "theta = 0.01" if theta == 0.01 else "theta = pi/32"
        lines.extend((
            f"## {heading}", "",
            "| `N` | `BB56` TMR-X | `BB64` TMR-X | `BB56` strict XZ | `BB64` strict XZ |",
            "|---:|---:|---:|---:|---:|",
        ))
        keyed = {(record["code"], record["policy"], record["N"]): record for record in records if math.isclose(record["theta"], theta, rel_tol=0, abs_tol=1e-14)}
        for logical_count in LOGICAL_COUNTS:
            cells = []
            for code, policy in (("BB56", "tmr-x"), ("BB64", "tmr-x"), ("BB56", "strict-xz"), ("BB64", "strict-xz")):
                record = keyed[(code, policy, logical_count)]
                cells.append(f'{fmt(record["acceptance"])} +/- {fmt(record["standard_error"])}')
            lines.append(f'| {logical_count} | ' + " | ".join(cells) + " |")
        lines.append("")
    lines.extend((
        "## Acceptance conventions", "",
        "- `TMR-X` is the paper-matching TMR-sensitive projection rule.", "",
        "- `strict XZ` additionally requires every raw Z-syndrome detector to be zero; it matches the original BB64 post-selection study.", "",
        "The companion CSV records shot counts, standard errors, Wilson intervals, and source JSON paths for every plotted point.",
    ))
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    args = parse_args()
    records = load_all(args.bb64_dir)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    write_csv(args.output_dir / "BB56_BB64_ACCEPTANCE_VS_N.csv", records)
    write_svg(args.output_dir / "BB56_BB64_ACCEPTANCE_VS_N.svg", records)
    write_markdown(args.output_dir / "BB56_BB64_ACCEPTANCE_VS_N.md", records)
    print(f"wrote comparison artifacts for {len(records)} measured points", flush=True)


if __name__ == "__main__":
    main()
