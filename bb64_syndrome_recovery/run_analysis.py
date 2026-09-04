#!/usr/bin/env python3
"""Run the complete ideal BB64 syndrome-recovery analysis."""

from __future__ import annotations

import argparse
import csv
from datetime import datetime, timezone
import json
import math
from pathlib import Path
import sys
from typing import Any

import numpy as np

from .algebra import (
    NUM_SYNDROME_CLASSES,
    add_angle_columns,
    angle_table,
    build_recovery_model,
    verify_factorization,
)
from .controller import compare_policies, monte_carlo_policy


PACKAGE_DIR = Path(__file__).resolve().parent
DEFAULT_OUTPUT = PACKAGE_DIR / "results" / "ideal_recovery_pi32"
THRESHOLDS = (0, 1, 2, 3, 8)
ANGLES = (math.pi / 128, math.pi / 64, math.pi / 32, math.pi / 16, math.pi / 8)


def _jsonable(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(key): _jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(item) for item in value]
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, Path):
        return str(value)
    return value


def _atomic_json(path: Path, payload: Any) -> None:
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(json.dumps(_jsonable(payload), indent=2) + "\n", encoding="utf-8")
    temporary.replace(path)


def _atomic_npz(path: Path, **arrays: np.ndarray) -> None:
    temporary = path.with_name(path.name + ".tmp")
    with temporary.open("wb") as destination:
        np.savez_compressed(destination, **arrays)
    temporary.replace(path)


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    temporary = path.with_name(path.name + ".tmp")
    fields: list[str] = []
    for row in rows:
        for key in row:
            if key not in fields:
                fields.append(key)
    with temporary.open("w", newline="", encoding="utf-8") as destination:
        writer = csv.DictWriter(destination, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)
    temporary.replace(path)


def _fmt(value: float) -> str:
    if abs(value) >= 100:
        return f"{value:.3f}"
    return f"{value:.6g}"


def _write_markdown(
    path: Path,
    *,
    certificate: dict[str, Any],
    records: list[dict[str, Any]],
    shots: int,
) -> None:
    pi32 = [row for row in records if row["angle_label"] == "pi/32"]
    table = certificate["local_branch_data"]
    lines = [
        "# BB64 ideal syndrome-conditioned TMR recovery",
        "",
        "This report studies the frozen preferred `[[64,8,8]]` basis and its saved `M=3` partitions. It is a noiseless branch-algebra and controller analysis; it does not claim circuit-level fault tolerance for the adaptive repair circuit.",
        "",
        "## Algebraic certificate",
        "",
        f"- Partial-product generators: `24` (`3` per logical qubit).",
        f"- Displayed X-check outcomes: `32`; attainable syndrome rank: `{certificate['syndrome_rank_gf2']}`.",
        f"- Enumerated syndrome classes: `{certificate['num_syndrome_classes']:,}`.",
        f"- Kernel dimension: `{certificate['kernel_dimension']}`; it is generated exactly by the eight full logical-`Z` triples.",
        f"- Exhaustively compared amplitudes: `{certificate['num_amplitudes_checked']:,}`.",
        f"- Maximum factorization error: `{certificate['max_factorization_amplitude_error']:.3e}`.",
        f"- Independent product rotations after canonical correction: `{certificate['corrected_branches_are_product_rotations']}`.",
        "",
        "Each syndrome has one canonical correction containing zero or one saved TMR piece per logical qubit. A nonzero local branch selects one of three pieces, so the syndrome labels are exactly the `4^8 = 65,536` strings over `{0,1,2,3}`. The correction table and all 64-bit physical supports are stored in `syndrome_classes.npz`.",
        "",
        "## Local branches at theta = pi/32",
        "",
        "| Branch | Probability per pattern | Multiplicity | Logical angle | Angle-frame Z |",
        "|---|---:|---:|---:|---:|",
        f"| target | {_fmt(float(table['probability_target']))} | 1 | {_fmt(float(table['target']['logical_angle']))} | `{table['target']['logical_z_frame']}` |",
        f"| alternative piece | {_fmt(float(table['alternative']['probability_per_pattern']))} | 3 | {_fmt(float(table['alternative']['logical_angle']))} | `{table['alternative']['logical_z_frame']}` |",
        "",
        "A retained alternative is repaired deterministically with an in-code `M=1` logical `R_Z(theta - alpha)` rotation. All eight logical resource states are correct at output; the threshold only decides whether an expensive branch is reset or repaired.",
        "",
        "## Policy comparison at theta = pi/32",
        "",
        "| Threshold | Keep/attempt | Reset/attempt | Initial attempts | Repair given kept | Bad logicals given kept | TMR stages | CNOT layers |",
        "|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in pi32:
        lines.append(
            "| {threshold} | {keep} | {reset} | {attempts} | {repair} | {bad} | {stages} | {layers} |".format(
                threshold=row["threshold"],
                keep=_fmt(row["keep_probability_per_attempt"]),
                reset=_fmt(row["reset_probability_per_attempt"]),
                attempts=_fmt(row["expected_initial_attempts"]),
                repair=_fmt(row["correction_probability_given_kept"]),
                bad=_fmt(row["expected_bad_logicals_given_kept"]),
                stages=_fmt(row["expected_total_tmr_stages"]),
                layers=_fmt(row["expected_cnot_layers"]),
            )
        )
    lines.extend(
        [
            "",
            f"The analytic moments were independently cross-checked with `{shots:,}` Bernoulli samples per angle and threshold. Standard errors are in `policy_comparison.csv` and the full records are in `policy_comparison.json`.",
            "",
            "Resource accounting uses the saved BB64 schedule. One initial full `M=3` attempt has `848` CNOTs, `24` CNOT layers, `24` physical rotations, and two syndrome cycles including initialization. A retained repair adds `14` CNOTs per bad logical, `14` CNOT layers per nonempty disjoint-support batch, and conservatively one final 512-CNOT/eight-layer syndrome round.",
            "",
            "## Scope",
            "",
            "This establishes ideal branch recoverability and compares first-stage reset policies. The next circuit-level phase must synthesize feed-forward, propagate physical noise through correction, and re-evaluate acceptance and logical error. No noisy recovery claim is made here.",
            "",
        ]
    )
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text("\n".join(lines), encoding="utf-8")
    temporary.replace(path)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--shots", type=int, default=1_000_000)
    parser.add_argument("--seed", type=int, default=20260904)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--skip-exhaustive", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    output = args.output.resolve()
    if output.exists() and any(output.iterdir()) and not args.overwrite:
        raise SystemExit(f"refusing to overwrite nonempty output directory: {output}")
    output.mkdir(parents=True, exist_ok=True)

    print(f"[1/6] Loading frozen BB64 artifacts and enumerating {NUM_SYNDROME_CLASSES:,} syndromes", flush=True)
    model = build_recovery_model()
    print("[2/6] Constructed canonical partial-product corrections and physical supports", flush=True)
    pi32_columns = add_angle_columns(model, math.pi / 32)
    _atomic_npz(
        output / "syndrome_classes.npz",
        syndrome_id=model.syndrome_ids,
        syndrome_coordinates=model.syndrome_coordinates,
        displayed_syndromes=model.displayed_syndromes,
        initial_piece_masks=model.initial_piece_masks,
        canonical_piece_masks=model.canonical_piece_masks,
        correction_piece_masks=model.correction_piece_masks,
        correction_physical_supports=model.correction_physical_supports,
        correction_weight=model.correction_weight,
        logical_z_frame=model.logical_z_frame,
        alternative_mask=model.alternative_mask,
        alternative_count=model.alternative_count,
        selected_piece=model.selected_piece,
        local_branch_labels=model.local_branch_labels,
        base4_branch_id=model.base4_branch_id,
        displayed_syndrome_weight=model.displayed_syndrome_weight,
        logical_angles_pi32=pi32_columns["logical_angles"],
        angle_z_frames_pi32=pi32_columns["angle_z_frames"],
        piece_supports=model.piece_supports,
        syndrome_matrix=model.syndrome_matrix,
        triple_generators=model.triple_generators,
        independent_piece_columns=np.asarray(model.independent_piece_columns, dtype=np.uint8),
        independent_syndrome_rows=np.asarray(model.independent_syndrome_rows, dtype=np.uint8),
    )

    print("[3/6] Exhaustively verifying product factorization (checkpoint every 2,048 syndromes)", flush=True)
    certificate = verify_factorization(
        model,
        math.pi / 32,
        exhaustive=not args.skip_exhaustive,
        batch_size=2048,
        progress=lambda message: print(f"       {message}", flush=True),
    )
    certificate.update(
        {
            "created_utc": datetime.now(timezone.utc).isoformat(),
            "basis_path": str(model.code.basis_path),
            "basis_sha256": model.code.basis_sha256,
            "schedule_path": str(model.code.schedule_path),
            "schedule_sha256": model.code.schedule_sha256,
            "independent_piece_columns": list(model.independent_piece_columns),
            "independent_syndrome_rows": list(model.independent_syndrome_rows),
            "canonical_correction_max_weight": int(np.max(model.correction_weight)),
            "canonical_correction_mean_weight": float(np.mean(model.correction_weight)),
        }
    )
    _atomic_json(output / "algebraic_certificate.json", certificate)
    if not certificate["corrected_branches_are_product_rotations"]:
        raise RuntimeError("factorization certificate failed")

    print("[4/6] Evaluating exact reset/repair policies", flush=True)
    records: list[dict[str, Any]] = []
    full_records: list[dict[str, Any]] = []
    for angle_index, theta in enumerate(ANGLES):
        angle_label = f"pi/{round(math.pi / theta)}"
        for threshold_index, analytic in enumerate(compare_policies(theta, THRESHOLDS)):
            seed = args.seed + 100 * angle_index + threshold_index
            monte_carlo = monte_carlo_policy(theta, analytic.threshold, shots=args.shots, seed=seed)
            full_records.append(
                {
                    "angle_label": angle_label,
                    "angle_table": angle_table(theta),
                    "analytic": analytic.to_json(),
                    "monte_carlo": monte_carlo,
                }
            )
            row = {"angle_label": angle_label, **analytic.to_json()}
            row.update({f"mc_{key}": value for key, value in monte_carlo.items()})
            records.append(row)
        print(f"       {angle_label}: {angle_index + 1}/{len(ANGLES)} angle points complete", flush=True)

    print(f"[5/6] Writing Monte Carlo cross-checks ({args.shots:,} shots per policy)", flush=True)
    _atomic_json(
        output / "policy_comparison.json",
        {
            "created_utc": datetime.now(timezone.utc).isoformat(),
            "shots_per_policy": args.shots,
            "thresholds": list(THRESHOLDS),
            "records": full_records,
        },
    )
    _write_csv(output / "policy_comparison.csv", records)
    _write_markdown(output / "RESULTS.md", certificate=certificate, records=records, shots=args.shots)

    print(f"[6/6] Complete: {output}", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
