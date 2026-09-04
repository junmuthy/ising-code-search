#!/usr/bin/env python3
"""Audit Stage-D3 BP+OSD and optional MCMC on labeled noisy histories."""

from __future__ import annotations

import argparse
from collections import Counter, defaultdict
from datetime import datetime, timezone
import json
import math
from pathlib import Path
import time

import numpy as np

from bb64_tmr_postselection.circuit import NoiseModel

from .circuit_decoder.adaptive_decoder import D3AdaptiveDecoder
from .circuit_decoder.factor_graph import load_d3_factor_graph
from .circuit_decoder.labeled_noise import sample_labeled_trajectory
from .circuit_decoder.list_decoder import D3ListDecoder
from .model import load_hybrid_model
from .syndrome_history import build_syndrome_history_circuit


PACKAGE_DIR = Path(__file__).resolve().parent
DEFAULT_GRAPH = PACKAGE_DIR / "results" / "d3_factor_graph_p1e3.npz"
DEFAULT_OUTPUT = PACKAGE_DIR / "results" / "d3_decoder_audit_p1e3.json"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--theta", type=float, default=math.pi / 32)
    parser.add_argument("--probability", type=float, default=0.001)
    parser.add_argument("--syndrome-rounds", type=int, default=3)
    parser.add_argument("--shots", type=int, default=20)
    parser.add_argument("--seed", type=int, default=20260904)
    parser.add_argument("--graph", type=Path, default=DEFAULT_GRAPH)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--checkpoint-every", type=int, default=1)
    parser.add_argument("--bp-iterations", type=int, default=40)
    parser.add_argument("--bp-damping", type=float, default=0.45)
    parser.add_argument("--osd-order", type=int, default=2)
    parser.add_argument("--osd-window", type=int, default=32)
    parser.add_argument("--maximum-candidates", type=int, default=128)
    parser.add_argument("--posterior", action="store_true")
    parser.add_argument("--chains", type=int, default=4)
    parser.add_argument("--burn-in", type=int, default=1000)
    parser.add_argument("--retained-per-chain", type=int, default=500)
    parser.add_argument("--thin", type=int, default=5)
    parser.add_argument("--minimum-posterior-lower-bound", type=float, default=0.99)
    parser.add_argument("--minimum-effective-samples", type=float, default=100.0)
    parser.add_argument("--maximum-chain-spread", type=float, default=0.10)
    parser.add_argument("--minimum-action-transitions", type=int, default=1)
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def _atomic_json(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    temporary.replace(path)


def _mean(values: list[float]) -> float | None:
    return float(np.mean(values)) if values else None


def _aggregate(records: list[dict[str, object]]) -> dict[str, object]:
    reset_reasons = Counter(
        str(record["reset_reason"])
        for record in records
        if record.get("reset_reason") is not None
    )
    by_fault_count: dict[int, list[dict[str, object]]] = defaultdict(list)
    for record in records:
        by_fault_count[int(record["fault_count"])].append(record)
    strata = {}
    for fault_count, items in sorted(by_fault_count.items()):
        strata[str(fault_count)] = {
            "shots": len(items),
            "list_leading_action_accuracy": _mean(
                [float(item["list_action_correct"]) for item in items]
            ),
            "accepted": sum(not bool(item.get("reset", True)) for item in items),
            "accepted_action_errors": sum(
                not bool(item["posterior_action_correct"])
                for item in items
                if not bool(item.get("reset", True))
            ),
        }
    posterior_items = [record for record in records if "reset" in record]
    accepted = [record for record in posterior_items if not bool(record["reset"])]
    return {
        "shots_completed": len(records),
        "fault_count_mean": _mean([float(record["fault_count"]) for record in records]),
        "bp_converged_fraction": _mean(
            [float(record["bp_converged"]) for record in records]
        ),
        "bp_exact_hard_fraction": _mean(
            [float(int(record["bp_residual_weight"]) == 0) for record in records]
        ),
        "list_leading_action_accuracy": _mean(
            [float(record["list_action_correct"]) for record in records]
        ),
        "literal_truth_in_osd_list_fraction": _mean(
            [float(record["literal_truth_in_osd_list"]) for record in records]
        ),
        "mean_list_leading_probability": _mean(
            [float(record["list_leading_probability"]) for record in records]
        ),
        "posterior_shots": len(posterior_items),
        "accepted": len(accepted),
        "acceptance_fraction": len(accepted) / len(posterior_items) if posterior_items else None,
        "accepted_action_errors": sum(
            not bool(record["posterior_action_correct"]) for record in accepted
        ),
        "reset_reasons": dict(sorted(reset_reasons.items())),
        "by_fault_count": strata,
    }


def _payload(args: argparse.Namespace, records: list[dict[str, object]], seconds: float) -> dict[str, object]:
    return {
        "schema": "bb64-d3-decoder-audit-v1",
        "completed": len(records) == args.shots,
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "theta": args.theta,
        "physical_probability": args.probability,
        "syndrome_rounds": args.syndrome_rounds,
        "factor_graph": str(args.graph),
        "configuration": {
            "shots": args.shots,
            "seed": args.seed,
            "bp_iterations": args.bp_iterations,
            "bp_damping": args.bp_damping,
            "osd_order": args.osd_order,
            "osd_window": args.osd_window,
            "maximum_candidates": args.maximum_candidates,
            "posterior_enabled": args.posterior,
            "chains": args.chains,
            "burn_in": args.burn_in,
            "retained_per_chain": args.retained_per_chain,
            "thin": args.thin,
            "minimum_posterior_lower_bound": args.minimum_posterior_lower_bound,
            "minimum_effective_samples": args.minimum_effective_samples,
            "maximum_chain_spread": args.maximum_chain_spread,
            "minimum_action_transitions": args.minimum_action_transitions,
        },
        "interpretation": {
            "list_probability": "normalized only over the finite OSD list; it is not a calibrated posterior",
            "mcmc_probability": "detector-conditioned categorical posterior estimate; acceptance also requires cross-chain and ESS diagnostics",
            "accepted_error": "leading sampled action differs from the exact action of the labeled trajectory",
        },
        "summary": _aggregate(records),
        "records": records,
        "elapsed_seconds": seconds,
    }


def main() -> None:
    args = parse_args()
    if args.output.exists() and not args.overwrite:
        raise SystemExit(f"refusing to overwrite {args.output}")
    if args.shots <= 0 or args.checkpoint_every <= 0:
        raise SystemExit("shots and checkpoint interval must be positive")
    model = load_hybrid_model()
    history = build_syndrome_history_circuit(
        model,
        theta=args.theta,
        noise=NoiseModel(probability=args.probability),
        syndrome_rounds=args.syndrome_rounds,
    )
    print(f"checkpoint phase=load_graph status=start path={args.graph}", flush=True)
    load_started = time.monotonic()
    graph = load_d3_factor_graph(args.graph, history)
    print(
        f"checkpoint phase=load_graph status=done seconds={time.monotonic()-load_started:.3f} "+
        f"variables={graph.num_variables} components={graph.num_components}",
        flush=True,
    )
    list_decoder = D3ListDecoder(
        graph,
        model,
        theta=args.theta,
        bp_iterations=args.bp_iterations,
        bp_damping=args.bp_damping,
        osd_order=args.osd_order,
        osd_window=args.osd_window,
        maximum_candidates=args.maximum_candidates,
        minimum_action_probability=0.0,
    )
    adaptive = D3AdaptiveDecoder(
        list_decoder,
        minimum_posterior_lower_bound=args.minimum_posterior_lower_bound,
        minimum_effective_samples=args.minimum_effective_samples,
        maximum_chain_spread=args.maximum_chain_spread,
        minimum_action_transitions=args.minimum_action_transitions,
        chains=args.chains,
        burn_in=args.burn_in,
        retained_per_chain=args.retained_per_chain,
        thin=args.thin,
        seed=args.seed + 1,
    )
    rng = np.random.default_rng(args.seed)
    records: list[dict[str, object]] = []
    started = time.monotonic()
    for shot in range(1, args.shots + 1):
        trajectory = sample_labeled_trajectory(
            graph, model, list_decoder.action_builder, rng
        )
        if args.posterior:
            adaptive_result = adaptive.decode(trajectory.observed_detector_mask)
            listed = adaptive_result.list_result
        else:
            adaptive_result = None
            listed = list_decoder.decode(trajectory.observed_detector_mask)
        leading_list = listed.action_posteriors[0].action if listed.action_posteriors else None
        record: dict[str, object] = {
            "shot": shot,
            "fault_count": trajectory.fault_count,
            "branch_class_id": trajectory.branch_class_id,
            "detector_weight": trajectory.observed_detector_mask.bit_count(),
            "bp_iterations": listed.bp.iterations,
            "bp_converged": listed.bp.converged,
            "bp_residual_weight": listed.bp.residual_detector_mask.bit_count(),
            "osd_candidates": listed.candidates,
            "osd_distinct_actions": listed.distinct_actions,
            "osd_attempted_patterns": listed.osd.attempted_patterns,
            "list_leading_probability": listed.leading_list_probability,
            "list_action_correct": leading_list == trajectory.correct_action,
            "literal_truth_in_osd_list": any(
                candidate.state_codes == trajectory.state_codes
                for candidate in listed.osd.candidates
            ),
        }
        if adaptive_result is not None:
            posterior = adaptive_result.posterior
            record.update(
                {
                    "reset": adaptive_result.reset,
                    "reset_reason": adaptive_result.reset_reason,
                    "posterior_action_correct": bool(
                        posterior is not None
                        and posterior.leading_action == trajectory.correct_action
                    ),
                    "posterior_leading_probability": (
                        posterior.leading_probability if posterior is not None else None
                    ),
                    "posterior_probability_lower_bound": (
                        posterior.leading_probability_lower_bound if posterior is not None else None
                    ),
                    "posterior_effective_samples": (
                        posterior.effective_sample_size if posterior is not None else None
                    ),
                    "posterior_chain_spread": (
                        posterior.maximum_chain_fraction_spread if posterior is not None else None
                    ),
                    "posterior_distinct_actions": (
                        posterior.distinct_actions if posterior is not None else 0
                    ),
                    "posterior_action_transitions": (
                        posterior.total_action_transitions if posterior is not None else 0
                    ),
                    "chain_acceptance_fractions": (
                        [chain.acceptance_fraction for chain in posterior.chains]
                        if posterior is not None
                        else []
                    ),
                }
            )
        records.append(record)
        if shot % args.checkpoint_every == 0 or shot == args.shots:
            elapsed = time.monotonic() - started
            _atomic_json(args.output, _payload(args, records, elapsed))
            print(
                f"checkpoint phase=decode completed={shot}/{args.shots} "+
                f"elapsed_seconds={elapsed:.3f} faults={trajectory.fault_count} "+
                f"list_correct={int(bool(record['list_action_correct']))} "+
                f"reset={record.get('reset', 'disabled')}",
                flush=True,
            )
    elapsed = time.monotonic() - started
    _atomic_json(args.output, _payload(args, records, elapsed))
    print(f"checkpoint phase=complete output={args.output} seconds={elapsed:.3f}", flush=True)


if __name__ == "__main__":
    main()
