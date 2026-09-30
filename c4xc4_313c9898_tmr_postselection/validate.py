"""Exact algebra plus sampled full-output tests for both saved schedules."""

import argparse
import math
from pathlib import Path

import clifft
import numpy as np

from .model import load_code_artifact
from .protocol import acceptance_masks, build_circuit, exact_output_audit, verification_tail
from .run import ANGLES, atomic_json, source_hashes


def validate(shots=10_000):
    exact, sampled = [], []
    for presentation in ("preferred", "original"):
        code = load_code_artifact(presentation)
        for n in (1, 4):
            for name, theta in ANGLES.items():
                exact.append({"presentation": presentation, "angle_label": name,
                              **exact_output_audit(code, n, theta)})
                for mode in ("ideal-projection", "full"):
                    bundle = build_circuit(code, n, theta, 0., mode)
                    text, logical_detectors = verification_tail(bundle, code)
                    program = clifft.compile(text)
                    sample = clifft.sample(program, shots=shots, seed=91231+len(sampled), threads=4)
                    accepted = acceptance_masks(sample.detectors, bundle.detector_groups)["strict-xz"]
                    survivors = int(np.sum(accepted))
                    errors = int(np.sum(np.any(sample.detectors[accepted][:, list(logical_detectors)], axis=1)))
                    expected = bundle.ideal_acceptance
                    se = math.sqrt(expected*(1-expected)/shots)
                    if not survivors or errors or abs(survivors/shots-expected) > 7*se+2/shots:
                        raise ValueError(f"ideal output/acceptance mismatch: {presentation}, N={n}, {name}, {mode}")
                    sampled.append({"presentation": presentation, "logical_count": n, "angle_label": name,
                                    "mode": mode, "shots": shots, "accepted": survivors,
                                    "ideal_acceptance": expected, "wrong_logical_output": errors,
                                    "peak_active_width": program.peak_active_width})
                    print(f"validated {presentation} N={n} {name} {mode}: {survivors}/{shots}, output errors={errors}", flush=True)
    return {"passed": True, "source_hashes": source_hashes(), "exact_audits": exact,
            "sampled_output_checks": sampled,
            "scope": "noiseless product-resource correctness on all sixteen logicals, not noisy fidelity"}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--shots", type=int, default=10_000)
    args = parser.parse_args()
    if args.output.exists() or args.shots <= 0:
        parser.error("output must be new and shot count positive")
    atomic_json(args.output, validate(args.shots))


if __name__ == "__main__":
    main()
