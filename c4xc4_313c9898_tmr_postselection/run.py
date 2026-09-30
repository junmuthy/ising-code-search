"""Fixed-shot, resumable paired-policy acceptance sweep (N=1 and/or N=4 only)."""

import argparse
from collections import Counter
import hashlib
from importlib.metadata import version
import json
import math
from pathlib import Path
import platform
import time

import clifft
import numpy as np

from bicycle_chain_l2_paper.tmr_postselection import circuit as engine
from bicycle_chain_l2_paper.tmr_postselection import model as algebra
from .model import load_code_artifact
from .protocol import acceptance_masks, build_circuit

ANGLES = {"zero": 0., "0p001": .001, "pi128": math.pi/128, "pi32": math.pi/32}


def atomic_json(path, payload):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(payload, indent=2, sort_keys=True, allow_nan=False) + "\n")
    tmp.replace(path)


def wilson(successes, shots):
    if not shots:
        return [0., 1.]
    z = 1.959963984540054
    p = successes/shots
    den = 1+z*z/shots
    center = (p+z*z/(2*shots))/den
    half = z*math.sqrt(p*(1-p)/shots+z*z/(4*shots*shots))/den
    return [max(0., center-half), min(1., center+half)]


def sha256_bytes(data):
    return hashlib.sha256(data).hexdigest()


def source_hashes():
    paths = sorted(Path(__file__).parent.glob("*.py"))
    result = {p.name: sha256_bytes(p.read_bytes()) for p in paths}
    for module in (engine, algebra):
        result[module.__name__] = sha256_bytes(Path(module.__file__).read_bytes())
    return result


def update_counts(counts, sample, bundle):
    d = np.asarray(sample.detectors, dtype=np.bool_)
    masks = acceptance_masks(d, bundle.detector_groups)
    counts["attempted"] += len(d)
    for policy, mask in masks.items():
        counts["accepted"][policy] += int(np.sum(mask))
    for name in ("post_X", "post_Z", "initialization_relations"):
        indices = bundle.detector_groups.get(name, ())
        hist = Counter(map(int, np.count_nonzero(d[:, list(indices)], axis=1)))
        dest = counts["syndrome_weight_histograms"].setdefault(name, {})
        for weight, count in hist.items():
            key = str(weight)
            dest[key] = dest.get(key, 0)+count
    return masks


def make_estimates(counts, bundle, num_qubits):
    shots = counts["attempted"]
    n = len(bundle.logicals)
    ops = bundle.operation_counts
    result = {}
    for policy, accepted in counts["accepted"].items():
        p = accepted/shots if shots else 0.
        result[policy] = {
            "accepted": accepted, "acceptance": p, "wilson_95": wilson(accepted, shots),
            "standard_error": math.sqrt(p*(1-p)/shots) if shots else None,
            "resources_per_attempt": n*p,
            "attempts_per_accepted_block": 1/p if p else None,
            "cnots_per_candidate_resource": ops["physical_cnots"]/(n*p) if p else None,
            "cnot_layers_per_candidate_resource": ops["cnot_layers"]/(n*p) if p else None,
            "allocated_qubit_cnot_layers_per_candidate_resource": num_qubits*ops["cnot_layers"]/(n*p) if p else None,
            "acceptance_over_ideal": p/bundle.ideal_acceptance,
        }
    return result


def run_point(path, code, configuration, *, shots, checkpoint_shots, threads, seed,
              deadline, resume=False, hashes=None):
    bundle = build_circuit(code, logical_count=configuration["logical_count"],
                           theta=configuration["theta"], probability=configuration["probability"],
                           mode=configuration["mode"], partition_count=configuration["partition_count"])
    hashes = source_hashes() if hashes is None else hashes
    config = {**configuration, "shots": shots, "checkpoint_shots": checkpoint_shots,
              "threads": threads, "seed": seed, "source_hashes": hashes,
              "circuit_sha256": sha256_bytes(bundle.text.encode()),
              "basis_sha256": code.basis_sha256, "schedule_sha256": code.schedule_sha256,
              "clifft": version("clifft"), "numpy": np.__version__, "stim": version("stim"),
              "python": platform.python_version()}
    counts = {"attempted": 0, "batches": 0, "sampling_seconds": 0.,
              "accepted": {p: 0 for p in ("tmr-x", "strict-xz", "z-only-diagnostic")},
              "syndrome_weight_histograms": {}}
    if path.exists():
        if not resume:
            raise ValueError(f"refusing to overwrite {path}")
        previous = json.loads(path.read_text())
        if previous["configuration"] != config:
            raise ValueError(f"resume configuration/source mismatch: {path}")
        if previous["completed"]:
            return previous
        counts = previous["counts"]
    circuit_path = path.with_suffix(".stim")
    if circuit_path.exists():
        if circuit_path.read_text() != bundle.text:
            raise ValueError(f"existing circuit differs: {circuit_path}")
    else:
        circuit_path.write_text(bundle.text)
    start = time.monotonic()
    program = clifft.compile(bundle.text)
    compile_seconds = time.monotonic()-start
    print(f"compile {path.stem} width={program.peak_active_width} seconds={compile_seconds:.3f}", flush=True)
    if program.peak_active_width > 16:
        raise ValueError("unexpected active width above the bounded N<=4 pilot budget")
    record = {"schema_version": 1, "configuration": config, "completed": False,
              "stopping_rule": "fixed shots; wall-time interruption is resumable, never survivor-target stopping",
              "counts": counts,
              "circuit": {"qubits": program.num_qubits, "peak_active_width": program.peak_active_width,
                          "compile_seconds": compile_seconds, "logical_indices": list(bundle.logicals),
                          "operation_counts": bundle.operation_counts,
                          "detector_groups": {k: list(v) for k, v in bundle.detector_groups.items()},
                          "partition_certificate": bundle.partition_certificate,
                          "theta_star": bundle.theta_star, "artifact": bundle.artifact},
              "ideal_acceptance": bundle.ideal_acceptance}
    while counts["attempted"] < shots and time.monotonic() < deadline:
        local = min(checkpoint_shots, shots-counts["attempted"])
        start = time.monotonic()
        sample = clifft.sample(program, shots=local, seed=seed+counts["batches"], threads=threads, batch_size="auto")
        counts["sampling_seconds"] += time.monotonic()-start
        update_counts(counts, sample, bundle)
        counts["batches"] += 1
        record["estimates"] = make_estimates(counts, bundle, program.num_qubits)
        record["completed"] = counts["attempted"] == shots
        atomic_json(path, record)
        print(f"sample {path.stem} shots={counts['attempted']}/{shots} "
              f"strict={record['estimates']['strict-xz']['acceptance']:.7f} "
              f"X={record['estimates']['tmr-x']['acceptance']:.7f}", flush=True)
    record["estimates"] = make_estimates(counts, bundle, program.num_qubits)
    record["completed"] = counts["attempted"] == shots
    atomic_json(path, record)
    return record


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--presentations", nargs="+", choices=("preferred", "original"), default=["preferred", "original"])
    parser.add_argument("--logical-counts", nargs="+", type=int, choices=(1, 4), default=[1, 4])
    parser.add_argument("--angles", nargs="+", choices=tuple(ANGLES), default=list(ANGLES))
    parser.add_argument("--mode", choices=("full", "ideal-projection", "scheduled-tmr-only"), default="full")
    parser.add_argument("--probability", type=float, default=.001)
    parser.add_argument("--partition-count", type=int, choices=(1, 3), default=3)
    parser.add_argument("--shots", type=int, default=200_000)
    parser.add_argument("--checkpoint-shots", type=int, default=10_000)
    parser.add_argument("--threads", type=int, default=4)
    parser.add_argument("--seed", type=int, default=3139898)
    parser.add_argument("--max-seconds", type=float, default=300.)
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()
    if min(args.shots, args.checkpoint_shots, args.threads, args.max_seconds) <= 0:
        parser.error("budgets must be positive")
    if not 0 <= args.probability <= 1 or (args.mode == "ideal-projection" and args.probability):
        parser.error("invalid probability for selected mode")
    for values in (args.presentations, args.logical_counts, args.angles):
        if len(values) != len(set(values)):
            parser.error("duplicate sweep values are not allowed")
    if args.output.exists() and not args.resume:
        parser.error("output directory exists; use --resume with identical configuration")
    args.output.mkdir(parents=True, exist_ok=True)
    hashes = source_hashes()
    manifest = {k: (str(v) if isinstance(v, Path) else v) for k, v in vars(args).items()
                if k not in ("output", "resume", "max_seconds")}
    manifest["source_hashes"] = hashes
    manifest_path = args.output / "manifest.json"
    if manifest_path.exists() and json.loads(manifest_path.read_text()) != manifest:
        parser.error("sweep manifest differs; choose a new output directory")
    atomic_json(manifest_path, manifest)
    deadline = time.monotonic()+args.max_seconds
    results = []
    point_index = 0
    for presentation in args.presentations:
        code = load_code_artifact(presentation)
        for n in args.logical_counts:
            for name in args.angles:
                path = args.output / f"{presentation}_N{n}_{name}.json"
                config = {"presentation": presentation, "logical_count": n, "angle_label": name,
                          "theta": ANGLES[name], "mode": args.mode, "probability": args.probability,
                          "partition_count": args.partition_count}
                if time.monotonic() >= deadline:
                    print("wall-time budget reached; checkpoints can be resumed", flush=True)
                    return
                result = run_point(path, code, config, shots=args.shots,
                                   checkpoint_shots=args.checkpoint_shots, threads=args.threads,
                                   seed=args.seed+100_000*point_index, deadline=deadline,
                                   resume=args.resume, hashes=hashes)
                results.append({"file": path.name, "completed": result["completed"]})
                atomic_json(args.output / "progress.json", {"points": results})
                point_index += 1
                if not result["completed"]:
                    print("wall-time budget reached; resume this suite to finish", flush=True)
                    return
    print(f"completed {len(results)} points", flush=True)


if __name__ == "__main__":
    main()
