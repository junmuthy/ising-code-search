#!/usr/bin/env python3
"""Recover, validate, compress, and audit C28xC4 syndrome schedules."""
from __future__ import annotations

import argparse
import hashlib
import json
import struct
from collections import Counter
from pathlib import Path

import numpy as np
import stim

from component_code import check_space_basis, gf2_rref, logical_fibres

ROOT = Path(__file__).resolve().parent
N, R = 112, 48


def save(path, obj):
    path = Path(path)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(obj, indent=2, sort_keys=True) + "\n")
    temporary.replace(path)


def validate(layers):
    checks = {kind: np.zeros((R, N), dtype=np.uint8) for kind in ("X", "Z")}
    times, seen = {}, set()
    for time, layer in enumerate(layers):
        used = set()
        for gate in layer:
            kind, row, data = gate["type"], int(gate["check"]), int(gate["data"])
            assert kind in checks and 0 <= row < R and 0 <= data < N
            key = kind, row, data
            assert key not in seen
            seen.add(key)
            anc = N + row + (R if kind == "Z" else 0)
            assert data not in used and anc not in used
            used.update((data, anc))
            checks[kind][row, data] = 1
            times[key] = time
    canonical = gf2_rref(check_space_basis())[0]
    for matrix in checks.values():
        assert np.all(matrix.sum(axis=1) == 16)
        assert np.array_equal(gf2_rref(matrix)[0], canonical)
    assert np.array_equal(checks["X"], checks["Z"])
    h = checks["X"]
    assert not np.any(h @ h.T % 2)
    for x in range(R):
        for z in range(R):
            overlap = np.flatnonzero(checks["X"][x] & checks["Z"][z])
            assert sum(times["X", x, int(q)] < times["Z", z, int(q)] for q in overlap) % 2 == 0
    assert len(seen) == 1536
    local_order = all(
        max(times["Z", int(r), q] for r in np.flatnonzero(h[:, q]))
        < min(times["X", int(r), q] for r in np.flatnonzero(h[:, q]))
        for q in range(N)
    )
    return {
        "cnot_depth": len(layers), "cnot_count": len(seen), "dedicated_ancillas": 96,
        "check_weight": 16, "check_rank_per_type": R,
        "data_degree_histogram_per_type": dict(sorted(Counter(map(int, h.sum(axis=0))).items())),
        "layer_sizes": list(map(len, layers)), "collision_free": True,
        "same_stabilizer_space": True, "clean_cross_ancilla_backaction": True,
        "local_z_before_x": local_order,
    }, checks


def compress(layers):
    """Earliest-start schedule preserving every qubit's complete CNOT order."""
    free = [0] * (N + 2 * R)
    result, before_orders, after_orders = [], [[] for _ in free], [[] for _ in free]
    for layer in layers:
        for gate in layer:
            anc = N + gate["check"] + (R if gate["type"] == "Z" else 0)
            data = gate["data"]
            time = max(free[anc], free[data])
            free[anc] = free[data] = time + 1
            while len(result) <= time:
                result.append([])
            result[time].append(gate)
            for q in (anc, data):
                before_orders[q].append((gate["type"], gate["check"], data))
    for layer in result:
        for gate in layer:
            anc = N + gate["check"] + (R if gate["type"] == "Z" else 0)
            for q in (anc, gate["data"]):
                after_orders[q].append((gate["type"], gate["check"], gate["data"]))
    assert before_orders == after_orders
    return result


def circuit(layers, basis, noisy=True):
    _validation, checks = validate(layers)
    out = stim.Circuit()
    data, ax, az = list(range(N)), list(range(N, N+R)), list(range(N+R, N+2*R))

    def rec(indices):
        return [stim.target_rec(int(i) - out.num_measurements) for i in indices]

    out.append("RX" if basis == "X" else "R", data)
    out.append("TICK")
    previous = None
    for round_index in range(3):
        out.append("RX", ax)
        out.append("R", az)
        out.append("TICK")
        for layer in layers:
            pairs = []
            for gate in layer:
                row, q = gate["check"], gate["data"]
                pairs.extend((N + row, q) if gate["type"] == "X" else (q, N + R + row))
            out.append("CX", pairs)
            if noisy and round_index == 1:
                out.append("DEPOLARIZE2", pairs, 0.001)
            out.append("TICK")
        start = out.num_measurements
        out.append("MX", ax)
        out.append("M", az)
        out.append("TICK")
        current = {"X": list(range(start, start+R)), "Z": list(range(start+R, start+2*R))}
        if previous is None:
            for i in current[basis]:
                out.append("DETECTOR", rec([i]))
        else:
            for kind in ("X", "Z"):
                for row in range(R):
                    out.append("DETECTOR", rec([current[kind][row], previous[kind][row]]))
        previous = current
    start = out.num_measurements
    out.append("MX" if basis == "X" else "M", data)
    for row, support in enumerate(checks[basis]):
        out.append("DETECTOR", rec([previous[basis][row], *(start + int(q) for q in np.flatnonzero(support))]))
    for logical, support in enumerate(logical_fibres()):
        out.append("OBSERVABLE_INCLUDE", rec([start + int(q) for q in np.flatnonzero(support)]), logical)
    return out


def effects(c):
    dem = c.detector_error_model(decompose_errors=False, flatten_loops=True, allow_gauge_detectors=False)
    unique = set()
    for instruction in dem:
        if instruction.type != "error":
            continue
        d = o = 0
        for target in instruction.targets_copy():
            if target.is_relative_detector_id():
                d ^= 1 << target.val
            elif target.is_logical_observable_id():
                o ^= 1 << target.val
            else:
                raise AssertionError(target)
        if d or o:
            unique.add((d, o))
    return sorted(unique)


def audit(layers, basis, prefix):
    ideal = circuit(layers, basis, noisy=False)
    ideal.detector_error_model(allow_gauge_detectors=False)
    samples = ideal.compile_detector_sampler(seed=1121607).sample(128, append_observables=True)
    assert not samples.any()
    c = circuit(layers, basis)
    signatures = effects(c)
    singles = {}
    direct_witness = None
    for d, o in signatures:
        if not d and o:
            direct_witness = [(d, o)]
            break
    for d, o in signatures:
        if direct_witness:
            break
        if d in singles and singles[d] != o:
            direct_witness = [(d, singles[d]), (d, o)]
            break
        singles[d] = o
    if direct_witness:
        errors = []
        for d, o in direct_witness:
            tokens = [f"D{k}" for k in range(c.num_detectors) if d >> k & 1] + [f"L{k}" for k in range(c.num_observables) if o >> k & 1]
            explanations = c.explain_detector_error_model_errors(
                dem_filter=stim.DetectorErrorModel("error(1) " + " ".join(tokens)),
                reduce_to_one_representative_error=True,
            )
            assert len(explanations) == 1 and explanations[0].circuit_error_locations
            errors.extend(explanations)
    else:
        errors = c.search_for_undetectable_logical_errors(
            dont_explore_detection_event_sets_with_size_above=8,
            dont_explore_edges_with_degree_above=12,
            dont_explore_edges_increasing_symptom_degree=False,
            canonicalize_circuit_errors=True,
        )
    witness, total_d, total_o = [], 0, 0
    for error in errors:
        d = o = 0
        for term in error.dem_error_terms:
            target = term.dem_target
            if target.is_relative_detector_id():
                d ^= 1 << target.val
            elif target.is_logical_observable_id():
                o ^= 1 << target.val
        assert (d, o) in signatures
        total_d ^= d
        total_o ^= o
        witness.append({"detector_mask": str(d), "observable_mask": o, "explanation": str(error)})
    assert not total_d and total_o
    with prefix.with_suffix(".bin").open("wb") as out:
        # Six detector words retain compatibility with the LP160 certificate implementation.
        out.write(struct.pack("<IIII", len(signatures), c.num_detectors, c.num_observables, 6))
        for d, o in signatures:
            out.write(struct.pack("<7Q", *((d >> (64*i)) & ((1 << 64)-1) for i in range(6)), o))
    c.to_file(prefix.with_suffix(".stim"))
    result = {
        "basis": basis, "noise_model": "CNOT depolarizing faults in middle of three rounds; ideal guard rounds and boundaries",
        "ideal_detectors_and_observables_verified": True,
        "distinct_fault_effects": len(signatures), "no_faults_at_most_2": direct_witness is None,
        "fault_distance_lower_bound": len(direct_witness) if direct_witness else 3, "fault_distance_upper_bound": len(errors),
        "witness": witness, "witness_combined_detector_mask": str(total_d), "witness_combined_observable_mask": total_o,
        "effects_sha256": hashlib.sha256(prefix.with_suffix(".bin").read_bytes()).hexdigest(),
    }
    save(prefix.with_suffix(".json"), result)
    return {k: v for k, v in result.items() if k != "witness"}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--audit", action="store_true")
    parser.add_argument("--basis", choices=("X", "Z"))
    parser.add_argument("--only", choices=("recovered_depth24", "recovered_depth32_hook_candidate", "original_sequential32"))
    args = parser.parse_args()
    sources = {
        "recovered_depth24": ROOT / "recovered/c28_clean_d24.json",
        "recovered_depth32_hook_candidate": ROOT / "recovered/c28_fault_schedule_best.json",
        "original_sequential32": ROOT / "recovered/original_one_type_schedule.json",
    }
    summary_path = ROOT / "recovery_summary.json"
    summary = json.loads(summary_path.read_text()) if summary_path.exists() else {}
    for name, path in sources.items():
        if args.only and name != args.only:
            continue
        payload = json.loads(path.read_text())
        if name == "original_sequential32":
            layers = [
                [{"type": kind, "check": gate["check"], "data": gate["qubit"]} for gate in layer]
                for kind in ("Z", "X") for layer in payload["schedule"]
            ]
        else:
            layers = payload["layers"]
        validation, checks = validate(layers)
        shortened = compress(layers)
        compressed_validation, _ = validate(shortened)
        record = {"code": "[[112,16,7]]", "source": str(path.relative_to(ROOT)),
                  "validation": validation, "compressed_validation": compressed_validation,
                  "compression_preserves_every_qubit_cnot_order": True,
                  "check_supports": {k: [np.flatnonzero(row).astype(int).tolist() for row in h] for k, h in checks.items()},
                  "layers": shortened}
        save(ROOT / f"{name}.json", record)
        summary[name] = {"source_depth": len(layers), "compressed_depth": len(shortened), "verified": True}
        if args.audit:
            summary[name]["fault_audit"] = {
                basis: audit(shortened, basis, ROOT / f"{name}_{basis}")
                for basis in ([args.basis] if args.basis else ("X", "Z"))
            }
        print(json.dumps({"name": name, **summary[name]}), flush=True)
    save(ROOT / "recovery_summary.json", summary)


if __name__ == "__main__":
    main()
