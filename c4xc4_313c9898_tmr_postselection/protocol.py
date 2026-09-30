"""Certify selected injections; reuse the historical circuit primitives unchanged."""

import math

import numpy as np

from bicycle_chain_l2_paper.tmr_postselection import circuit as engine
from bicycle_chain_l2_paper.tmr_postselection.model import gf2_rank, support_to_pauli

from .model import ACTIVE

NoiseModel = engine.NoiseModel
physical_tmr_angle = engine.physical_tmr_angle
ideal_tmr_acceptance = engine.ideal_tmr_acceptance


def selected_logicals(logical_count):
    if logical_count not in (1, 4):
        raise ValueError("this milestone permits only N=1 or N=4")
    return ACTIVE[:logical_count]


def partition_certificate(code, logical_count):
    selected = selected_logicals(logical_count)
    partitions = {}
    vectors = []
    for logical in selected:
        support = code.logicals_z[logical]
        parts = tuple(support[j:j+2] for j in (0, 2, 4))
        if tuple(map(len, parts)) != (2, 2, 2):
            raise ValueError("expected a 2+2+2 decomposition")
        partitions[logical] = parts
        for part in parts:
            v = np.zeros(64, dtype=np.uint8)
            v[list(part)] = 1
            vectors.append(v)
    pieces = np.asarray(vectors, dtype=np.uint8)
    syndromes = (code.matrix_x @ pieces.T) % 2
    rank = gf2_rank(syndromes)
    for j in range(logical_count):
        if np.any(np.bitwise_xor.reduce(syndromes[:, 3*j:3*j+3], axis=1)):
            raise ValueError("full triple is not syndrome-free")
    # Independent full triples already span an N-dimensional kernel. Equality
    # proves that these are all surviving terms, with no stabilizer loophole.
    if rank != 2 * logical_count:
        raise ValueError("unintended syndrome-free partial products")
    return {
        "logicals": list(selected),
        "partitions": {str(k): [list(p) for p in v] for k, v in partitions.items()},
        "partial_syndrome_rank": rank,
        "kernel_dimension": 3 * logical_count - rank,
        "single_final_projection_certified": True,
    }


def exact_output_audit(code, logical_count, theta):
    """Enumerate all 2^(3N) physical terms, identifying ALL 16 logical bits."""
    cert = partition_certificate(code, logical_count)
    selected = selected_logicals(logical_count)
    masks = [sum(1 << q for q in part) for i in selected for part in cert["partitions"][str(i)]]
    checks = [sum(1 << q for q in row) for row in code.checks_x]
    logicals = [sum(1 << q for q in row) for row in code.logicals_x]
    phi = physical_tmr_angle(theta, 3)
    c, s = math.cos(phi / 2), math.sin(phi / 2)
    amplitudes = np.zeros(1 << 16, dtype=np.complex128)
    surviving_terms = 0
    for choice in range(1 << len(masks)):
        physical = 0
        for j, mask in enumerate(masks):
            if choice >> j & 1:
                physical ^= mask
        if any((physical & check).bit_count() % 2 for check in checks):
            continue
        surviving_terms += 1
        logical = sum(((physical & row).bit_count() % 2) << j for j, row in enumerate(logicals))
        w = choice.bit_count()
        amplitudes[logical] += c ** (len(masks) - w) * (-1j * s) ** w
    probability = float(np.vdot(amplitudes, amplitudes).real)
    expected = np.zeros_like(amplitudes)
    for choice in range(1 << logical_count):
        logical = sum(((choice >> j) & 1) << i for j, i in enumerate(selected))
        w = choice.bit_count()
        expected[logical] = math.cos(theta/2) ** (logical_count-w) * (-1j*math.sin(theta/2)) ** w
    normalized = amplitudes / math.sqrt(probability)
    overlap = np.vdot(expected, normalized)
    phase = overlap / abs(overlap)
    error = float(np.max(np.abs(normalized - phase * expected)))
    analytic = ideal_tmr_acceptance(theta, 3) ** logical_count
    if surviving_terms != 1 << logical_count or error > 1e-12 or abs(probability-analytic) > 1e-12:
        raise ValueError("exact logical output or acceptance audit failed")
    return {**cert, "theta": theta, "theta_star": phi,
            "enumerated_physical_terms": 1 << len(masks), "surviving_terms": surviving_terms,
            "acceptance": probability, "analytic_acceptance": analytic,
            "max_output_amplitude_error": error, "inactive_logicals_unchanged": True}


def build_circuit(code, logical_count=4, theta=math.pi/32, probability=0.,
                  mode="full", partition_count=3):
    selected = selected_logicals(logical_count)
    if not math.isfinite(theta):
        raise ValueError("angle must be finite")
    if mode not in ("full", "ideal-projection", "scheduled-tmr-only"):
        raise ValueError("unknown mode")
    if partition_count not in (1, 3):
        raise ValueError("only M=1 and M=3 are supported")
    noise = NoiseModel(probability=probability)
    if mode == "ideal-projection" and noise.active:
        raise ValueError("ideal projection cannot contain noise")
    certificate = partition_certificate(code, logical_count)
    partitions = {int(i): tuple(tuple(p) for p in ps) for i, ps in certificate["partitions"].items()}
    phi = physical_tmr_angle(theta, partition_count)
    builder = engine._TextBuilder()
    if mode == "full":
        engine._append_initialization(builder, code, noise)
    else:
        builder.lines.extend(str(engine.encoded_plus_circuit(code)).strip().splitlines())
        builder.tick()
    if mode == "ideal-projection":
        engine._append_direct_tmr(builder, code, selected, partitions, partition_count, phi)
        engine._append_ideal_projection(builder, code, "post")
    else:
        engine._append_ladder_tmr(builder, code, selected, partitions, partition_count, phi, noise)
        engine._append_full_syndrome_round(builder, code, noise, "post")
    groups = {k: tuple(v) for k, v in builder.detector_groups.items()}
    return engine.CircuitBundle(
        text="\n".join(builder.lines) + "\n", mode=mode, protocol="single-final-check",
        theta=float(theta), theta_star=phi, partitions_per_logical=partition_count,
        logicals=selected, batch_order=(0,), detector_groups=groups,
        postselection_detectors=groups["post_X"] + groups["post_Z"],
        measurement_groups=builder.measurement_groups, operation_counts=builder.operation_counts,
        artifact={"basis_path": str(code.basis_path), "schedule_path": str(code.schedule_path),
                  "basis_sha256": code.basis_sha256, "schedule_sha256": code.schedule_sha256},
        partition_certificate=certificate, noise=noise.to_json(),
    )


def verification_tail(bundle, code, inverse_sign=-1):
    """Noiseless diagnostic only: inverse logical rotations, then all 16 X_L."""
    if bundle.noise["probability"] != 0:
        raise ValueError("this tail is an ideal correctness test, not a noisy fidelity estimator")
    lines = [bundle.text]
    for i in bundle.logicals:
        lines.append(f"R_PAULI({inverse_sign*bundle.theta/math.pi:.17g}) {support_to_pauli('Z', code.logicals_z[i])}")
    start = sum(len(g) for g in bundle.detector_groups.values())
    for support in code.logicals_x:
        lines += [f"MPP {support_to_pauli('X', support)}", "DETECTOR rec[-1]"]
    return "\n".join(lines) + "\n", tuple(range(start, start+16))


def acceptance_masks(detectors, groups):
    x = ~np.any(detectors[:, list(groups["post_X"])], axis=1)
    z = ~np.any(detectors[:, list(groups["post_Z"])], axis=1)
    return {"tmr-x": x, "strict-xz": x & z, "z-only-diagnostic": z}
