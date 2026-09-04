"""Compact, reproducible NPZ storage for scheduled fault catalogs."""

from __future__ import annotations

import hashlib
from pathlib import Path

import numpy as np

from bb64_hybrid_nonclifford.syndrome_history import SyndromeHistoryCircuit

from .catalog import FaultCatalog, PairSignatureGroup, SignatureGroup
from .frames import BoundaryFrame
from .ledger import build_fault_ledger
from .propagate import PauliSignature, parse_circuit


def _words(value: int, count: int) -> tuple[int, ...]:
    return tuple((value >> (64 * index)) & ((1 << 64) - 1) for index in range(count))


def _integer(words: np.ndarray) -> int:
    return sum(int(value) << (64 * index) for index, value in enumerate(words))


def _signature_arrays(groups) -> dict[str, np.ndarray]:
    return {
        "detector": np.asarray([_words(group.signature.detector_mask, 4) for group in groups], dtype=np.uint64),
        "final_x": np.asarray([group.signature.final_x_mask for group in groups], dtype=np.uint64),
        "final_z": np.asarray([group.signature.final_z_mask for group in groups], dtype=np.uint64),
        "rotation": np.asarray([group.signature.rotation_sign_mask for group in groups], dtype=np.uint32),
        "frame_physical_x": np.asarray([group.frame.physical_x_correction for group in groups], dtype=np.uint64),
        "frame_physical_z": np.asarray([group.frame.physical_z_correction for group in groups], dtype=np.uint64),
        "frame_logical_x": np.asarray([group.frame.logical_x_frame for group in groups], dtype=np.uint8),
        "frame_logical_z": np.asarray([group.frame.logical_z_frame for group in groups], dtype=np.uint8),
        "relative_weight": np.asarray([group.relative_weight for group in groups], dtype=np.float64),
    }


def save_fault_catalog(path: Path, catalog: FaultCatalog) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    single = _signature_arrays(catalog.single_groups)
    pair = _signature_arrays(catalog.pair_groups)
    offsets = [0]
    location_ids: list[int] = []
    location_weights: list[float] = []
    for group in catalog.single_groups:
        for location, weight in group.location_weights:
            location_ids.append(location)
            location_weights.append(weight)
        offsets.append(len(location_ids))
    payload: dict[str, np.ndarray] = {
        "schema": np.asarray("bb64-scheduled-fault-catalog-v1"),
        "circuit_sha256": np.asarray(hashlib.sha256(catalog.history.text.encode()).hexdigest()),
        "single_mechanism_count": np.asarray([group.mechanism_count for group in catalog.single_groups], dtype=np.uint32),
        "single_minimum_mechanism_id": np.asarray([group.minimum_mechanism_id for group in catalog.single_groups], dtype=np.int32),
        "single_location_offsets": np.asarray(offsets, dtype=np.uint32),
        "single_location_ids": np.asarray(location_ids, dtype=np.uint32),
        "single_location_weights": np.asarray(location_weights, dtype=np.float64),
        "pair_source_group_pairs": np.asarray([group.source_group_pairs for group in catalog.pair_groups], dtype=np.uint32),
    }
    payload.update({f"single_{key}": value for key, value in single.items()})
    payload.update({f"pair_{key}": value for key, value in pair.items()})
    temporary = path.with_name(path.name + ".tmp.npz")
    np.savez_compressed(temporary, **payload)
    temporary.replace(path)


def _load_signature(arrays: dict[str, np.ndarray], prefix: str, index: int) -> tuple[PauliSignature, BoundaryFrame, float]:
    signature = PauliSignature(
        _integer(arrays[f"{prefix}_detector"][index]),
        int(arrays[f"{prefix}_final_x"][index]),
        int(arrays[f"{prefix}_final_z"][index]),
        int(arrays[f"{prefix}_rotation"][index]),
    )
    frame = BoundaryFrame(
        int(arrays[f"{prefix}_frame_physical_x"][index]),
        int(arrays[f"{prefix}_frame_physical_z"][index]),
        int(arrays[f"{prefix}_frame_logical_x"][index]),
        int(arrays[f"{prefix}_frame_logical_z"][index]),
    )
    return signature, frame, float(arrays[f"{prefix}_relative_weight"][index])


def load_fault_catalog(path: Path, history: SyndromeHistoryCircuit) -> FaultCatalog:
    archive = np.load(Path(path), allow_pickle=False)
    if str(archive["schema"]) != "bb64-scheduled-fault-catalog-v1":
        raise ValueError("unknown scheduled fault catalog schema")
    expected_hash = hashlib.sha256(history.text.encode()).hexdigest()
    if str(archive["circuit_sha256"]) != expected_hash:
        raise ValueError("fault catalog was generated from a different circuit")
    ledger = build_fault_ledger(history.text)
    parsed = parse_circuit(history.text)
    zero_frame = BoundaryFrame(0, 0, 0, 0)
    no_fault = SignatureGroup(PauliSignature(0, 0, 0, 0), zero_frame, 1.0, 0, (), -1)
    # ``NpzFile`` decompresses on every keyed read.  Materialize each member
    # once before the reconstruction loops.
    arrays = {name: archive[name] for name in archive.files}
    offsets = arrays["single_location_offsets"]
    ids = arrays["single_location_ids"]
    weights = arrays["single_location_weights"]
    single_counts = arrays["single_mechanism_count"]
    single_minimum = arrays["single_minimum_mechanism_id"]
    singles = []
    for index in range(len(arrays["single_relative_weight"])):
        signature, frame, relative_weight = _load_signature(arrays, "single", index)
        start, stop = int(offsets[index]), int(offsets[index + 1])
        singles.append(
            SignatureGroup(
                signature,
                frame,
                relative_weight,
                int(single_counts[index]),
                tuple((int(ids[item]), float(weights[item])) for item in range(start, stop)),
                int(single_minimum[index]),
            )
        )
    pairs = []
    for index in range(len(arrays["pair_relative_weight"])):
        signature, frame, relative_weight = _load_signature(arrays, "pair", index)
        pairs.append(
            PairSignatureGroup(
                signature,
                frame,
                relative_weight,
                int(arrays["pair_source_group_pairs"][index]),
            )
        )
    return FaultCatalog(
        parsed,
        history,
        ledger,
        no_fault,
        tuple(singles),
        tuple(pairs),
        float(np.sum(arrays["pair_relative_weight"])),
    )
