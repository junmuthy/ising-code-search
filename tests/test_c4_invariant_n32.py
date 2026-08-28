"""Tests for the direct C4-invariant n=32 geometry."""

from __future__ import annotations

import json
import pathlib
import random

import numpy as np

from c4_invariant_lagrangian.search import (
    decompose_free_orbits,
    vector_from_mask as vector_from_n28_mask,
)
from c4_invariant_n32.search import (
    NUM_QUBITS,
    TARGET_STABILIZER_RANK,
    coupled_extension_candidate,
    decompose_4442,
    embed_n28_stabilizer,
    gf2_rank,
    logical_columns,
    module_partitions,
    nilpotent_translation_power,
    rank_two_extensions,
    spectator_support,
    translate_vector,
)

PROJECT_DIR = pathlib.Path(__file__).resolve().parents[1]


def test_n32_logical_geometry() -> None:
    logicals = logical_columns()
    assert logicals.shape == (4, NUM_QUBITS)
    assert np.all(np.count_nonzero(logicals, axis=1) == 7)
    assert np.array_equal((logicals @ logicals.T) % 2, np.eye(4, dtype=np.uint8))
    assert np.array_equal(translate_vector(logicals[0]), logicals[1])
    assert spectator_support() == [7, 15, 23, 31]
    assert not np.any(logicals[:, spectator_support()])
    assert not np.any(nilpotent_translation_power(4))
    assert (4, 4, 4, 2) in module_partitions()
    assert all(sum(module_type) == 14 for module_type in module_partitions())


def test_saved_n28_seed_has_rank_two_n32_extensions() -> None:
    path = (
        PROJECT_DIR
        / "results/c4-invariant-lagrangian/orbit-replacement-all75x500-v1"
        / "best-candidates.jsonl"
    )
    saved = json.loads(path.read_text().splitlines()[0])
    old = np.asarray(
        [vector_from_n28_mask(int(mask)) for mask in saved["stabilizer_masks"]],
        dtype=np.uint8,
    )
    embedded = embed_n28_stabilizer(old)
    assert gf2_rank(embedded) == 12
    extension = next(rank_two_extensions(embedded, maximum_check_weight=12))
    assert gf2_rank(extension) == TARGET_STABILIZER_RANK
    assert not np.any((extension @ extension.T) % 2)
    assert not np.any((extension @ logical_columns().T) % 2)

    generators = decompose_free_orbits(old)
    coupled = coupled_extension_candidate(
        generators,
        omitted_orbit=0,
        rng=random.Random(0),
        maximum_check_weight=12,
        attempts_per_orbit=5000,
    )
    assert coupled is not None
    assert gf2_rank(coupled) == TARGET_STABILIZER_RANK
    assert not np.any((coupled @ coupled.T) % 2)
    assert not np.any((coupled @ logical_columns().T) % 2)
    decomposition = decompose_4442(coupled)
    assert [rank for rank, _generator in decomposition] == [4, 4, 4, 2]
