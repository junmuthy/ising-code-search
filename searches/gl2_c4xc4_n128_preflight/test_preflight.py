"""Focused algebra checks for the n=128 preflight implementation."""

from __future__ import annotations

import importlib.util
import pathlib
import random
import sys

import numpy as np


MODULE_PATH = pathlib.Path(__file__).with_name("preflight.py")
SPEC = importlib.util.spec_from_file_location("preflight", MODULE_PATH)
assert SPEC is not None and SPEC.loader is not None
p = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = p
SPEC.loader.exec_module(p)


def checks(coefficients: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    chosen = np.flatnonzero(coefficients)
    if not len(chosen):
        zero = np.zeros((p.NUM_CHECKS, p.NUM_QUBITS), dtype=np.uint8)
        return zero, zero.copy()
    hx = np.bitwise_xor.reduce(
        np.asarray([p.coefficient_checks()[index][0] for index in chosen]), axis=0
    )
    hz = np.bitwise_xor.reduce(
        np.asarray([p.coefficient_checks()[index][1] for index in chosen]), axis=0
    )
    return hx, hz


def test_graph_seed_has_disjoint_rank_16_orbit() -> None:
    support = p.random_graph_seed(random.Random(1), 6)
    seed = np.zeros(p.NUM_QUBITS, dtype=np.uint8)
    seed[list(support)] = 1
    orbit = p.translation_orbit(seed)
    assert orbit.shape == (16, 128)
    assert np.all(np.sum(orbit, axis=0) <= 1)
    assert p.gf2_rank(orbit) == 16


def test_automatic_css_relations_really_commute() -> None:
    rng = np.random.default_rng(2)
    for relation in ("identity", "swap"):
        for q in (((0, 0),), ((1, 2),), ((0, 0), (1, 0), (0, 3))):
            equations = p.automatic_css_equations(relation, q)
            reduced, pivots = p.gf2_rref(equations)
            basis = p.gf2_nullspace_from_rref(
                reduced, pivots, p.NUM_COEFFICIENTS
            )
            bits = rng.integers(0, 2, size=len(basis), dtype=np.uint8)
            coefficients = (
                np.bitwise_xor.reduce(basis[np.flatnonzero(bits)], axis=0)
                if np.any(bits)
                else np.zeros(p.NUM_COEFFICIENTS, dtype=np.uint8)
            )
            assert not np.any((equations @ coefficients) % 2)
            hx, hz = checks(coefficients)
            assert not np.any((hx @ hz.T) % 2)


def test_exact_fold_nullspace_satisfies_seed_and_fold() -> None:
    geometry = p.search_geometry(
        rng_seed=3, weight=6, trials_per_fold=5, target=1
    )
    witness = geometry["witnesses"][0]
    data_fold = p.fold_from_dict(witness["fold"])
    check_fold = next(iter(p.iter_structured_gl_folds(p.NUM_CHECK_BLOCKS)))
    seed_constraints = p.seed_constraint_columns(witness["seed_support"])
    fold_constraints = p.forward_fold_columns(
        data_fold.forward_blocks,
        data_fold.backward_blocks,
        check_fold.forward_blocks,
        check_fold.backward_blocks,
    )
    equations = np.vstack([seed_constraints, fold_constraints])
    reduced, pivots = p.gf2_rref(equations)
    basis = p.gf2_nullspace_from_rref(
        reduced, pivots, p.NUM_COEFFICIENTS
    )
    assert len(basis)
    coefficients = basis[0]
    hx, hz = checks(coefficients)
    seed = np.zeros(p.NUM_QUBITS, dtype=np.uint8)
    seed[witness["seed_support"]] = 1
    assert not np.any((hx @ seed) % 2)
    folded_hx = p._permute_rows(
        p._permute_columns(hx, data_fold.permutation), check_fold.permutation
    )
    assert np.array_equal(hz, folded_hx)
