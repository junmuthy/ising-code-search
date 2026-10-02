"""Regression checks for the equivalent minimum-weight stabilizer basis."""

from collections import Counter
import hashlib
import json
from pathlib import Path

import numpy as np
import pytest

from check_weight_reduction import (
    SOURCE, coordinates, enumerate_space, load_score, orbit_search,
)
from algebra import combine, permute, power, rank, same_space, unpack
from extract_component import audit


HERE = Path(__file__).parent
OUTPUT = HERE / "check_weight_reduction"


@pytest.fixture(scope="module")
def artifacts():
    return (
        json.loads(SOURCE.read_text()),
        json.loads((OUTPUT / "candidate.json").read_text()),
        json.loads((OUTPUT / "report.json").read_text()),
    )


@pytest.mark.parametrize("rows", [[1], [3, 5, 6], [7, 25, 42, 7 ^ 25]])
def test_enumeration_against_scalar_reference(rows):
    words = {combine(rows, mask) for mask in range(1 << len(rows))}
    result = enumerate_space(rows, ceiling=6, chunk=1)
    assert result["weight_histogram"] == Counter(v.bit_count() for v in words)
    assert result["enumerated"] == len(words)
    assert set(result["low_weight_rows"]) == words - {0}
    assert result["rank_filtration"] == {
        w: rank([v for v in words if v.bit_count() <= w]) for w in range(7)
    }


@pytest.mark.parametrize("sector", ["hx", "hz"])
def test_complete_weight_enumeration(artifacts, sector):
    original, _, report = artifacts
    fresh = enumerate_space(original[sector])
    saved = report["enumerations"][sector]
    # JSON stores integer dictionary keys as strings.
    assert json.loads(json.dumps(fresh)) == saved
    assert fresh["enumerated"] == 2**24
    assert fresh["weight_histogram"][12] == 56
    assert fresh["weight_histogram"][14] == 112
    assert fresh["rank_filtration"][11] == 0
    assert fresh["rank_filtration"][12] == 18
    assert fresh["rank_filtration"][13] == 18
    assert fresh["rank_filtration"][14] == 24
    assert fresh["minimum_basis_weight_counts"] == {12: 18, 14: 6}
    assert fresh["minimum_basis_total_weight"] == 300
    assert fresh["minimum_possible_maximum_check_weight"] == 14


def test_basis_changes_and_exports(artifacts):
    original, optimized, report = artifacts
    assert hashlib.sha256(SOURCE.read_bytes()).hexdigest() == report["source_sha256"]
    assert hashlib.sha256((HERE / "check_weight_reduction.py").read_bytes()).hexdigest() == report["script_sha256"]
    for key, value in original.items():
        if key not in ("hx", "hz"):
            assert optimized[key] == value
    supports = json.loads((OUTPUT / "supports.json").read_text())
    with np.load(OUTPUT / "checks.npz") as matrices:
        for side in ("hx", "hz"):
            old, new = original[side], optimized[side]
            assert len(new) == rank(new) == 24
            assert same_space(old, new)
            assert Counter(v.bit_count() for v in new) == {12: 18, 14: 6}
            changes = report["basis_changes"][side]
            assert [combine(old, c) for c in changes["new_from_original"]] == new
            assert [combine(new, c) for c in changes["original_from_new"]] == old
            assert np.array_equal(matrices[side], unpack(new, 64))
            assert supports[side] == [[q for q in range(64) if v >> q & 1] for v in new]
    assert sum(v.bit_count() for side in ("hx", "hz") for v in original[side]) == 896
    assert sum(v.bit_count() for side in ("hx", "hz") for v in optimized[side]) == 600
    assert load_score(list(range(24)), unpack(optimized["hx"], 64), unpack(optimized["hz"], 64)) == (12, 7, 5772)


def test_stabilizer_actions(artifacts):
    _, code, report = artifacts
    actions = report["stabilizer_actions"]
    for side in ("hx", "hz"):
        for name, permutation in (("Tx", code["px"]), ("Ty", code["py"])):
            assert [combine(code[side], c) for c in actions[side][name]] == [
                permute(v, permutation) for v in code[side]
            ]
    fold = code["certificate"]["fold"]
    for name, source, target in (("x_to_z", "hx", "hz"), ("z_to_x", "hz", "hx")):
        assert [combine(code[target], c) for c in actions["H"][name]] == [
            permute(v, fold) for v in code[source]
        ]
    assert actions["H"]["x_to_z"] == [1 << i for i in range(24)]
    assert power(fold, 2) == power(code["px"], 4) == code["central"]


def test_symmetry_closed_optimum(artifacts):
    original, _, report = artifacts
    words = report["enumerations"]["hx"]["low_weight_rows"]
    fresh = orbit_search(words, original["px"], original["py"], original["certificate"]["fold"])
    assert fresh == report["symmetry_closed_search"]
    assert fresh["orbit_count"] == 9
    assert fresh["subsets_examined"] == 511
    orbit_rows = [v for orbit in fresh["orbits"] for v in orbit["rows"]]
    assert len(orbit_rows) == len(set(orbit_rows)) == 168
    assert set(orbit_rows) == set(words)
    assert fresh["minimum_total_weight_per_sector"] == 448
    assert fresh["number_checks_per_sector"] == 32
    assert set(fresh["best_x_rows"]) == set(original["hx"])


def test_z_symmetry_closed_optimum(artifacts):
    original, _, report = artifacts
    words = report["enumerations"]["hz"]["low_weight_rows"]
    result = orbit_search(words, original["px"], original["py"], original["certificate"]["fold"])
    assert result["orbit_count"] == 9
    assert result["subsets_examined"] == 511
    assert result["minimum_total_weight_per_sector"] == 448
    assert result["number_checks_per_sector"] == 32
    assert set(result["best_x_rows"]) == set(original["hz"])


def test_independent_full_code_audit(artifacts):
    _, optimized, _ = artifacts
    result, _ = audit(optimized, seconds=0)
    assert (result["n"], result["k"], result["d"]) == (64, 16, 6)
    assert result["metrics"]["row_space_component_sizes"] == [64]
    assert result["actions"]["H"]["logical_permutation"] == [
        ((-i) % 4) * 4 + (3 - j) % 4 for i in range(4) for j in range(4)
    ]
    for sector in ("X", "Z"):
        assert result["distance"][sector]["exclusion"]["zero_syndrome_total"] == 0
        assert len(result["distance"][sector]["weight_six_witness"]) == 6


def test_coordinates_redundancy_and_nonmember():
    rows = [3, 5, 6, 0]
    for mask in range(1 << len(rows)):
        v = combine(rows, mask)
        assert combine(rows, coordinates(rows, v)) == v
    with pytest.raises(ValueError, match="outside generator span"):
        coordinates(rows, 8)
