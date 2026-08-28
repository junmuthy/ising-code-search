from __future__ import annotations

import numpy as np

import scan_frontier_folds as scan
import scan_shifted_folds as shifted
from folded_refinement import analyze_folded, seed_from_record


def test_active_thickness_has_232_involutions() -> None:
    values = tuple(scan.involutions(tuple(range(scan.ACTIVE_THICKNESS))))
    assert len(values) == 232
    assert len(set(values)) == 232
    for value in values:
        assert tuple(value[value[index]] for index in range(len(value))) == tuple(
            range(len(value))
        )


def test_every_scanned_fold_has_clean_logical_pairing() -> None:
    logicals = scan.logical_columns()
    for epsilon in (1, -1):
        for images in scan.involutions(tuple(range(scan.ACTIVE_THICKNESS))):
            permutation = scan.fold_permutation(images, epsilon)
            assert np.array_equal(permutation[permutation], np.arange(scan.NUM_QUBITS))
            pairing = (logicals @ scan.act_rows(logicals, permutation).T) % 2
            assert scan.pairing_is_permutation(pairing)


def test_shifted_fold_catalog_is_involutive_with_clean_pairing() -> None:
    logicals = scan.logical_columns()
    catalog = shifted.shifted_fold_catalog()
    assert catalog
    for fold in catalog:
        permutation = fold["permutation"]
        assert np.array_equal(permutation[permutation], np.arange(scan.NUM_QUBITS))
        pairing = (logicals @ scan.act_rows(logicals, permutation).T) % 2
        assert scan.pairing_is_permutation(pairing)


def test_saved_distinct_folded_seed_analyzes_as_32_4_5() -> None:
    import json
    import pathlib

    path = pathlib.Path(
        "/home/judah_unmuth/gala-code-search/inverse_c4_n32_targeted/"
        "results/shifted-identity-thickness-all100-260828-v1/folded-seeds.jsonl"
    )
    record = json.loads(path.read_text().splitlines()[0])
    stabilizer, permutation = seed_from_record(record)
    analysis = analyze_folded(stabilizer, permutation)
    assert analysis["n"] == 32
    assert analysis["k"] == 4
    assert analysis["distance"] == 5
    assert analysis["checks"]["css_orthogonal"]
    assert analysis["checks"]["distinct_check_spaces"]
    assert analysis["checks"]["logical_pairing_permutation"]
