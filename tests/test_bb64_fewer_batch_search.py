"""Regression tests for the BB64 one-/two-batch follow-up search."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from searches.bb64_simultaneous_basis_search.generate_representation_document import (
    DEFAULT_BASIS as DOCUMENT_BASIS,
    DEFAULT_OUTPUT as REPRESENTATION_DOCUMENT,
    DEFAULT_SCHEDULE as DOCUMENT_SCHEDULE,
    markdown_table,
    render_document,
)
from searches.bb64_simultaneous_basis_search.fewer_batch_search.search import (
    DEFAULT_BASIS,
    DEFAULT_CANDIDATES,
    base_from_record,
    certify_basis,
    partitions_from_colors,
    x_partition_from_z,
)


def test_saved_candidate_catalog_is_the_expected_eight_class_sets() -> None:
    candidates = json.loads(DEFAULT_CANDIDATES.read_text())
    assert len(candidates) == 8
    assert {tuple(candidate["hadamard_permutation"]) for candidate in candidates} == {
        (5, 4, 7, 6, 1, 0, 3, 2)
    }
    assert all(base_from_record(candidate).shape == (8, 64) for candidate in candidates)


def test_partition_helpers_preserve_exact_h_support_map() -> None:
    permutation = np.asarray([5, 4, 7, 6, 1, 0, 3, 2])
    z_partition = [[0, 1, 7], [4, 6], [2, 3, 5]]
    assert x_partition_from_z(z_partition, permutation) == [
        [2, 4, 5],
        [1, 3],
        [0, 6, 7],
    ]
    assert partitions_from_colors([0, 0, 1, 1, 1, 0, 1, 0], 2) == [
        [0, 1, 5, 7],
        [2, 3, 4, 6],
    ]


def test_current_three_batch_basis_passes_the_shared_certificate() -> None:
    source = np.load(DEFAULT_BASIS)
    logical_z = np.asarray(source["logical_z"], dtype=np.uint8)
    pairing = np.asarray(source["zx_pairing"], dtype=np.uint8)
    permutation = np.asarray(source["hadamard_permutation"], dtype=np.int64)
    z_batch = np.asarray(source["disjoint_batch_of_z_logical"], dtype=np.int64)
    partition = [
        np.flatnonzero(z_batch == batch).astype(int).tolist()
        for batch in sorted(set(map(int, z_batch)))
    ]
    certificate = certify_basis(
        np.asarray(source["matrix_z"], dtype=np.uint8),
        logical_z,
        pairing,
        permutation,
        partition,
        np.asarray(source["grid_x_physical_permutation"], dtype=np.int64),
        np.asarray(source["grid_y_physical_permutation"], dtype=np.int64),
    )
    assert np.array_equal(certificate["logical_x"], source["logical_x"])
    assert certificate["x_partition"] == [[2, 4, 5], [1, 3], [0, 6, 7]]


def test_complete_representation_document_is_current_and_has_all_cnots() -> None:
    rendered = render_document(DOCUMENT_BASIS, DOCUMENT_SCHEDULE)
    assert REPRESENTATION_DOCUMENT.read_text() == rendered
    assert rendered.count(r"\!\to") == 512
    assert rendered.count("### CNOT layer ") == 8
    assert "\\mathcal B_Z^{(0)}=\\{0,3,4,7\\}" in rendered
    assert "p=(0\\;5)(1\\;4)(2\\;7)(3\\;6)" in rendered
    assert "## Explicit physical automorphisms" in rendered
    assert "(q_{0}\\;q_{1}\\;q_{2}\\;q_{3}\\;q_{4}\\;q_{5}\\;q_{6}\\;q_{7})" in rendered
    assert "(q_{0}\\;q_{32})" in rendered
    assert "P_y:\\quad L(u,v)\\mapsto R(u-2v,-v)" in rendered


def test_markdown_table_pipe_columns_are_source_aligned() -> None:
    table = markdown_table(
        ("short", "longer heading", "last"),
        (("a", "short", "one"), ("much longer", "b", "three")),
    )
    pipe_positions = [
        tuple(index for index, character in enumerate(line) if character == "|")
        for line in table.splitlines()
    ]
    assert len(set(pipe_positions)) == 1
