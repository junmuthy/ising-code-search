"""Regression controls for a same-code logical representative change."""
import json

import numpy as np
import pytest

from distance8_logical_handoff import SOURCE, export, simplify


def test_all_logicals_have_weight_twelve_with_original_labels(tmp_path):
    data, report, arrays = simplify()
    original = json.loads((SOURCE/'independent/candidate.json').read_text())
    assert data['hx'] == original['hx'] and data['hz'] == original['hz']
    assert data['certificate']['z'] == original['certificate']['z']
    assert report['x_weights'] == report['z_weights'] == {12: 16}
    assert report['logical_support_total_before'] == 504
    assert report['logical_support_total_after'] == 384
    assert len(report['hadamard_pairs']) == 8
    assert all(len(v['logical_grid_labels']) == 10 for v in report['saved_distance_witnesses'].values())
    assert report['logical_weights_optimal'] is False
    exported = export(tmp_path/'handoff')
    assert exported['status'] == report['status']
    with np.load(tmp_path/'handoff/logical_bases.npz') as loaded:
        assert np.array_equal(loaded['x'], arrays[2])
        assert np.array_equal(loaded['z'], arrays[3])
    with pytest.raises(FileExistsError):
        export(tmp_path/'handoff')
