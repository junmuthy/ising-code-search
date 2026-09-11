"""Portable backup regression tests, including a separate distance oracle."""
import hashlib
import json
import random

import pytest

from certified_css_grid_codes import verify as v


@pytest.mark.parametrize("name,n,k,d", [
    ("n64_k16_d8", 64, 16, 8),
    ("n128_k32_d6", 128, 32, 6),
    ("n128_k32_d7", 128, 32, 7),
])
def test_saved_codes_exact(name, n, k, d):
    report = v.verify_code(v.ROOT / name, exact_distance=True)
    assert (report["n"], report["k"], report["claimed_distance"]) == (n, k, d)
    assert report["status"] == "exact_distance_verified"
    for side in ("X", "Z"):
        assert report["distance"][side]["lower_bound"] == d
        assert report["distance"][side]["upper_bound"] == d


def test_package_manifest_is_complete():
    manifest = json.loads((v.ROOT / "MANIFEST.json").read_text())["files"]
    actual = {
        path.relative_to(v.ROOT).as_posix()
        for path in v.ROOT.rglob("*")
        if path.is_file() and path.name != "MANIFEST.json"
        and "__pycache__" not in path.parts
    }
    assert set(manifest) == actual
    assert v.verify_manifest() == len(actual)


def test_manifest_rejects_corruption(tmp_path):
    original = b"saved code\n"
    (tmp_path / "data").write_bytes(original)
    manifest = {"files": {"data": {
        "bytes": len(original), "sha256": hashlib.sha256(original).hexdigest(),
    }}}
    (tmp_path / "MANIFEST.json").write_text(json.dumps(manifest))
    assert v.verify_manifest(tmp_path) == 1
    (tmp_path / "data").write_bytes(b"bad! code\n")
    with pytest.raises(AssertionError):
        v.verify_manifest(tmp_path)


def test_distance_enumerator_handles_degeneracy():
    report = v.exclude_through([0b0011, 0b1100], [0b0011, 0b1100], 4, 4)
    assert report["zero_syndrome_counts"] == {1: 0, 2: 2, 3: 0, 4: 1}


def test_distance_enumerator_rejects_short_logical():
    with pytest.raises(AssertionError, match="Logical below claimed distance"):
        v.exclude_through([0b0011], [0b0011], 4, 1)


@pytest.mark.parametrize("seed", range(16))
def test_distance_matches_independent_brute_force(seed):
    rng = random.Random(seed)
    width, maximum = 10, 5
    checks = [rng.randrange(1 << width) for _ in range(4)]
    # Enumerate the whole small kernel independently, then select stabilizers.
    kernel = [word for word in range(1 << width)
              if all((word & row).bit_count() % 2 == 0 for row in checks)]
    stabilizers = rng.sample(kernel, seed % 8)
    span = {0}
    for row in stabilizers:
        span |= {word ^ row for word in tuple(span)}
    short = [word for word in kernel if 0 < word.bit_count() <= maximum]
    if any(word not in span for word in short):
        with pytest.raises(AssertionError, match="Logical below claimed distance"):
            v.exclude_through(checks, stabilizers, width, maximum)
    else:
        report = v.exclude_through(checks, stabilizers, width, maximum)
        assert report["zero_syndrome_counts"] == {
            weight: sum(word.bit_count() == weight for word in short)
            for weight in range(1, maximum + 1)
        }


def test_component_graph_does_not_row_reduce_mixed_sectors():
    assert v.components([0b0011, 0b1111], 4) == [4]
    assert v.components([0b0011, 0b1100], 4) == [2, 2]


def test_transpose_and_permutation_conventions():
    rows = [0b0011, 0b1100]
    assert v.transpose(rows, 4) == [1, 1, 2, 2]
    p = [1, 2, 3, 0]
    assert v.move(0b0011, p) == 0b0110
    assert v.order(p) == 4
    assert v.compose(p, p) == [2, 3, 0, 1]
