import json
import random
from collections import Counter
from pathlib import Path


ROWS = [
    [8, 9, 18, 20, 23, 26, 30, 31],
    [5, 6, 7, 12, 14, 15, 24, 27],
    [1, 5, 10, 11, 16, 19, 24, 28],
    [7, 15, 23, 31],
    [1, 4, 18, 21, 25, 26, 28, 29],
    [2, 5, 9, 10, 12, 13, 17, 20],
    [0, 3, 13, 14, 15, 20, 22, 23],
    [4, 6, 8, 11, 24, 26],
    [0, 4, 8, 14, 18, 21, 27, 29],
    [3, 5, 9, 14, 19, 21, 25, 30],
    [1, 2, 17, 18, 19, 21, 25, 30],
    [1, 6, 11, 13, 17, 22, 27, 29],
    [4, 6, 15, 16, 19, 23, 29, 30],
    [8, 10, 20, 22, 24, 27],
    [9, 10, 11, 13, 17, 22, 25, 26],
]


def edge_coloring(real_edges, left_count, right_count):
    left_degrees = Counter(left for left, _ in real_edges)
    right_degrees = Counter(right for _, right in real_edges)
    delta = max(max(left_degrees.values()), max(right_degrees.values()))
    size = max(left_count, right_count)
    multiplicity = Counter(real_edges)
    real_multiplicity = Counter(real_edges)
    left_deficit = [left for left in range(size) for _ in range(delta - left_degrees[left])]
    right_deficit = [right for right in range(size) for _ in range(delta - right_degrees[right])]
    rng = random.Random(32)
    rng.shuffle(right_deficit)
    for left, right in zip(left_deficit, right_deficit):
        multiplicity[left, right] += 1

    layers = []
    for _ in range(delta):
        match_right = [None] * size

        def augment(left, seen):
            for right in range(size):
                if not multiplicity[left, right] or seen[right]:
                    continue
                seen[right] = True
                if match_right[right] is None or augment(match_right[right], seen):
                    match_right[right] = left
                    return True
            return False

        for left in range(size):
            assert augment(left, [False] * size)
        matching = {left: right for right, left in enumerate(match_right)}
        assert len(matching) == size
        layer = []
        for left in range(size):
            right = matching[left]
            pair = left, right
            multiplicity[pair] -= 1
            if real_multiplicity[pair]:
                real_multiplicity[pair] -= 1
                layer.append(pair)
        layers.append(sorted(layer))
    assert not any(real_multiplicity.values())
    return layers


edges = [(check, qubit) for check, row in enumerate(ROWS) for qubit in row]
layers = edge_coloring(edges, len(ROWS), 32)
observed = [edge for layer in layers for edge in layer]
assert Counter(observed) == Counter(edges)
assert all(len({c for c, _ in layer}) == len(layer) for layer in layers)
assert all(len({q for _, q in layer}) == len(layer) for layer in layers)

column_degrees = [sum(q in row for row in ROWS) for q in range(32)]
result = {
    "code": "[[32,4,5]]",
    "generator_rows": ROWS,
    "independent_checks": 14,
    "measured_checks_per_type": 15,
    "redundant_check": 14,
    "row_weights": [len(row) for row in ROWS],
    "column_degrees_per_type": column_degrees,
    "syndrome_parity_even": all(degree % 2 == 0 for degree in column_degrees),
    "corrected_transversal_s": {
        "physical_operation": "Z(t) S^tensor32",
        "t_support": [1, 2, 4, 5, 6, 7, 14, 15, 16, 24],
        "logical_action": "S on each of the four disjoint logical qubits",
        "note": "Plain S^tensor32 is not code-preserving because weight-6 stabilizers occur.",
    },
    "safe_schedule": {
        "cnot_layers": 16,
        "instructions": (
            "Apply layers 1..8 as data->Z-ancilla CNOTs, then apply layers "
            "1..8 again as X-ancilla->data CNOTs."
        ),
        "per_type_layers": [
            [{"check": check, "data": qubit} for check, qubit in layer]
            for layer in layers
        ],
    },
    "simultaneous_target": {
        "lower_bound_cnot_layers": 8,
        "status": (
            "An 8-layer collision-free edge coloring exists, but a hook-aware "
            "simultaneous X/Z ordering and its circuit fault distance are not certified."
        ),
    },
}
payload = json.dumps(result, indent=2) + "\n"
output = Path(
    "/home/judah_unmuth/gala-code-search/c4_invariant_n32/schedule/"
    "safe_sequential_schedule.json"
)
output.write_text(payload)
print(payload, end="")
