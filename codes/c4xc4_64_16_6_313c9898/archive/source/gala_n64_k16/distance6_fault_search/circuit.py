"""Generic dedicated-ancilla CSS memory, with explicit boundary/idle noise."""
from __future__ import annotations

import stim

NOISE = {'DEPOLARIZE1', 'DEPOLARIZE2', 'X_ERROR', 'Y_ERROR', 'Z_ERROR'}


def build(code, schedule, basis, *, experiment='memory', rounds=3, noise='full_wait', probability=0.001):
    assert basis in ('X', 'Z') and noise in ('none', 'cnot', 'full', 'full_wait')
    assert experiment in ('memory', 'bulk') and rounds >= 1 and 0 <= probability <= 1
    if experiment == 'bulk':
        rounds = 3
    data = list(range(64))
    counts = {s: len(code['checks'][s]) for s in ('X', 'Z')}
    anc = dict(X=list(range(64, 64 + counts['X'])),
               Z=list(range(64 + counts['X'], 64 + counts['X'] + counts['Z'])))
    all_qubits = data + anc['X'] + anc['Z']
    c = stim.Circuit()
    p = probability if noise != 'none' else 0
    def depolarize(qubits, enabled=True):
        if p and enabled and qubits:
            c.append('DEPOLARIZE1', qubits, p)
    def measurement_noise(qubits, kind, enabled=True):
        if p and enabled and noise in ('full', 'full_wait'):
            c.append('Z_ERROR' if kind == 'X' else 'X_ERROR', qubits, p)
    def detector(indices, coords):
        c.append('DETECTOR', [stim.target_rec(i - c.num_measurements) for i in indices], coords)
    c.append('RX' if basis == 'X' else 'R', data)
    depolarize(data, experiment == 'memory' and noise in ('full', 'full_wait'))
    c.append('TICK')
    previous = None
    for r in range(rounds):
        noisy = p > 0 and (experiment == 'memory' or r == 1)
        c.append('RX', anc['X'])
        c.append('R', anc['Z'])
        depolarize(anc['X'] + anc['Z'], noisy and noise in ('full', 'full_wait'))
        depolarize(data, noisy and noise == 'full_wait')
        c.append('TICK')
        for layer in schedule['layers']:
            pairs, used = [], set()
            for gate in layer:
                a = anc[gate['type']][gate['check']]
                pair = [a, gate['data']] if gate['type'] == 'X' else [gate['data'], a]
                pairs.extend(pair)
                used.update(pair)
            if pairs:
                c.append('CX', pairs)
                if noisy:
                    c.append('DEPOLARIZE2', pairs, p)
            depolarize([q for q in all_qubits if q not in used], noisy and noise in ('full', 'full_wait'))
            c.append('TICK')
        depolarize(data, noisy and noise == 'full_wait')
        current = {}
        for kind in ('X', 'Z'):
            measurement_noise(anc[kind], kind, noisy)
            start = c.num_measurements
            c.append('MX' if kind == 'X' else 'M', anc[kind])
            current[kind] = list(range(start, start + counts[kind]))
        for kind in ('X', 'Z'):
            if previous is not None:
                for check in range(counts[kind]):
                    detector([previous[kind][check], current[kind][check]], [r, int(kind == 'Z'), check])
            elif kind == basis:
                for check in range(counts[kind]):
                    detector([current[kind][check]], [r, int(kind == 'Z'), check])
            for i, relation in enumerate(code['relations'][kind]):
                detector([current[kind][check] for check in relation], [r, int(kind == 'Z'), counts[kind] + i])
        c.append('TICK')
        previous = current
    measurement_noise(data, basis, experiment == 'memory')
    start = c.num_measurements
    c.append('MX' if basis == 'X' else 'M', data)
    for check, support in enumerate(code['supports'][basis]):
        detector([previous[basis][check]] + [start + q for q in support], [rounds, int(basis == 'Z'), check])
    for logical, support in enumerate(code['logical_supports'][basis]):
        c.append('OBSERVABLE_INCLUDE', [stim.target_rec(start + q - c.num_measurements) for q in support], logical)
    expected = (rounds - 1) * sum(counts.values()) + 2 * counts[basis]
    expected += rounds * sum(map(len, code['relations'].values()))
    assert c.num_observables == 16 and c.num_detectors == expected
    assert c.num_qubits == len(all_qubits) and c.num_measurements == sum(counts.values()) * rounds + 64
    return c
