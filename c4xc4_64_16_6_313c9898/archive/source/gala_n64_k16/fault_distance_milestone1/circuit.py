"""Explicit Clifford memory circuit; separate X/Z dependencies and 16 observables."""
from __future__ import annotations

import stim

DATA = list(range(64))
ANCILLA = dict(X=list(range(64, 96)), Z=list(range(96, 128)))
ALL = list(range(128))
NOISE = {'DEPOLARIZE1', 'DEPOLARIZE2', 'X_ERROR', 'Y_ERROR', 'Z_ERROR'}


def build(code, schedule, basis, *, experiment='memory', rounds=3, noise='full', probability=0.001):
    assert basis in ('X', 'Z') and noise in ('none', 'cnot', 'full', 'full_wait')
    assert experiment in ('memory', 'bulk') and rounds >= 1
    assert 0 <= probability <= 1
    if experiment == 'bulk':
        rounds = 3
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
    for q in DATA:
        c.append('QUBIT_COORDS', [q], [q // 32, (q % 32) // 4, q % 4])
    c.append('RX' if basis == 'X' else 'R', DATA)
    depolarize(DATA, experiment == 'memory' and noise in ('full', 'full_wait'))
    c.append('TICK')
    previous = None
    for r in range(rounds):
        noisy = p > 0 and (experiment == 'memory' or r == 1)
        c.append('RX', ANCILLA['X'])
        c.append('R', ANCILLA['Z'])
        depolarize(ANCILLA['X'] + ANCILLA['Z'], noisy and noise in ('full', 'full_wait'))
        depolarize(DATA, noisy and noise == 'full_wait')
        c.append('TICK')
        for layer in schedule['layers']:
            pairs, used = [], set()
            for gate in layer:
                ancilla = ANCILLA[gate['type']][gate['check']]
                pair = [ancilla, gate['data']] if gate['type'] == 'X' else [gate['data'], ancilla]
                pairs.extend(pair)
                used.update(pair)
            if pairs:
                c.append('CX', pairs)
                if noisy:
                    c.append('DEPOLARIZE2', pairs, p)
            depolarize([q for q in ALL if q not in used], noisy and noise in ('full', 'full_wait'))
            c.append('TICK')
        depolarize(DATA, noisy and noise == 'full_wait')
        current = {}
        for kind in ('X', 'Z'):
            measurement_noise(ANCILLA[kind], kind, noisy)
            start = c.num_measurements
            c.append('MX' if kind == 'X' else 'M', ANCILLA[kind])
            current[kind] = list(range(start, start + 32))
        for kind in ('X', 'Z'):
            if previous is not None:
                for check in range(32):
                    detector([previous[kind][check], current[kind][check]], [r, int(kind == 'Z'), check])
            elif kind == basis:
                for check in range(32):
                    detector([current[kind][check]], [r, int(kind == 'Z'), check])
            for i, relation in enumerate(code['relations'][kind]):
                detector([current[kind][check] for check in relation], [r, int(kind == 'Z'), 32 + i])
        c.append('TICK')
        previous = current
    measurement_noise(DATA, basis, experiment == 'memory')
    start = c.num_measurements
    c.append('MX' if basis == 'X' else 'M', DATA)
    for check, support in enumerate(code['supports'][basis]):
        detector([previous[basis][check]] + [start + q for q in support], [rounds, int(basis == 'Z'), check])
    for logical, support in enumerate(code['logical_supports'][basis]):
        c.append('OBSERVABLE_INCLUDE', [stim.target_rec(start + q - c.num_measurements) for q in support], logical)
    assert c.num_observables == 16 and c.num_detectors == 80 * rounds
    assert c.num_qubits == 128 and c.num_measurements == 64 * rounds + 64
    return c
