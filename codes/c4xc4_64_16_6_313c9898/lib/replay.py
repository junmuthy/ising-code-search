"""Original physical witness replay against the unmodified reference."""
import numpy as np
import stim
from model import digest
from circuit import NOISE

def replay_saved(circuit, witness, depth):
    assert digest(str(circuit).encode()) == witness['circuit_sha256']
    insertion, channels, described = {}, set(), []
    for fault in witness['faults']:
        offset = fault['instruction_offset']
        start, stop = fault['target_range']
        inst = circuit[offset]
        assert inst.name == fault['gate'] and inst.name in NOISE
        assert inst.gate_args_copy()[0] > 0
        targets = inst.targets_copy()[start:stop]
        expected_length = 2 if inst.name == 'DEPOLARIZE2' else 1
        assert len(targets) == stop - start == expected_length
        assert start % expected_length == 0
        qubits = {target.value for target in targets}
        key = offset, start, stop
        assert key not in channels
        channels.add(key)
        assert fault['paulis']
        assert len({q for q, _ in fault['paulis']}) == len(fault['paulis'])
        for q, pauli in fault['paulis']:
            assert q in qubits and pauli in ('X', 'Y', 'Z')
            if inst.name.endswith('_ERROR'):
                assert pauli == inst.name[0]
            insertion.setdefault(offset, []).append((q, pauli))
        ticks = sum(item.name == 'TICK' for item in list(circuit)[:offset])
        described.append(dict(gate=inst.name, paulis=fault['paulis'], ticks=ticks,
                              round=(ticks - 2) // (depth + 2) if inst.name == 'DEPOLARIZE2' else None,
                              layer=(ticks - 2) % (depth + 2) if inst.name == 'DEPOLARIZE2' else None))
    actual = stim.Circuit()
    for offset, instruction in enumerate(circuit):
        if instruction.name not in NOISE:
            actual.append(instruction)
        for q, pauli in insertion.get(offset, []):
            actual.append(pauli, [q])
    actual.detector_error_model(allow_gauge_detectors=False)
    converter = circuit.without_noise().compile_m2d_converter()
    reference_d, reference_o = converter.convert(measurements=actual.reference_sample()[None, :], separate_observables=True)
    dm = sum(int(v) << i for i, v in enumerate(reference_d[0]))
    om = sum(int(v) << i for i, v in enumerate(reference_o[0]))
    assert dm == witness['detector_mask'] == 0
    assert om == witness['observable_mask'] and om != 0
    assert len(channels) == witness['fault_count']
    d, o = converter.convert(measurements=actual.compile_sampler(seed=17).sample(32), separate_observables=True)
    assert np.all(d == reference_d) and np.all(o == reference_o)
    return dict(fault_count=len(channels), zero_detectors=True,
                logical_indices=[i for i in range(16) if om >> i & 1], physical_locations=described)

