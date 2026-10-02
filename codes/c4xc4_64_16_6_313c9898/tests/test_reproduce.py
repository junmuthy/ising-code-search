import copy

import pytest

from reproduce import check_manifest,reconstruct,validate_circuit,validate_static
from schedule import verify


def test_manifest_and_construction():
    assert check_manifest()>100
    data,code,schedule=reconstruct()
    assert len(data['hx'])==len(data['hz'])==24
    assert schedule['depth']==12
    assert sum(map(len,schedule['layers']))==576


def test_distance_and_gates():
    data,_,_=reconstruct()
    result=validate_static(data)
    assert result['actions']['H']['physical_order']==4
    assert result['actions']['Tx']['physical_order']==8
    assert result['actions']['Ty']['physical_order']==4


@pytest.mark.parametrize('basis',['X','Z'])
def test_full_circuit_and_six_fault_witness(basis):
    _,code,schedule=reconstruct()
    assert validate_circuit(code,schedule,basis).num_qubits==112


def test_corrupt_schedule_rejected():
    _,code,schedule=reconstruct()
    bad=copy.deepcopy(schedule)
    bad['layers'][0].append(bad['layers'][0][0])
    with pytest.raises(AssertionError):
        verify(code,bad)
