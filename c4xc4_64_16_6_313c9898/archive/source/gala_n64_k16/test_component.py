import json
from pathlib import Path

from extract_component import audit, reconstruct
from algebra import compose, power


HERE=Path(__file__).parent


def test_reconstruction_matches_saved_candidate():
    saved=json.loads((HERE/'certified/64_16_6_c4xc4/candidate.json').read_text())
    assert reconstruct()==saved


def test_independent_algebra_and_exact_distance():
    result,_=audit(reconstruct(),seconds=0)
    assert (result['n'],result['k'],result['d'])==(64,16,6)
    assert result['metrics']['row_space_component_sizes']==[64]
    assert result['actions']['H']['logical_permutation']==[
        ((-i)%4)*4+(3-j)%4 for i in range(4) for j in range(4)]
    for sector in ('X','Z'):
        assert result['distance'][sector]['exclusion']['zero_syndrome_total']==0
        assert len(result['distance'][sector]['weight_six_witness'])==6


def test_physical_vs_logical_orders():
    data=reconstruct()
    identity=list(range(64))
    assert power(data['px'],4)==data['central']!=identity
    assert power(data['px'],8)==identity
    assert power(data['py'],4)==identity and power(data['py'],2)!=identity
    assert compose(data['px'],data['py'])==compose(data['py'],data['px'])
