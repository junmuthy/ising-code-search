"""Exact controls, independent small brute force, and representation tests."""
import io
import itertools
import random
import subprocess
import json
from pathlib import Path

import numpy as np
import pytest

from algebra import (basis, canonical_x, combine, commute, hadamard, inverse,
                     low_logical, module, nullspace, pack, pairing, permute,
                     quotient_basis, rank, reduce, transpose, unpack)
from models import (Extension, affine_fold, decode_mat, gl_checks, lift_gl,
                    mat_commutator, mat_mul, positive_control, translation)


def test_binary_linear_algebra():
    rng = random.Random(17)
    for n in range(1, 20):
        rows = [rng.getrandbits(n) for _ in range(n//2)]
        kernel = nullspace(rows, n)
        assert len(kernel) == n-rank(rows)
        assert commute(rows, kernel)
        assert pack(unpack(rows, n)) == rows
        assert transpose(transpose(rows, n), len(rows)) == rows
        invertible = [1 << j for j in range(n)]
        for _ in range(n*2):
            i,j = rng.randrange(n),rng.randrange(n)
            if i != j: invertible[i] ^= invertible[j]
        inv = inverse(invertible)
        assert [combine(inv, row) for row in invertible] == [1 << j for j in range(n)]


@pytest.mark.parametrize('alpha,beta', list(itertools.product(range(2),repeat=2)))
def test_extension(alpha,beta):
    e = Extension(alpha,beta)
    t = e.table
    for a in range(64):
        assert np.array_equal(t[t[a,:],:],t[a,t])  # All 64^3 triples.
    assert e.mul(16,2) == e.mul(e.mul(2,16),1)
    assert e.mul(1,1) == 0
    for g in (1,2,16):
        left = unpack(e.lift(1 << g),64)
        assert np.array_equal(left@left.T % 2,np.eye(64,dtype=np.uint8))
        for h in (1,2,16):
            # Left and right multiplication must commute on every coordinate.
            for q in range(64):
                assert t[g,t[q,h]] == t[t[g,q],h]
    for p in e.folds():
        assert sorted(p) == list(range(128))


def test_faithful_lift_and_actual_transpose():
    rng = random.Random(18)
    for _ in range(5):
        a,b = decode_mat(rng.getrandbits(128)),decode_mat(rng.getrandbits(128))
        aa,bb = unpack(lift_gl(a),64),unpack(lift_gl(b),64)
        assert pack(aa@bb % 2) == lift_gl(mat_mul(a,b))
        hx,hz = gl_checks(a,b)
        assert commute(hx,hz) == (mat_commutator(a,b) == 0)


def test_saved_positive_control():
    hx,hz,px,py,fold = positive_control()
    saved = subprocess.check_output(['git','-C','/home/judah_unmuth/gala-code-search',
                                    'show','8481d1e:c4xc8_abelian_lp_160_32_6/checks.npz'])
    data = np.load(io.BytesIO(saved))
    # Names are deliberately checked here, not inferred from matrix order.
    assert pack(data['hx']) == hx
    assert pack(data['hz']) == hz
    info,grid = module(hx,hz,px,py)
    assert info['status'] == 'regular_grid'
    assert (info['n'], info['k'], info['rank_x'],info['rank_z']) == (160,32,64,64)
    h,logical = hadamard(hx,hz,grid,fold)
    assert h['status'] == 'logical_hadamard'
    for check,dual,support in ((hx,logical['x'],[34,47,56,105,108,125]),
                               (hz,logical['z'],[84,87,91,110,119,126])):
        assert low_logical(check,dual,160) is None
        v = sum(1 << q for q in support)
        assert commute(check,[v])
        assert pairing([v],dual)[0]


def test_norm_negative_control():
    identity = list(range(32))
    info,_ = module([],[],identity,identity)
    assert info['status'] == 'norm_zero'


def test_physical_zx_does_not_imply_correct_logical_h():
    hx,hz,px,py,p = positive_control()
    _,grid = module(hx,hz,px,py)
    z = grid['z'].copy()
    z[0] ^= z[1]
    x = canonical_x(z,grid['x'])
    h,_ = hadamard(hx,hz,{'z':z,'x':x,'perms':grid['perms']},p,solve_seconds=0)
    assert h['status'] == 'basis_unresolved'


def test_basis_solver_can_recover_equivariant_basis():
    hx,hz,px,py,p = positive_control()
    _,grid = module(hx,hz,px,py)
    seed = grid['z'][0] ^ grid['z'][1] ^ grid['z'][8]
    z = [permute(seed,t) for t in grid['perms']]
    x = canonical_x(z,grid['x'])
    mixed = {'z':z,'x':x,'perms':grid['perms']}
    assert hadamard(hx,hz,mixed,p,solve_seconds=0)[0]['status'] == 'basis_unresolved'
    assert hadamard(hx,hz,mixed,p,solve_seconds=10)[0]['status'] == 'logical_hadamard'


def test_weight_six_geometry_only():
    seed = sum(1 << q for q in (0,1,32,33,64,96))
    z = [permute(seed,translation(i,j)) for i in range(4) for j in range(8)]
    p = affine_fold((1,2,3,0))
    assert rank(z) == rank(pairing(z,[permute(v,p) for v in z])) == 32


def test_distance_hashing_against_bruteforce():
    rng = random.Random(19)
    for _ in range(15):
        n=9
        hx = [rng.getrandbits(n) for _ in range(3)]
        kernel = nullspace(hx,n)
        hz = [combine(kernel,rng.getrandbits(len(kernel))) for _ in range(2)]
        z,xraw = quotient_basis(hx,hz,n),quotient_basis(hz,hx,n)
        x = canonical_x(z,xraw)
        for checks,dual,stabilizers in ((hx,x,hz),(hz,z,hx)):
            for maximum in range(1,6):
                witness = low_logical(checks,dual,n,maximum)
                brute = next((support for w in range(1,maximum+1)
                              for support in itertools.combinations(range(n),w)
                              if commute(checks,[sum(1 << q for q in support)])
                              and reduce(sum(1 << q for q in support),basis(stabilizers))),None)
                assert (witness is None) == (brute is None)
                if witness:
                    assert len(witness) <= maximum
                    v = sum(1 << q for q in witness)
                    assert commute(checks,[v]) and reduce(v,basis(stabilizers))


def test_certified_extension_reconstruction():
    path=Path(__file__).parent/'certified/extension_128_32_6/candidate.json'
    data=json.loads(path.read_text())
    e=Extension(*data['extension'])
    hx,hz=e.checks(data['a'],data['b'])
    assert hx==data['hx'] and hz==data['hz']
    assert module(hx,hz,data['px'],data['py'],data['central'])[0]['status']=='regular_grid'
    cert=data['certificate']
    assert hadamard(hx,hz,cert,cert['fold'],solve_seconds=0)[0]['status']=='logical_hadamard'
    assert low_logical(hx,cert['x'],128) is None
    assert low_logical(hz,cert['z'],128) is None


def test_equivariant_control_reconstruction():
    from fallback import initial,transvection
    path=Path(__file__).parent/'certified/connected_128_32/candidate.json'
    data=json.loads(path.read_text())
    h,z=initial()
    for a,b,order in data['recipe']: h,z=transvection(h,z,a,b,order)
    assert h==data['hx']==data['hz'] and z==data['certificate']['z']==data['certificate']['x']
    assert module(h,h,data['px'],data['py'])[0]['status']=='regular_grid'
    assert commute(h,h) and pairing(z,z)==[1 << i for i in range(32)]


def test_independent_distance_degeneracy():
    from certify import independent_small_errors
    h=unpack([0b0011,0b1100],4)
    result=independent_small_errors(h,h)
    assert result['status']=='certified'
    assert result['zero_syndrome_counts']=={2:2,4:1}
    result=independent_small_errors(unpack([0b1111],4),unpack([0b1111],4))
    assert result['status']=='logical_witness' and len(result['support'])==2


def test_capacity_kernel_equations():
    from preflight import capacity_space,parameter_checks
    seed=sum(1 << q for q in (0,1,32,33,64,96))
    p=affine_fold((1,2,3,0))
    q=affine_fold((0,1))
    variables=[parameter_checks(1 << i) for i in range(256)]
    kernel,rx,rz=capacity_space(seed,p,q,q,variables)
    assert (len(kernel),rx,rz)==(18,32,32)
    for v in kernel:
        hx,hz=parameter_checks(v)
        assert commute(hx,[seed])
        assert [permute(row,p) for row in hx]==[hz[j] for j in q]
        assert [permute(row,p) for row in hz]==[hx[j] for j in q]


def test_connected_exact_distance_seven():
    from certify import weight_six_witness
    path=Path(__file__).parent/'certified/connected_128_32/candidate.json'
    data=json.loads(path.read_text())
    h=data['hx']
    z=data['certificate']['z']
    assert h==data['hz'] and z==data['certificate']['x']
    assert low_logical(h,z,128) is None
    assert weight_six_witness(unpack(h,128),unpack(h,128)) is None
    witness=sum(1 << q for q in (1,21,30,42,64,75,94))
    assert commute(h,[witness]) and pairing([witness],z)[0]
