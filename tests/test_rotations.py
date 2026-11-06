"""Week 3, Mon-Tue: rotations. Spec for rtvm/rotations.py."""
import pytest
import torch

from rtvm.rotations import cayley, cayley_lowrank_apply, lowrank_skew, skew

pytestmark = pytest.mark.week3


def _gen(seed=0):
    return torch.Generator().manual_seed(seed)


def test_skew_is_antisymmetric():
    P = torch.randn(3, 5, 5, generator=_gen())
    A = skew(P)
    assert torch.allclose(A, -A.transpose(-1, -2))


def test_cayley_of_zero_is_identity():
    assert torch.allclose(cayley(torch.zeros(6, 6)), torch.eye(6))


def test_cayley_is_orthogonal_in_fp32_at_block_size_64():
    A = skew(0.1 * torch.randn(16, 64, 64, generator=_gen()))
    U = cayley(A)
    err = (U @ U.transpose(-1, -2) - torch.eye(64)).abs().max()
    assert err < 1e-5, f"orthogonality error {err:.2e}"
    assert torch.allclose(torch.linalg.det(U), torch.ones(16), atol=1e-4)


def test_cayley_inverse_is_cayley_of_negative():
    A = skew(torch.randn(8, 8, generator=_gen(1)))
    assert torch.allclose(cayley(-A), cayley(A).T, atol=1e-5)


def test_cayley_preserves_batch_shape():
    A = skew(torch.randn(2, 3, 4, 4, generator=_gen(2)))
    assert cayley(A).shape == (2, 3, 4, 4)


@pytest.mark.parametrize("r", [1, 2, 4])
def test_lowrank_skew_is_skew_with_rank_at_most_2r(r):
    U = torch.randn(16, r, generator=_gen(3), dtype=torch.float64)
    V = torch.randn(16, r, generator=_gen(4), dtype=torch.float64)
    A = lowrank_skew(U, V)
    assert torch.allclose(A, -A.T)
    assert torch.linalg.matrix_rank(A) <= 2 * r


@pytest.mark.parametrize("r", [1, 2, 4])
def test_lowrank_apply_matches_dense(r):
    b = 64
    U = 0.3 * torch.randn(b, r, generator=_gen(5))
    V = 0.3 * torch.randn(b, r, generator=_gen(6))
    x = torch.randn(10, b, generator=_gen(7))
    dense = x @ cayley(lowrank_skew(U, V)).T
    fast = cayley_lowrank_apply(U, V, x)
    assert fast.shape == x.shape
    assert torch.allclose(fast, dense, atol=1e-5)


def test_lowrank_apply_preserves_norm():
    U = torch.randn(32, 2, generator=_gen(8))
    V = torch.randn(32, 2, generator=_gen(9))
    x = torch.randn(5, 32, generator=_gen(10))
    y = cayley_lowrank_apply(U, V, x)
    assert torch.allclose(y.norm(dim=-1), x.norm(dim=-1), rtol=1e-4)


def test_rank2_cayley_is_a_single_plane_rotation():
    """Section 4.2 of the proposal: a rank-2 generator only rotates span{u, v}."""
    u = torch.randn(64, 1, generator=_gen(11), dtype=torch.float64)
    v = torch.randn(64, 1, generator=_gen(12), dtype=torch.float64)
    C = cayley(lowrank_skew(u, v))
    assert torch.linalg.matrix_rank(C - torch.eye(64, dtype=torch.float64)) == 2
