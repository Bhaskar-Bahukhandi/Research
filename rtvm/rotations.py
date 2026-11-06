"""Orthogonal block rotations for RTVM.

Conventions
-----------
A state vector has width d = n_blocks * b. A block-diagonal rotation is stored
as a tensor of shape (..., n_blocks, b, b); block j acts on coordinates
[j*b, (j+1)*b) of the state. The prototype uses d = 128, b = 64; the final
model uses d = 1024, b = 64.

Exercises (week 3): ``skew``, ``cayley``, ``lowrank_skew``, ``cayley_lowrank_apply``.
Spec: tests/test_rotations.py. Run ``pytest -m week3``.
"""
from __future__ import annotations

import torch
from torch import Tensor


def skew(P: Tensor) -> Tensor:
    """Skew-symmetric part-maker: map any square P to A with A^T = -A.

    Works on the last two dimensions, so P may carry batch dimensions.
    Hint: one line. Which combination of P and its transpose is antisymmetric?
    """
    return 0.5 * (P - P.transpose(-1, -2))


def cayley(A: Tensor) -> Tensor:
    """Cayley map U = (I - A)(I + A)^{-1} of a skew-symmetric A.

    For skew A the result is orthogonal with determinant +1 (prove this by hand
    first: it is your Wed Nov 11 exercise).

    Hints:
      * Never call ``torch.linalg.inv``. (I - A) and (I + A)^{-1} commute, so
        U = (I + A)^{-1}(I - A), which is one call to ``torch.linalg.solve``.
      * Build the identity with the same dtype/device as A and let it
        broadcast over batch dimensions.

    Shape: (..., b, b) -> (..., b, b).
    """
    size = A.shape[-1]
    eye = torch.eye(size, dtype=A.dtype, device=A.device)
    return torch.linalg.solve(eye + A, eye - A)


def lowrank_skew(U: Tensor, V: Tensor) -> Tensor:
    """Rank-2r skew generator A = U V^T - V U^T from U, V of shape (..., b, r).

    With r = 1 this is the proposal's u v^T - v u^T. Rank is at most 2r.
    """
    return U @ V.transpose(-1, -2) - V @ U.transpose(-1, -2)


def cayley_lowrank_apply(U: Tensor, V: Tensor, x: Tensor) -> Tensor:
    """Apply cayley(lowrank_skew(U, V)) to vectors x WITHOUT forming a b x b matrix.

    Shapes: U, V (..., b, r); x (..., b) -> (..., b). Cost should be O(b r + r^3).

    Derivation to follow (proposal Section 4.2):
      * Write A = W Z^T with W = [U, -V] and Z = [V, U], both (b, 2r).
      * Woodbury: (I + W Z^T)^{-1} x = x - W (I_2r + Z^T W)^{-1} Z^T x.
      * Then apply (I - A) = (I - W Z^T) to that result.
    Only a 2r x 2r system is ever solved.

    Check your algebra against the dense version in the tests, and confirm the
    corrected determinant det(I_2 + Z^T W) = 1 + |u|^2 |v|^2 - (u.v)^2 for r = 1.
    """
    U, V = torch.broadcast_tensors(U, V)
    W = torch.cat((U, -V), dim=-1)
    Z = torch.cat((V, U), dim=-1)

    # Woodbury reduces the solve to the 2r-dimensional column space.
    ZtW = Z.transpose(-1, -2) @ W
    eye = torch.eye(ZtW.shape[-1], dtype=U.dtype, device=U.device)
    rhs = Z.transpose(-1, -2) @ x.unsqueeze(-1)
    coeff = torch.linalg.solve(eye + ZtW, rhs)
    return (x.unsqueeze(-1) - 2 * (W @ coeff)).squeeze(-1)


# ---------------------------------------------------------------------------
# Provided helpers (complete)
# ---------------------------------------------------------------------------

def identity_blocks(n_blocks: int, b: int, dtype=torch.float32, device=None) -> Tensor:
    """Identity block rotation of shape (n_blocks, b, b)."""
    return torch.eye(b, dtype=dtype, device=device).expand(n_blocks, b, b).clone()


def block_apply(R: Tensor, x: Tensor) -> Tensor:
    """Apply block rotations R (..., n_blocks, b, b) to states x (..., d)."""
    n_blocks, b = R.shape[-3], R.shape[-1]
    xb = x.reshape(*x.shape[:-1], n_blocks, b)
    y = (R @ xb.unsqueeze(-1)).squeeze(-1)
    return y.reshape(*y.shape[:-2], n_blocks * b)


def block_transpose(R: Tensor) -> Tensor:
    """Transpose (= inverse, for rotations) of every block."""
    return R.transpose(-1, -2)
