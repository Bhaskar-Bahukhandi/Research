"""Shared fixtures, and a hook that turns unimplemented exercises into skips.

A test that hits ``raise NotImplementedError`` in one of your stubs is reported
as SKIPPED ("not implemented yet") instead of FAILED, so the test run doubles
as a to-do list. Once you implement a function, its tests pass or fail for real.
"""
from __future__ import annotations

import pytest
import torch

from rtvm.synthetic import consistent_graph, random_connected_graph


@pytest.hookimpl(wrapper=True)
def pytest_runtest_call(item):
    try:
        return (yield)
    except NotImplementedError as exc:
        pytest.skip(f"not implemented yet: {exc}")


@pytest.fixture
def small_edges():
    """20 vertices, 45 edges, so 26 independent cycles."""
    return random_connected_graph(20, 45, seed=1)


@pytest.fixture
def consistent(small_edges):
    """A consistent graph with two blocks of size 8 (d = 16), plus its frames."""
    return consistent_graph(20, small_edges, n_blocks=2, b=8, seed=2)


@pytest.fixture
def psi_root():
    gen = torch.Generator().manual_seed(3)
    return torch.randn(16, generator=gen)
