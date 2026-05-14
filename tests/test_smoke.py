"""Smoke tests for the v0.0.1 scaffold.

The views render under Streamlit; the smoke tests cover the pure-computation
primitives (Hilbert path, box counting) and confirm the stubs are present.
"""
from __future__ import annotations

import numpy as np

import sys
sys.path.insert(0, "src")

from viz import __version__
from viz.box_counting import box_count, box_counting_dimension
from viz.hilbert_d2 import d2xy, hilbert_path, xy2d


def test_version_pinned() -> None:
    assert __version__ == "0.0.1"


def test_hilbert_path_order_2_has_16_points() -> None:
    path = hilbert_path(2)
    assert path.shape == (16, 2)
    assert path.min() == 0
    assert path.max() == 3


def test_hilbert_d2xy_xy2d_round_trip() -> None:
    order = 4
    n = 1 << order
    for d in range(n * n):
        x, y = d2xy(order, d)
        assert xy2d(order, x, y) == d


def test_hilbert_path_is_a_permutation_of_grid() -> None:
    order = 3
    path = hilbert_path(order)
    n = 1 << order
    seen = {(int(p[0]), int(p[1])) for p in path}
    assert len(seen) == n * n


def test_box_counting_dimension_on_uniform_2d_close_to_2() -> None:
    rng = np.random.default_rng(0)
    points = rng.uniform(0, 1, size=(3000, 2))
    d0, _, _ = box_counting_dimension(points, n_scales=12)
    assert 1.5 < d0 < 2.3


def test_box_counting_dimension_on_line_close_to_1() -> None:
    rng = np.random.default_rng(1)
    n = 2000
    t = rng.uniform(0, 1, size=n)
    points = np.column_stack([t, 0.5 + 1e-4 * rng.standard_normal(n)])
    d0, _, _ = box_counting_dimension(points, n_scales=12)
    assert 0.7 < d0 < 1.4


def test_box_count_handles_empty_bounds() -> None:
    points = np.zeros((10, 2))
    bounds = (10.0, 11.0, 10.0, 11.0)  # everything outside
    assert box_count(points, 0.1, bounds) == 0
