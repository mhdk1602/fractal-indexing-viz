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
    assert __version__ == "0.1.0"


def test_new_view_modules_expose_render() -> None:
    """The v1/v2/v3 modules must surface a callable ``render`` even with no
    heavy dependencies installed. The bodies degrade gracefully via st.warning;
    here we only assert importability + attribute presence."""
    from viz import hnsw_lid, mfdfa_stream, selectivity_fan

    assert callable(hnsw_lid.render)
    assert callable(mfdfa_stream.render)
    assert callable(selectivity_fan.render)


def test_app_entrypoint_injects_src_path() -> None:
    """app.py must prepend ``src/`` to sys.path before importing ``viz`` so
    that ``streamlit run app.py`` works on Streamlit Community Cloud without
    a ``pip install -e .`` step."""
    import re
    from pathlib import Path

    app_text = Path("app.py").read_text(encoding="utf-8")
    # Two checks: the sys.path.insert call exists, and it appears *before*
    # the ``from viz`` line in the file.
    insert_match = re.search(r"sys\.path\.insert\s*\(\s*0\s*,", app_text)
    assert insert_match is not None
    from_viz_match = re.search(r"^from viz import", app_text, re.MULTILINE)
    assert from_viz_match is not None
    assert insert_match.start() < from_viz_match.start()


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
