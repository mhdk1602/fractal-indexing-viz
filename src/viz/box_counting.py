"""v4 — Box-counting walk-through.

Implements 2-D box counting and renders the live log-log fit. The Streamlit
view animates box-size reductions and updates the slope estimate of D₀
(the box-counting fractal dimension) in real time.

A separate Manim scene is sketched for the higher-production-value teaching
artifact; that lives behind the ``manim`` optional dependency.

References
----------
- Mandelbrot, B. (1967). How long is the coast of Britain? Statistical
  self-similarity and fractional dimension. Science.
- Theiler, J. (1990). Estimating fractal dimension. JOSA A 7(6).
"""
from __future__ import annotations

import numpy as np


def box_count(points: np.ndarray, box_size: float, bounds: tuple[float, float, float, float]) -> int:
    """Count the number of 2-D boxes of side ``box_size`` that contain at least one point.

    ``bounds`` is (xmin, xmax, ymin, ymax). Points outside the bounds are ignored.
    """
    xmin, xmax, ymin, ymax = bounds
    mask = (
        (points[:, 0] >= xmin)
        & (points[:, 0] < xmax)
        & (points[:, 1] >= ymin)
        & (points[:, 1] < ymax)
    )
    if not mask.any():
        return 0
    p = points[mask]
    ix = np.floor((p[:, 0] - xmin) / box_size).astype(np.int64)
    iy = np.floor((p[:, 1] - ymin) / box_size).astype(np.int64)
    # Pack (ix, iy) into a unique key to count distinct boxes
    keys = ix * (10**7) + iy
    return int(np.unique(keys).size)


def box_counting_dimension(
    points: np.ndarray,
    n_scales: int = 20,
    bounds: tuple[float, float, float, float] | None = None,
) -> tuple[float, np.ndarray, np.ndarray]:
    """Estimate the box-counting dimension D₀ via log-log fit.

    Returns
    -------
    D0 : float
        Box-counting dimension (slope of -log N(r) vs log(1/r)).
    log_inv_r : ndarray
    log_n : ndarray
        The fitted points.
    """
    if bounds is None:
        bounds = (
            float(points[:, 0].min()),
            float(points[:, 0].max()) + 1e-9,
            float(points[:, 1].min()),
            float(points[:, 1].max()) + 1e-9,
        )
    extent = max(bounds[1] - bounds[0], bounds[3] - bounds[2])
    sizes = np.geomspace(extent / 2, extent / (2 ** (n_scales - 1)), n_scales)
    raw_counts = np.array([box_count(points, s, bounds) for s in sizes])
    counts = np.clip(raw_counts, 1, None)
    log_inv_r = np.log(1.0 / sizes)
    log_n = np.log(counts)
    # Restrict the fit to the linear scaling region: drop the saturated tail
    # where N(r) approaches the point count (no more new boxes to fill) and
    # drop the trivial head where N(r) ≤ 1. This is more robust than a fixed
    # middle-60% window, which leaks into the saturated regime for sparse
    # point sets in their embedding dimension.
    n_pts = len(points)
    in_scaling = (raw_counts > 1) & (raw_counts < 0.5 * n_pts)
    if int(in_scaling.sum()) >= 2:
        slope, _ = np.polyfit(log_inv_r[in_scaling], log_n[in_scaling], 1)
    else:
        # Fall back to the middle-60% window if the adaptive mask is too narrow.
        lo = int(0.2 * n_scales)
        hi = max(int(0.8 * n_scales), lo + 2)
        slope, _ = np.polyfit(log_inv_r[lo:hi], log_n[lo:hi], 1)
    return float(slope), log_inv_r, log_n


def render(st):
    """Streamlit render callable."""
    import plotly.graph_objects as go

    st.title("v4 — Box-counting walk-through")
    st.markdown(
        "Pick a synthetic shape and watch the box-counting dimension converge as "
        "box size shrinks. D₀ is the slope of log N(r) versus log(1/r) in the "
        "scaling region."
    )

    shape = st.selectbox(
        "Shape",
        ["Uniform 2-D (expect D₀ ≈ 2)", "Line segment (expect D₀ ≈ 1)", "Cantor-like point set (expect D₀ ≈ log 2 / log 3 ≈ 0.63)"],
    )
    n_scales = st.slider("Number of scales", 8, 32, 16)

    rng = np.random.default_rng(20260514)
    if shape.startswith("Uniform"):
        points = rng.uniform(0, 1, size=(2000, 2))
    elif shape.startswith("Line"):
        t = rng.uniform(0, 1, size=2000)
        points = np.column_stack([t, 0.5 + 0.005 * rng.standard_normal(2000)])
    else:
        # 1-D Cantor set sampled in x, fixed y. Iterative middle-third removal.
        n_iter = 8
        xs = np.array([0.0])
        ends = np.array([1.0])
        for _ in range(n_iter):
            third = (ends - xs) / 3
            xs = np.concatenate([xs, xs + 2 * third])
            ends = np.concatenate([xs[: len(third) * 0 + len(xs) // 2] + third, ends])
            xs = np.sort(xs)
            ends = np.sort(ends)
        # Sample within the surviving intervals
        n_samp = 2000
        which = rng.integers(0, len(xs), size=n_samp)
        within = rng.uniform(0, 1, size=n_samp)
        xx = xs[which] + within * (ends[which] - xs[which])
        points = np.column_stack([xx, 0.5 * np.ones(n_samp)])

    d0, log_inv_r, log_n = box_counting_dimension(points, n_scales=n_scales)
    st.metric("Box-counting dimension D₀", f"{d0:.3f}")

    col_left, col_right = st.columns(2)
    with col_left:
        st.markdown("**Point set**")
        fig_pts = go.Figure(
            go.Scatter(x=points[:, 0], y=points[:, 1], mode="markers", marker=dict(size=3))
        )
        fig_pts.update_layout(
            height=400,
            margin=dict(l=0, r=0, t=0, b=0),
            xaxis=dict(scaleanchor="y", scaleratio=1),
        )
        st.plotly_chart(fig_pts, use_container_width=True)
    with col_right:
        st.markdown("**log N(r) vs log(1/r)**")
        fig_fit = go.Figure()
        fig_fit.add_trace(
            go.Scatter(x=log_inv_r, y=log_n, mode="markers+lines", name="counts")
        )
        # Best-fit line
        slope, intercept = np.polyfit(log_inv_r, log_n, 1)
        fig_fit.add_trace(
            go.Scatter(
                x=log_inv_r,
                y=slope * log_inv_r + intercept,
                mode="lines",
                line=dict(dash="dash"),
                name=f"slope D₀ ≈ {d0:.3f}",
            )
        )
        fig_fit.update_layout(
            height=400, margin=dict(l=0, r=0, t=0, b=0), xaxis_title="log(1/r)", yaxis_title="log N(r)"
        )
        st.plotly_chart(fig_fit, use_container_width=True)
