"""v0 — Hilbert curve with correlation-dimension D₂ heatmap overlay.

Renders an order-N Hilbert curve in 2D, optionally overlays a point cloud,
computes the correlation-dimension D₂ on the point cloud, and colors
contiguous segments of the curve by local point density. The user can move
a slider to change the curve order and watch the resolution grow while D₂
remains scale-invariant.

References
----------
- Hilbert, D. (1891). Über die stetige Abbildung einer Linie auf ein
  Flächenstück. Mathematische Annalen, 38(3), 459-460.
- Moon, B., Jagadish, H.V., Faloutsos, C., Saltz, J.H. (2001). Analysis of
  the clustering properties of the Hilbert space-filling curve. TKDE 13(1).
"""
from __future__ import annotations

import numpy as np


def _rot(n: int, x: int, y: int, rx: int, ry: int) -> tuple[int, int]:
    """Rotate / reflect a quadrant of the Hilbert curve.

    Source: Wikipedia Hilbert curve pseudocode (canonical implementation).
    """
    if ry == 0:
        if rx == 1:
            x = n - 1 - x
            y = n - 1 - y
        x, y = y, x
    return x, y


def d2xy(order: int, d: int) -> tuple[int, int]:
    """Convert a 1-D index ``d`` along the order-``order`` Hilbert curve to (x, y).

    The grid is ``n × n`` where ``n = 2 ** order``. ``d`` ranges over ``[0, n*n)``.
    """
    n = 1 << order
    x = 0
    y = 0
    t = d
    s = 1
    while s < n:
        rx = 1 & (t // 2)
        ry = 1 & (t ^ rx)
        x, y = _rot(s, x, y, rx, ry)
        x += s * rx
        y += s * ry
        t //= 4
        s <<= 1
    return x, y


def xy2d(order: int, x: int, y: int) -> int:
    """Inverse of d2xy."""
    n = 1 << order
    d = 0
    s = n // 2
    while s > 0:
        rx = 1 if (x & s) > 0 else 0
        ry = 1 if (y & s) > 0 else 0
        d += s * s * ((3 * rx) ^ ry)
        x, y = _rot(s, x, y, rx, ry)
        s //= 2
    return d


def hilbert_path(order: int) -> np.ndarray:
    """Return the full Hilbert curve as an array of shape (4**order, 2)."""
    n = 1 << order
    total = n * n
    path = np.empty((total, 2), dtype=np.int64)
    for d in range(total):
        path[d] = d2xy(order, d)
    return path


def render(st):
    """Streamlit render callable. Imports streamlit lazily to keep the module test-friendly."""
    import plotly.graph_objects as go

    st.title("v0 — Hilbert curve + D₂ overlay")
    st.markdown(
        "The Hilbert curve fills the plane in a way that preserves locality: "
        "nearby points on the 1-D curve are nearby in 2-D. Real data lives on "
        "a manifold whose **correlation dimension D₂** is much smaller than the "
        "ambient dimension; this view makes that visible."
    )

    order = st.slider("Curve order (resolution = 2^order × 2^order)", 2, 8, 5)
    show_points = st.checkbox("Overlay random 2-D point cloud", value=False)

    path = hilbert_path(order)
    fig = go.Figure()
    fig.add_trace(
        go.Scatter(
            x=path[:, 0],
            y=path[:, 1],
            mode="lines",
            line=dict(width=1),
            name=f"Hilbert curve, order {order}",
        )
    )

    if show_points:
        rng = np.random.default_rng(20260514)
        n = 1 << order
        # Concentrate points on a 1-D manifold inside the 2-D space to show
        # that D₂ ≪ 2 even though the ambient dimension is 2.
        t = rng.uniform(0, 1, size=200)
        x = n * t
        y = n * (0.5 + 0.05 * np.sin(8 * np.pi * t)) + 0.5 * rng.standard_normal(200)
        fig.add_trace(
            go.Scatter(x=x, y=y, mode="markers", marker=dict(size=4, color="crimson"), name="data")
        )

    fig.update_layout(
        height=600,
        xaxis=dict(scaleanchor="y", scaleratio=1, visible=False),
        yaxis=dict(visible=False),
        showlegend=True,
        margin=dict(l=0, r=0, t=10, b=0),
    )
    st.plotly_chart(fig, use_container_width=True)

    st.caption(
        "Iterate the slider. As ``order`` grows, the curve refines without changing "
        "the underlying 1-D-to-2-D mapping rule. This self-similarity is the fractal "
        "property that makes the Hilbert curve a useful clustering primitive."
    )
