"""v2 — Streaming MFDFA spectrum with Hurst-aware chunk boundaries.

Uses the synthetic fGn corpus from ``hurst_partitioning.io.make_fgn_corpus``
together with the registered ``HurstCIPartitioner``. The view picks the
series in the corpus with the highest registered ``true_H`` (most persistent),
runs the partitioner to obtain boundaries, and overlays them on the series
together with a rolling DFA(2) Hurst estimate and its 95% bootstrap CI band.

Optional heavy dependencies (``nolds``, ``MFDFA``, ``hurst_partitioning``)
are wrapped in try/except so the view degrades to the raw series + an
``st.warning`` if the upstream package fails to install.
"""
from __future__ import annotations

import numpy as np


def _rolling_dfa_with_ci(
    values: np.ndarray,
    window: int,
    step: int,
    n_bootstrap: int = 30,
    rng_seed: int = 0,
):
    """Roll DFA(2) across a series and return (positions, point, ci_lo, ci_hi).

    Heavy dependencies are imported lazily inside the function so that an
    ImportError can be caught upstream without breaking the render.
    """
    from hurst_partitioning.estimators import _dfa2, block_bootstrap_ci

    rng = np.random.default_rng(rng_seed)
    n = values.size
    positions: list[int] = []
    points: list[float] = []
    los: list[float] = []
    his: list[float] = []
    t = window
    while t <= n:
        win = values[t - window : t]
        try:
            point = float(_dfa2(win))
            lo, hi = block_bootstrap_ci(
                win, _dfa2, n_bootstrap=n_bootstrap, level=0.95, rng=rng,
            )
        except Exception:
            t += step
            continue
        positions.append(t)
        points.append(point)
        los.append(lo)
        his.append(hi)
        t += step
    return (
        np.asarray(positions, dtype=int),
        np.asarray(points, dtype=float),
        np.asarray(los, dtype=float),
        np.asarray(his, dtype=float),
    )


def render(st):
    """Streamlit render callable. All heavy imports are local and guarded."""
    import plotly.graph_objects as go
    from plotly.subplots import make_subplots

    st.title("v2 — Streaming MFDFA + partition boundaries")
    st.markdown(
        "A synthetic fGn series from the registered ``hurst-partitioning`` D3 "
        "corpus (seed=20260514). The HURST-CI partitioner places a boundary "
        "wherever the rolling DFA(2) Hurst estimate's bootstrap CI shifts "
        "discontinuously. On a single-H fGn series we expect *no* boundaries, "
        "which is the negative-control behaviour the H2 pre-registration relies on."
    )

    window = st.select_slider("Rolling window length", options=[200, 500, 1000], value=500)

    # Pull the corpus and pick the most persistent series.
    series_values: np.ndarray | None = None
    true_H: float | None = None
    series_id: int | None = None
    try:
        from hurst_partitioning.io import make_fgn_corpus

        corpus = make_fgn_corpus(seed=20260514, n_series=4)
        idx = int(np.argmax([s.attrs.get("true_H", -1) for s in corpus]))
        s = corpus[idx]
        series_values = np.asarray(s.values, dtype=float).ravel()
        true_H = float(s.attrs["true_H"])
        series_id = int(s.attrs.get("series_id", idx))
    except Exception as exc:  # pragma: no cover — env-specific
        st.warning(
            "hurst-partitioning could not be imported "
            f"({type(exc).__name__}: {exc}). Plotting a synthetic Brownian "
            "path as a fallback. On Streamlit Community Cloud, ensure the "
            "package is listed in requirements.txt."
        )
        rng = np.random.default_rng(20260514)
        series_values = np.cumsum(rng.standard_normal(4096))

    # Optionally compute boundaries via the partitioner.
    boundaries: np.ndarray = np.array([0], dtype=int)
    try:
        from hurst_partitioning.partitioner import HurstCIPartitioner

        part = HurstCIPartitioner(
            window=int(window), step=100, n_bootstrap=30, rng_seed=0,
        ).fit(__import__("pandas").Series(series_values))
        boundaries = np.asarray(part.boundaries, dtype=int)
    except Exception as exc:  # pragma: no cover — env-specific
        st.warning(
            "Partitioner unavailable in this environment "
            f"({type(exc).__name__}: {exc}). Showing the raw series only."
        )

    # Optionally compute the rolling DFA Hurst trace.
    positions = np.array([], dtype=int)
    points = los = his = np.array([], dtype=float)
    try:
        positions, points, los, his = _rolling_dfa_with_ci(
            series_values, window=int(window), step=100, n_bootstrap=30, rng_seed=0,
        )
    except Exception as exc:  # pragma: no cover — env-specific
        st.warning(
            "Rolling DFA(2) unavailable in this environment "
            f"({type(exc).__name__}: {exc}). Showing the raw series only."
        )

    # Compose the two-panel figure.
    fig = make_subplots(rows=2, cols=1, shared_xaxes=True, row_heights=[0.55, 0.45])
    fig.add_trace(
        go.Scatter(
            y=series_values, mode="lines",
            line=dict(width=1, color="#4A6FA5"),
            name=(
                f"fGn series id={series_id}, true H={true_H:.2f}"
                if true_H is not None else "series"
            ),
        ),
        row=1, col=1,
    )
    for b in boundaries[1:]:  # skip the implicit start
        fig.add_vline(
            x=int(b), line=dict(color="crimson", dash="dash", width=1),
            row=1, col=1,
        )

    if positions.size > 0:
        # CI band as a filled area between his (upper) and los (lower).
        fig.add_trace(
            go.Scatter(
                x=positions, y=his, mode="lines",
                line=dict(width=0), showlegend=False, hoverinfo="skip",
            ),
            row=2, col=1,
        )
        fig.add_trace(
            go.Scatter(
                x=positions, y=los, mode="lines",
                line=dict(width=0), fill="tonexty",
                fillcolor="rgba(74,111,165,0.2)",
                name="95% bootstrap CI",
            ),
            row=2, col=1,
        )
        fig.add_trace(
            go.Scatter(
                x=positions, y=points, mode="lines+markers",
                line=dict(color="#4A6FA5"), name="rolling DFA(2) Ĥ",
            ),
            row=2, col=1,
        )
        if true_H is not None:
            fig.add_hline(
                y=true_H, line=dict(color="black", dash="dot"),
                annotation_text=f"true H = {true_H:.2f}",
                annotation_position="top right",
                row=2, col=1,
            )

    fig.update_layout(
        height=620, margin=dict(l=0, r=0, t=30, b=0), showlegend=True,
    )
    fig.update_yaxes(title_text="value", row=1, col=1)
    fig.update_yaxes(title_text="Ĥ (DFA(2))", row=2, col=1)
    fig.update_xaxes(title_text="sample index", row=2, col=1)

    st.plotly_chart(fig, use_container_width=True)

    n_internal_bounds = max(0, int(boundaries.size - 1))
    st.caption(
        f"Boundaries placed by HurstCIPartitioner(window={window}, step=100): "
        f"{n_internal_bounds}. On a single-H fGn series we expect ≈ 0 "
        "boundaries (the registered H2 negative-control behaviour)."
    )
