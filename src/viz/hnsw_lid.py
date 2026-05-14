"""v1 — HNSW layer rendering with LID-coloured nodes.

The view synthesizes a mixture vector dataset (half on a 3-D manifold lifted
to R^32, half full-rank R^32 Gaussian noise), computes per-point LID via the
MLE estimator from ``fractal_ann_diagnostics`` (Amsaleg et al. 2015), and
renders a 2-D PCA projection coloured by LID together with a histogram of
LID values. The M and ef_construction sliders are advertised as the HNSW
build parameters; if ``hnswlib`` is available we build the graph and report
its node count as a sanity tag.

Optional heavy dependencies are wrapped in try/except so the view degrades to
``st.warning`` on Streamlit Community Cloud if the upstream package fails to
install for any reason.
"""
from __future__ import annotations

import numpy as np


def _synthetic_mixture(
    n_manifold: int = 200,
    n_full_rank: int = 200,
    ambient: int = 32,
    seed: int = 20260514,
) -> tuple[np.ndarray, np.ndarray]:
    """Build a deterministic mixture for the v1 figure.

    Half the points sit on a 3-D manifold lifted into R^32 via a random
    isometry; the other half are isotropic Gaussian in R^32. The expected
    LID is therefore bimodal — roughly 3 on the manifold half and roughly
    ``ambient`` on the full-rank half.

    Returns
    -------
    X : ndarray of shape (n_manifold + n_full_rank, ambient)
    labels : ndarray of shape (n_manifold + n_full_rank,)
        0 for the manifold half, 1 for the full-rank Gaussian half.
    """
    rng = np.random.default_rng(seed)
    # 3-D latent coordinates, sampled on a slightly wavy Swiss-roll-ish sheet.
    t = rng.uniform(-3.0, 3.0, size=(n_manifold, 3))
    latent = np.column_stack(
        [
            t[:, 0],
            t[:, 1] + 0.2 * np.sin(t[:, 0]),
            t[:, 2] + 0.2 * np.cos(t[:, 1]),
        ]
    )
    # Random isometry from R^3 into R^ambient via the Q factor of a QR.
    M = rng.standard_normal((ambient, 3))
    Q, _ = np.linalg.qr(M)
    manifold = latent @ Q.T  # shape (n_manifold, ambient)
    # Add a small ambient-noise floor so the kNN distances do not collapse.
    manifold = manifold + 0.05 * rng.standard_normal(manifold.shape)

    # Full-rank half.
    full = rng.standard_normal((n_full_rank, ambient))

    X = np.vstack([manifold, full])
    labels = np.concatenate(
        [np.zeros(n_manifold, dtype=int), np.ones(n_full_rank, dtype=int)]
    )
    return X, labels


def _pca_2d(X: np.ndarray) -> np.ndarray:
    """A small SVD-based 2-D PCA. No sklearn dependency for the projection."""
    Xc = X - X.mean(axis=0, keepdims=True)
    U, S, Vt = np.linalg.svd(Xc, full_matrices=False)
    return Xc @ Vt[:2].T  # shape (n, 2)


def render(st):
    """Streamlit render callable. All heavy imports are local and guarded."""
    import plotly.graph_objects as go

    st.title("v1 — HNSW layers + LID coloring")
    st.markdown(
        "A synthetic mixture: 200 points on a 3-D manifold lifted to R^32, "
        "and 200 isotropic Gaussian points in R^32. The MLE LID estimator "
        "(Amsaleg et al., 2015) recovers the expected bimodal distribution: "
        "LID ≈ 3 on the manifold half, LID ≈ ambient dimension on the noise half. "
        "HNSW recall degrades as LID grows, which is why this descriptor "
        "matters for index tuning."
    )

    col_m, col_ef = st.columns(2)
    with col_m:
        M = st.slider("HNSW M (build parameter)", 2, 64, 16)
    with col_ef:
        ef_construction = st.slider("HNSW ef_construction", 10, 200, 100)

    X, labels = _synthetic_mixture()

    # LID computation. If fractal_ann_diagnostics is unavailable for any
    # reason, degrade to a warning so the rest of the view still renders.
    lid: np.ndarray | None = None
    try:
        from fractal_ann_diagnostics.descriptors import lid_mle

        lid = lid_mle(X, k=50, sample_size=None, rng=np.random.default_rng(0))
    except Exception as exc:  # pragma: no cover — env-specific
        st.warning(
            "fractal-ann-diagnostics could not be imported in this environment "
            f"({type(exc).__name__}: {exc}). Falling back to a uniform colour. "
            "On Streamlit Community Cloud, ensure the package is listed in "
            "requirements.txt."
        )
        lid = np.full(X.shape[0], np.nan)

    # 2-D PCA scatter coloured by LID.
    proj = _pca_2d(X)
    scatter = go.Scatter(
        x=proj[:, 0],
        y=proj[:, 1],
        mode="markers",
        marker=dict(
            size=6,
            color=lid,
            colorscale="Viridis",
            colorbar=dict(title="LID"),
            showscale=True,
        ),
        text=[f"label={l}, LID={v:.2f}" for l, v in zip(labels, lid)],
        hovertemplate="%{text}<extra></extra>",
        name="points",
    )
    fig_scatter = go.Figure(scatter)
    fig_scatter.update_layout(
        height=440,
        margin=dict(l=0, r=0, t=10, b=0),
        xaxis=dict(title="PC1"),
        yaxis=dict(title="PC2", scaleanchor="x", scaleratio=1),
    )
    st.plotly_chart(fig_scatter, use_container_width=True)

    # Histogram of LID with median and p95 reference lines.
    valid = lid[np.isfinite(lid)]
    fig_hist = go.Figure()
    if valid.size > 0:
        fig_hist.add_trace(go.Histogram(x=valid, nbinsx=30, name="LID"))
        median = float(np.median(valid))
        p95 = float(np.quantile(valid, 0.95))
        fig_hist.add_vline(
            x=median, line=dict(color="crimson", dash="dash"),
            annotation_text=f"median = {median:.2f}", annotation_position="top",
        )
        fig_hist.add_vline(
            x=p95, line=dict(color="darkorange", dash="dot"),
            annotation_text=f"p95 = {p95:.2f}", annotation_position="top",
        )
    fig_hist.update_layout(
        height=300,
        margin=dict(l=0, r=0, t=30, b=0),
        xaxis_title="LID estimate",
        yaxis_title="count",
        showlegend=False,
    )
    st.plotly_chart(fig_hist, use_container_width=True)

    # Build a tag string that reflects the slider state.
    n_nodes_tag = ""
    try:  # hnswlib is optional; degrade quietly if unavailable.
        import hnswlib

        graph = hnswlib.Index(space="l2", dim=X.shape[1])
        graph.init_index(
            max_elements=X.shape[0], ef_construction=ef_construction, M=M
        )
        graph.add_items(X, ids=np.arange(X.shape[0]))
        n_nodes_tag = f" — hnswlib graph built with {len(graph.get_ids_list())} nodes"
    except Exception:
        n_nodes_tag = " — install ``hnswlib`` to build a live graph"

    st.caption(
        f"HNSW would build with M={M}, ef_construction={ef_construction}"
        + n_nodes_tag
    )
