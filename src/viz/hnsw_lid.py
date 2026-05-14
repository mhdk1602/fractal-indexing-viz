"""v1 — HNSW layer rendering with LID-coloured nodes.

Builds a small HNSW graph (via hnswlib), then visualizes the layer structure
with nodes coloured by local intrinsic dimensionality. The user can drag
sliders for M and ef_construction; the visualization rebuilds and re-renders.

Stub in v0.0.1 — implementation lands once the fractal-ann-diagnostics
descriptor library is callable as a dependency.
"""
from __future__ import annotations


def render(st):
    """Streamlit render callable."""
    st.title("v1 — HNSW layers + LID coloring")
    st.warning(
        "Stub. Implementation depends on the descriptor library in "
        "[fractal-ann-diagnostics](https://github.com/mhdk1602/fractal-ann-diagnostics) "
        "reaching v0.1.0."
    )
    st.markdown(
        "**Intended behaviour.** A small HNSW graph (∼500 nodes, 32-dimensional "
        "embeddings) is built with sliders controlling M and ef_construction. "
        "Each node is coloured by its LID (computed via the descriptor library). "
        "The layer structure is rendered as a stack of 3-D scatter plots, with "
        "edges fading between layers. Selecting a query point animates the "
        "search trajectory."
    )
    st.markdown(
        "**Why it matters.** HNSW's documentation says higher M values "
        "work better for high-intrinsic-dimensionality data, but does not say "
        "what 'high' means quantitatively. This view makes the relationship "
        "between LID, M, and recall visible at one glance."
    )
