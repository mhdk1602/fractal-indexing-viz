"""Streamlit entry point. Sidebar selects which of the v0-v4 views to render.

Run with:
    streamlit run app.py
"""
from __future__ import annotations

import sys
from pathlib import Path

# Make ``src/viz`` importable when the app is launched by ``streamlit run app.py``
# from the repo root (e.g. on Streamlit Community Cloud, which does not run
# ``pip install -e .`` against the checkout). The local test suite injects the
# same path via conftest, so this keeps the two entry points consistent.
sys.path.insert(0, str(Path(__file__).resolve().parent / "src"))

import streamlit as st

from viz import box_counting, hilbert_d2, hnsw_lid, mfdfa_stream, selectivity_fan

VIEWS = {
    "v0 — Hilbert curve + D₂ overlay": hilbert_d2,
    "v1 — HNSW layers + LID coloring (stub)": hnsw_lid,
    "v2 — Streaming MFDFA + partitions (stub)": mfdfa_stream,
    "v3 — Selectivity error fan (stub)": selectivity_fan,
    "v4 — Box-counting walk-through": box_counting,
}


def main() -> None:
    st.set_page_config(
        page_title="Fractal Indexing Viz",
        page_icon="🌀",
        layout="wide",
    )
    st.sidebar.title("Fractal Indexing — Viz")
    st.sidebar.caption(
        "Companion to the [fractal-indexing](https://github.com/mhdk1602/hurst-aware-partitioning) "
        "research program. Source for figures in the H2 pre-registration and the H3 pivot paper."
    )
    choice = st.sidebar.radio("View", list(VIEWS.keys()), index=0)
    VIEWS[choice].render(st)


if __name__ == "__main__":
    main()
