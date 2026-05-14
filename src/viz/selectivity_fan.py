"""v3 — Selectivity-estimation error fan chart.

Three estimators of multidimensional range-query selectivity, applied to a
single workload: PostgreSQL histogram, DeepDB-style learned estimator, and
the correlation-fractal-dimension estimator (Belussi & Faloutsos 1995). The
chart shows per-bucket error envelopes, fan-style.

Stub in v0.0.1 — requires the PostgreSQL EXPLAIN harness and a DeepDB
re-implementation or its release artefact.
"""
from __future__ import annotations


def render(st):
    """Streamlit render callable."""
    st.title("v3 — Selectivity estimation error fan")
    st.warning(
        "Stub. Implementation requires the PostgreSQL EXPLAIN harness and a "
        "DeepDB-equivalent learned cardinality estimator."
    )
    st.markdown(
        "**Intended behaviour.** For a fixed multidimensional workload "
        "(NYC taxi or an embedding workload), compute selectivity estimates "
        "under three estimators and plot per-bucket relative error as a fan "
        "chart over the workload range. The correlation-fractal-dimension "
        "estimator (Faloutsos-Kamel 1994 line of work) is the third fan."
    )
    st.markdown(
        "**Why it matters.** This is the headline figure for H4 in the master "
        "plan. It shows visually whether a 1994 fractal-dimension estimator "
        "still beats modern OLAP estimators on real workloads — the bold and "
        "controversial claim of the wider research program."
    )
