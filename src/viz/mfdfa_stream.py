"""v2 — Streaming MFDFA spectrum with Hurst-aware chunk boundaries.

A sliding window over the S&P 500 close-price series. The multifractal
spectrum is computed on each window and animated. Vertical lines mark
the chunk boundaries the Hurst-aware partitioner from
hurst-aware-partitioning would draw.

Stub in v0.0.1 — implementation requires both the hurst-aware-partitioning
partitioner (when its analysis path is wired at v0.2.0) and the MFDFA
library wiring.
"""
from __future__ import annotations


def render(st):
    """Streamlit render callable."""
    st.title("v2 — Streaming MFDFA + partition boundaries")
    st.warning(
        "Stub. Implementation depends on the partitioner from "
        "[hurst-aware-partitioning](https://github.com/mhdk1602/hurst-aware-partitioning) "
        "reaching v0.2.0."
    )
    st.markdown(
        "**Intended behaviour.** A sliding-window MFDFA spectrum on the S&P 500 "
        "close-price series, animated. Top panel: the raw price series with "
        "candidate chunk boundaries from the HURST-CI partitioner overlaid in "
        "one colour, fixed-daily boundaries in a second colour, variance-CUSUM "
        "in a third. Bottom panel: the multifractal spectrum f(α) for the "
        "current window, with the spectrum width annotated."
    )
    st.markdown(
        "**Why it matters.** This is the F1 figure of the H2 paper. It shows, "
        "in real time, that the candidate partitioner draws boundaries at "
        "moments when the multifractal structure of the series shifts — moments "
        "the fixed-interval baseline misses."
    )
