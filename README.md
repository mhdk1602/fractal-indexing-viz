# Fractal Indexing — Visualization Spike

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python](https://img.shields.io/badge/python-3.10%2B-blue.svg)](https://www.python.org/downloads/)
[![Status](https://img.shields.io/badge/status-spike-orange.svg)](https://github.com/mhdk1602/fractal-indexing-viz)

**Interactive visualizations that make the fractal mathematics under modern indexing systems visceral and legible.**

A practitioner who uses Z-order, Hilbert clustering, H3, S2, or HNSW typically does so without seeing the fractal mathematics underneath. This Streamlit application closes that gap. Five views:

| View | Topic | Status |
|---|---|---|
| **v0** | Hilbert curve with intrinsic-dimension D₂ heatmap overlay (interactive zoom) | implemented |
| **v1** | HNSW layer rendering, nodes coloured by local intrinsic dimensionality | stub |
| **v2** | Streaming MFDFA spectrum on a rolling S&P window with Hurst-aware chunk boundaries overlaid | stub |
| **v3** | Selectivity-estimation error fan chart: PostgreSQL histogram vs. DeepDB vs. correlation-dimension estimator | stub |
| **v4** | Box-counting walk-through: boxes shrink, log-log plot grows, slope updates in real time | implemented |

**Author:** [Dineshkumar Malempati Hari](https://orcid.org/0009-0003-1036-9477).

## Purpose

Three audiences:

1. **The H2 paper.** v2 is the natural figure for Hurst-aware partitioning.
2. **The H3 pivot paper.** v1 and v3 anchor the diagnostic narrative.
3. **The teaching surface.** Chapter 14 of [python_training](https://github.com/mhdk1602/python_training) embeds the same visualizations as worked examples.

The teaching value stands independent of the research outcome. Even if H2 or H3 are falsified by the empirical run, the fact that modern indexing systems sit on top of fractal mathematics is true, observable, and worth teaching.

## Running the app

```bash
git clone https://github.com/mhdk1602/fractal-indexing-viz.git
cd fractal-indexing-viz
pip install -e .[manim,hnsw]
streamlit run app.py
```

Visit `http://localhost:8501`. The sidebar selects which view to render.

For the v4 Manim animation specifically:

```bash
manim -ql -p src/viz/box_counting.py BoxCountingScene
```

(Manim renders separately; the Streamlit view embeds the pre-rendered video.)

## Repository structure

```
fractal-indexing-viz/
├── app.py                       # Streamlit entry; sidebar routes to each view
├── src/viz/
│   ├── hilbert_d2.py            # v0 — implemented
│   ├── hnsw_lid.py              # v1 — stub
│   ├── mfdfa_stream.py          # v2 — stub
│   ├── selectivity_fan.py       # v3 — stub
│   └── box_counting.py          # v4 — implemented
├── assets/                      # static media (gitignored unless small)
├── tests/test_smoke.py
├── pyproject.toml
├── CITATION.cff
├── .zenodo.json
└── LICENSE                      # MIT
```

## Connection to the wider research program

This is the **(c)** track of the fractal-indexing program. Source repos for the two paper tracks:

- (a) **[hurst-aware-partitioning](https://github.com/mhdk1602/hurst-aware-partitioning)** — pre-registered, [DOI 10.5281/zenodo.20188013](https://doi.org/10.5281/zenodo.20188013).
- (b) **[fractal-ann-diagnostics](https://github.com/mhdk1602/fractal-ann-diagnostics)** — H3 pivot, descriptors implemented, recommender pending.
- (c) **this repo** — visualizations that source figures for both papers and the teaching artifact.

## License

[MIT](./LICENSE).
