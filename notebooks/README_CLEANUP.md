# Notebook cleanup and execution record

The research workflow has been reduced from 11 active notebooks to five. This file explains the new order, preserves the original work and records the local execution check.

## Main notebook sequence

1. **01_eda.ipynb** — dataset exploration.
2. **02_split_and_baselines.ipynb** — filtering, indexing, temporal splits and ranking baselines.
3. **03_collaborative_filtering.ipynb** — explicit matrix factorization and implicit ALS (merged from original notebooks 03 and 04).
4. **04_content_based.ipynb** — genre/decade content features and feature ablation (cleaned from original notebook 05).
5. **05_final_hybrid_evaluation.ipynb** — ALS, LightFM and the selected rank-gated warm-tail reranker (cleaned from original notebook 10).

## Execution check

The five uploaded, locally executed notebook files were inspected on 30 October 2026. Every code cell had an execution count, and no code cell contained a recorded execution error.

| Notebook | Code cells | Executed | Recorded errors |
|---|---:|---:|---:|
| 01 — EDA | 10 | 10 | 0 |
| 02 — Data and baselines | 8 | 8 | 0 |
| 03 — Collaborative filtering | 15 | 15 | 0 |
| 04 — Content-based | 22 | 22 | 0 |
| 05 — Final hybrid evaluation | 18 | 18 | 0 |
| **Total** | **73** | **73** | **0** |

This check is based on the saved outputs and execution counts in the uploaded notebooks. It is not a claim that the notebooks were rerun in the GitHub environment. A fresh clone still needs the local MovieLens data, split files and generated feature artifacts.

## Dependencies and execution order

Run the notebooks from the project's **notebooks/** directory. Notebook 04 writes **data/interim/hybrid_features/content_v1.npz**, which Notebook 05 loads. Notebook 05 also requires LightFM 1.17:

```bash
python -m pip install lightfm==1.17
```

The deployed Streamlit application remains ALS-only and does not require LightFM.

## Why the notebook count changed

- Original notebooks 03 and 04 were merged because they share the data split and evaluation foundation. Their explicit and implicit ALS update equations remain separate because they optimize different objectives.
- Original notebook 05 now has one shared content-feature/profile/scoring pipeline for its feature ablation.
- Original notebooks 06–09 and 11 are preserved as research history. Their full implementations are in the archive; their decisions and limitations are summarized in **reports/experiment_decisions.md**.
- Original notebook 10 was shortened to the ALS reference, LightFM Identity and Content-v1 comparisons, validation selection, the final rank-gated reranking experiment and its frozen test evaluation.
- Debugging, introspection, repeated setup, duplicate functions and exploratory code that did not affect a model decision were removed from the main reading path.

## Original notebooks

All 11 original notebooks are retained under **notebooks/archive/original_notebooks/**. They are historical copies; the five notebooks directly under **notebooks/** are the active, cleaned sequence.

## Important interpretation

The selected research method is **ALS with a selective LightFM Content-v1 warm-tail reranker**. It is not a standalone LightFM replacement or an ordinary ALS/content score blend. The 500-rank gate improves warm-tail discovery in the final test sample while its overall NDCG and Recall confidence intervals cross zero. The deployed Streamlit application remains ALS-only.
