# CineMatch+ — Movie Recommendation System

**A movie recommendation project built with MovieLens 32M.**

🎬 **[Try CineMatch+ Live](https://movie-recommender-als-v1.streamlit.app/)**

CineMatch+ explores collaborative filtering, content-based recommendation and hybrid ranking. The repository contains a deployed Streamlit prototype and a notebook-based research workflow.

> **Current status:** The live application uses Implicit ALS. A LightFM-based warm-tail reranker was evaluated separately in the research workflow, but it has not been integrated into the deployed application. The live prototype therefore remains ALS-only.

## What the project includes

- Personalized movie recommendations from a user's selected movies.
- Implicit ALS collaborative filtering with a popularity fallback.
- Movie search, TMDB poster artwork and exclusion of selected titles.
- Content-feature experiments using genres, release decade and external metadata.
- A validation-selected LightFM Content-v1 reranking experiment for warm-tail discovery.
- Five main research notebooks, with the earlier experimental notebooks preserved in an archive.

## Live application

**[Launch CineMatch+](https://movie-recommender-als-v1.streamlit.app/)**

The Streamlit app loads a pre-trained ALS model bundle. It does not train a model when a visitor uses the application. The model and catalogue currently reflect the project's 10% MovieLens research sample.

## Recommendation model

The deployed recommender uses **Implicit Alternating Least Squares (ALS)** trained on positive user–movie interactions.

Research configuration:

- Positive interaction threshold: rating of 4.0 or higher.
- ALS: 64 latent factors, alpha = 5, regularization lambda = 80, 10 iterations, seed 42.
- The application estimates a user factor from selected movies and ranks unseen catalogue items using the learned item factors.
- Popularity-based recommendations are used when a personalized profile cannot be formed.

## Research workflow

The main research path is intentionally limited to five notebooks:

| Notebook | Contents |
|---|---|
| 01 — EDA | Rating distribution, user/movie activity, sparsity, timestamps and metadata |
| 02 — Data and baselines | Filtering, re-indexing, chronological splits, leakage checks and ranking baselines |
| 03 — Collaborative filtering | Explicit matrix factorization benchmark and implicit-feedback ALS |
| 04 — Content-based | Genre/decade features, user profiles and feature ablation |
| 05 — Final hybrid evaluation | ALS vs LightFM, validation comparisons, rank-gated reranking and final test evaluation |

Run them in order. The original 11 notebooks are retained under **notebooks/archive/original_notebooks/** for reference. The cleanup rationale and execution check are in **README_CLEANUP.md**; archived experiment findings are summarized in **reports/experiment_decisions.md**.

### Reproducing the notebooks

The research sample is a user-level 10% subset of MovieLens 32M. The prepared split contains 20,093 users and 17,241 catalogue items:

| Split | Rows |
|---|---:|
| Train | 2,919,680 |
| Validation | 137,197 |
| Test | 137,197 |

The notebooks rely on local MovieLens data and generated intermediate artifacts, which are not committed. Use the scripts in **scripts/** and Notebook 02 to prepare the data and split files. Run Jupyter from the **notebooks/** directory so relative paths resolve.

Notebook 04 exports the frozen content-v1 item-feature matrix to **data/interim/hybrid_features/content_v1.npz**. Notebook 05 loads that file and requires LightFM 1.17 in the research Python environment:

```bash
python -m pip install lightfm==1.17
```

The root requirements file covers the project's standard dependencies; LightFM is an additional dependency for the final hybrid research notebook. It is not required by the deployed ALS application.

## Final warm-tail reranking experiment

The selected research candidate keeps the ALS top-10 ranking as its starting point. LightFM Content-v1 ranks the eligible catalogue; when the highest-ranked unseen warm-tail candidate is within rank 500, it replaces the last ALS recommendation. The gate was selected on validation and then held fixed for the test evaluation.

The test window contains 26,658 relevant user–movie pairs across 5,831 users. The warm-tail slice (1–19 positive interactions in pre-test history) contains 1,810 relevant pairs across 1,121 users.

| Test metric | ALS | 500-rank gate | 1,000-rank gate |
|---|---:|---:|---:|
| Overall NDCG@10 | **0.076251** | 0.076092 | 0.075069 |
| Overall Recall@10 | 0.097556 | 0.097151 | 0.094994 |
| Overall HitRate@10 | 0.302864 | 0.302864 | 0.297548 |
| Warm-tail NDCG@10 | 0.000000 | 0.001685 | 0.002101 |
| Warm-tail Recall@10 | 0.000000 | 0.004891 | 0.006230 |
| Warm-tail HitRate@10 | 0.000000 | 0.009813 | 0.011597 |

Paired 95% user-bootstrap intervals for the 500-rank gate:

| Metric change vs. ALS | Mean change | 95% CI |
|---|---:|---:|
| Overall NDCG@10 | -0.000159 | [-0.000399, 0.000071] |
| Overall Recall@10 | -0.000405 | [-0.001092, 0.000199] |
| Warm-tail Recall@10 | +0.004891 | [0.001858, 0.008727] |
| Warm-tail HitRate@10 | +0.009813 | [0.004460, 0.016057] |

The 500-rank gate improves warm-tail recall and hit rate in this test sample. Its overall NDCG and Recall confidence intervals cross zero, so this evaluation does not establish a statistically clear overall-quality change. Warm-tail HitRate@10 remains below 1%; this is a limited improvement, not a complete solution to rare-item discovery. Strict-cold-start recommendation remains unsolved.

Standalone LightFM candidates did not pass the project's overall-NDCG retention guardrail. The selected approach is therefore a selective reranker, not a replacement for ALS. The 1,000-rank gate recovered more warm-tail hits but showed a clearer overall-metric reduction.

The final test evaluation was run after the 500-rank gate had been selected on validation. Test results are reported as a final assessment and must not be used to retune the gate.

## Repository layout

- **deployment/als_v1/** — Streamlit app, deployment dependencies and the trained ALS bundle.
- **notebooks/** — five active research notebooks.
- **notebooks/archive/original_notebooks/** — all 11 original notebooks.
- **reports/** — model configurations, metrics, diagnostics and experiment decisions.
- **scripts/** — data download, sample creation and shared recommendation utilities.
- **data/** — local raw data, splits, caches and generated features; not committed.

## Run the application locally

From the repository root:

```bash
cd deployment/als_v1
python -m venv .venv
```

Activate the environment.

On Linux or WSL:

```bash
source .venv/bin/activate
```

On Windows PowerShell:

```powershell
.\.venv\Scripts\Activate.ps1
```

Install the deployment dependencies and launch Streamlit:

```bash
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
python -m streamlit run app.py
```

The committed ALS bundle allows the application to run without downloading MovieLens or retraining the model.

## Data and acknowledgements

Raw MovieLens data, local caches, intermediate matrices and virtual environments are excluded from version control. The repository retains the scripts, notebooks, experiment configurations and reported evaluation outputs needed to understand the work.

- **MovieLens** — ratings and movie metadata.
- **The Movie Database (TMDB)** — metadata and poster artwork. This product uses the TMDB API but is not endorsed or certified by TMDB.
