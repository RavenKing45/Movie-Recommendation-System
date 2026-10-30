# CineMatch+ — Movie Recommendation System

**Discover movies you'll love, powered by collaborative filtering.**

🎬 **[Try CineMatch+ Live](https://movie-recommender-als-v1.streamlit.app/)**

CineMatch+ is a movie recommendation system built using the **MovieLens 32M dataset**. The project explores collaborative filtering, content-based recommendation, metadata enrichment, and hybrid recommendation strategies, with the goal of developing a robust, personalized movie discovery experience.

The project includes a deployed Streamlit application and a reproducible, notebook-driven research workflow for evaluating recommendation models.

> **Current status:** The live application still uses the Implicit ALS model trained on the 10% research sample. The research workflow has completed a held-out evaluation of a selective warm-tail reranker, but that reranker has **not** been integrated into the live application. The deployed prototype and the latest research experiment are separate artifacts.

## Features

- **Personalized recommendations:** Search for movies and add titles you like to build a taste profile.
- **Implicit ALS collaborative filtering:** Estimates a user representation from selected movies and scores the catalogue using learned item factors.
- **Cinematic interface:** A dark, black-and-red theme with a three-column movie gallery.
- **Movie posters:** Displays TMDB poster artwork where available, with fallback covers for missing posters.
- **Preference exclusion:** Movies already selected for your taste profile are excluded from the recommendations.
- **Popularity fallback:** Provides popular catalogue picks when no personalized profile is available.
- **Lightweight deployment:** The trained model bundle is loaded at runtime; model training does not happen inside the web application.

## Live Demo

**[Launch CineMatch+](https://movie-recommender-als-v1.streamlit.app/)**

The live version is an interactive prototype. Its model and catalogue reflect the current 10% research sample. The warm-tail reranking experiment described below is a research result and is not currently served by the live app.

## Technology Stack

| Component | Technology |
|---|---|
| Language | Python |
| Application | Streamlit |
| Numerical computation | NumPy, SciPy, pandas |
| Collaborative filtering | Implicit Alternating Least Squares (ALS) |
| Hybrid recommendation research | LightFM (WARP) |
| Content features | Movie genres, release decade, TMDB metadata |
| Primary dataset | MovieLens 32M |
| Poster metadata and artwork | The Movie Database (TMDB) |
| Version control | Git and GitHub |
| Hosting | Streamlit Community Cloud |

## Recommendation Model

The deployed application uses **Implicit ALS**, a collaborative-filtering approach that learns latent representations for movies from positive user–movie interactions.

For the research prototype:

- Positive interactions are defined using a rating threshold of **4.0**.
- The ALS configuration uses **64 latent factors**, `alpha = 5`, regularization `= 80`, and **10 iterations**, with seed `42`.
- The deployed model bundle contains the item-factor matrix and catalogue metadata.
- When a visitor selects movies, the application estimates a compatible user factor using the existing item factors, then ranks candidate movies by their scores.
- Selected movies are excluded from the results.
- Popularity-based recommendations provide a fallback when a personalized result cannot be generated.

The live application does not retrain ALS when a visitor uses it.

## Research and Experiments

Research is organized as a sequence of notebooks, with experiment configurations and metric outputs retained in `reports/`. The work includes exploratory analysis, popularity baselines, matrix factorization, Implicit ALS, genre/decade content features, TMDB metadata enrichment, LightFM, and full-catalogue hybrid evaluation.

### Dataset and evaluation protocol

The reported reranking experiment uses a **user-level 10% sample of MovieLens 32M** (`sample_10pct`), containing 20,093 users and a 17,241-movie catalogue. The split files contain 2,919,680 training rows, 137,197 validation rows, and 137,197 test rows.

The temporal evaluation protocol used for the final reranking comparison is:

- Relevant interactions have rating **at least 4.0**.
- Evaluation targets are restricted to timestamps on or after **2016-01-01 00:00:00 UTC**.
- Duplicate user–movie pairs are removed from the target set.
- Models are refitted using `train + validation`; each user's complete pre-test history is excluded from recommendations.
- The warm-tail catalogue slice is defined using positive interactions in `train + validation`: **1–19 positive interactions**. Strict-cold items have zero positive interactions in that history.
- Recommendations are evaluated at **K = 10**, averaged over 10 deterministic tie-break seeds.

The split audit found chronological ordering at the user level and no user–movie pair overlap between train, validation, and test. Some users have timestamp ties at split boundaries, so the protocol uses non-strict (`<=`) temporal ordering where ties occur.

### Final warm-tail reranking experiment

Notebook `10_lightfm_hybrid.ipynb` compares the ALS baseline with a selective reranking strategy. The experiment refits:

| Model | Configuration |
|---|---|
| Implicit ALS | 64 factors, `alpha = 5`, regularization `= 80`, 10 iterations, seed 42 |
| LightFM Content-v1 | WARP, 32 components, 15 epochs, learning rate `0.05`, item/user regularization `1e-6`, `max_sampled = 10`, seed 42 |

Content-v1 uses genre and release-decade features. For the reranker, ALS supplies the original top-10 list. LightFM Content-v1 scores the catalogue; the highest-ranked eligible warm-tail candidate may replace the last ALS recommendation if its full-catalogue Content-v1 rank is within the selected gate. The selected gate is **500**. A 1,000-rank gate is retained as a secondary comparison, not the selected setting.

#### Final test results

The test evaluation contains **26,658 relevant user–movie pairs across 5,831 users**. The warm-tail slice contains **1,810 relevant pairs across 1,121 users**.

| Evaluation slice / metric | ALS baseline | 500-rank gate | 1,000-rank gate |
|---|---:|---:|---:|
| Overall NDCG@10 | **0.076251** | 0.076092 | 0.075069 |
| Overall Recall@10 | 0.097556 | 0.097151 | 0.094994 |
| Overall HitRate@10 | 0.302864 | 0.302864 | 0.297548 |
| Warm-tail NDCG@10 | 0.000000 | 0.001685 | 0.002101 |
| Warm-tail Recall@10 | 0.000000 | 0.004891 | 0.006230 |
| Warm-tail HitRate@10 | 0.000000 | 0.009813 | 0.011597 |

The 500-rank gate used **2.21%** of recommendation slots for warm-tail items on average; the 1,000-rank gate used 5.43%.

#### Paired confidence intervals

The following are paired 95% user-bootstrap confidence intervals for changes relative to the ALS baseline. User metrics were averaged over the 10 tie-break seeds before bootstrapping (3,000 bootstrap resamples).

| Gate | Metric change vs. ALS | Mean change | 95% CI |
|---:|---|---:|---:|
| 500 | Overall NDCG@10 | -0.000159 | [-0.000399, 0.000071] |
| 500 | Overall Recall@10 | -0.000405 | [-0.001092, 0.000199] |
| 500 | Warm-tail Recall@10 | +0.004891 | [0.001858, 0.008727] |
| 500 | Warm-tail HitRate@10 | +0.009813 | [0.004460, 0.016057] |
| 1,000 | Overall NDCG@10 | -0.001182 | [-0.001568, -0.000801] |
| 1,000 | Overall Recall@10 | -0.002563 | [-0.003636, -0.001535] |
| 1,000 | Warm-tail Recall@10 | +0.006230 | [0.002661, 0.010645] |
| 1,000 | Warm-tail HitRate@10 | +0.011597 | [0.005352, 0.017841] |

**Interpretation:** On this test evaluation, the 500-rank gate improves warm-tail Recall@10 and HitRate@10, with confidence intervals above zero. Its overall NDCG@10 and Recall@10 changes have confidence intervals that cross zero, so the experiment does not establish a statistically clear overall-quality loss at this gate. The 1,000-rank gate produces somewhat greater warm-tail discovery but has a measurable overall NDCG@10 and Recall@10 reduction. The **500-rank gate is therefore the selected research candidate** among the settings tested.

This is a measured trade-off, not evidence that warm-tail discovery is solved: warm-tail HitRate@10 with the selected gate is still 0.98%. The experiment also does not solve strict-cold-start recommendation, where an item has no positive pre-test interaction history.

**Evaluation-scope note:** The 500-rank threshold was selected using validation results and then held fixed for this final reranking assessment. The repository also contains outputs from other model experiments, which may use different evaluation views or protocols. Test metrics from different experiments should not be compared unless their data split, time filter, relevance definition, candidate eligibility, and aggregation are aligned. The paired intervals quantify user-sampling uncertainty for this evaluation; they do not quantify variability from retraining models with different random seeds.

## Repository Structure

```text
Movie-Recommendation-System/
├── deployment/
│   └── als_v1/
│       ├── app.py
│       ├── requirements.txt
│       ├── models/
│       │   └── als_v1_bundle.npz
│       └── add_tmdb_posters.py
├── notebooks/
│   ├── 01_eda.ipynb
│   ├── 02_split_and_baselines.ipynb
│   ├── 03_matrix_factorization.ipynb
│   ├── 04_implicit_als.ipynb
│   ├── 05_content_based.ipynb
│   ├── 06_content_enrichment.ipynb
│   ├── 07_hybrid_recommender.ipynb
│   ├── 08_tmdb_enrichment.ipynb
│   ├── 09_reserved_slot_hybrid.ipynb
│   └── 10_lightfm_hybrid.ipynb
├── reports/                 # Experiment configs, validation/test metrics and diagnostics
├── scripts/
│   ├── download_data.py
│   ├── make_subset.py
│   └── recutils.py
├── data/                    # Local datasets and generated features; not committed
└── README.md
```

## Run CineMatch+ Locally

### 1. Clone the repository

```bash
git clone https://github.com/RavenKing45/Movie-Recommendation-System.git
cd Movie-Recommendation-System
```

### 2. Create and activate a virtual environment

On Linux or WSL:

```bash
cd deployment/als_v1

python -m venv .venv
source .venv/bin/activate
```

On Windows PowerShell, activate it using:

```powershell
.\.venv\Scripts\Activate.ps1
```

### 3. Install dependencies

```bash
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

### 4. Launch the application

```bash
python -m streamlit run app.py
```

Open the local URL printed by Streamlit, usually `http://localhost:8501`.

The committed deployment bundle allows the application to run without downloading MovieLens data or retraining ALS. The research notebooks and poster-metadata regeneration workflow have additional local data requirements.

## Data and Reproducibility

The repository contains notebooks, scripts, experiment configurations, and evaluation outputs. Large raw datasets, generated local caches, and virtual environments are excluded from version control.

To reproduce the research sample from a local MovieLens 32M download, the scripts in `scripts/` provide the data-download/checksum and user-level subsampling workflows. Run the relevant notebook with its matching report configuration and split artifacts. Results should be interpreted only alongside their corresponding data split and evaluation protocol.

The live application remains on its existing ALS bundle. Any future production-model change should be separately validated, exported, and deployed rather than assumed from the research reranking notebook.

## Acknowledgements

- **MovieLens:** For the movie ratings and metadata used in the research.
- **The Movie Database (TMDB):** For movie metadata and poster artwork. This product uses the TMDB API but is not endorsed or certified by TMDB.

## Roadmap

- Integrate the selected warm-tail reranking strategy into a separate deployment candidate and assess its product impact.
- Investigate strict-cold-start recommendations, which are not addressed by the current ALS + warm-tail reranking experiment.
- Evaluate model behavior on broader user samples and additional time periods.
- Retrain and export a production model using the intended eligible data, with a clearly documented evaluation protocol.
- Update the deployed app only after the production candidate has been validated.
- Continue improving the movie discovery experience and recommendation quality.
