# CineMatch+ — Movie Recommendation System

**Discover movies you'll love, powered by collaborative filtering.**

🎬 **[Try CineMatch+ Live](https://movie-recommender-als-v1.streamlit.app/)**

CineMatch+ is a movie recommendation system built using the **MovieLens 32M dataset**. The project explores collaborative filtering, content-based recommendation, metadata enrichment, and hybrid recommendation strategies, with the goal of developing a robust, personalized movie discovery experience.

The project includes a deployed Streamlit application and a reproducible research workflow for evaluating recommendation models.

> **Current status:** The live application uses an Implicit ALS model trained on a 10% research sample. Model selection and evaluation are still in progress; this is not yet the final full-data model.

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

The live version is intended as an interactive prototype. Its model and catalogue reflect the current experimental sample, and recommendations may change when the final model is trained and deployed.

## Technology Stack

| Component | Technology |
|---|---|
| Language | Python |
| Application | Streamlit |
| Numerical computation | NumPy |
| Collaborative filtering | Implicit Alternating Least Squares (ALS) |
| Primary dataset | MovieLens 32M |
| Poster metadata and artwork | The Movie Database (TMDB) |
| Version control | Git and GitHub |
| Hosting | Streamlit Community Cloud |

## Recommendation Model

The current app uses **Implicit ALS**, a collaborative-filtering approach that learns latent representations for movies from positive user–movie interactions.

For the research prototype:

- Positive interactions are derived using a rating threshold of 4.0.
- The current ALS configuration uses 64 latent factors, alpha = 5, regularization = 80, and 10 iterations.
- The deployed model contains the item-factor matrix and catalogue metadata.
- When a user selects movies, the application estimates a compatible user factor using the existing item factors, then ranks candidate movies by their scores.
- Selected movies are excluded from the results.
- Popularity-based recommendations provide a fallback when a personalized result cannot be generated.

The current application does not retrain the ALS model when a visitor uses it.

## Research and Experiments

The project uses a notebook-driven workflow to investigate recommendation quality and compare alternative approaches.

Areas explored so far include:

- Exploratory data analysis and data preparation.
- Train, validation, and test splits with held-out evaluation.
- Popularity baselines and matrix factorization.
- Implicit ALS collaborative filtering.
- Genre- and metadata-based content recommendation.
- TMDB metadata enrichment.
- Collaborative–content hybrid strategies.
- LightFM hybrid recommendation models.
- Evaluation of full-catalogue recommendation quality and recommendations for less frequently interacted-with movies.

**Current research direction:** ALS is the baseline for the deployed prototype. Content-based and hybrid alternatives are being investigated using the established validation protocol. The final model choice will follow the remaining diagnostics and held-out evaluation.

Validation and test results should be interpreted in the context of the dataset sample and evaluation protocol used for each experiment.

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
├── reports/
├── scripts/
├── data/                  # Local datasets; not committed
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

The committed deployment bundle allows the application to run without downloading MovieLens data or retraining ALS. The research notebooks and the poster-metadata regeneration workflow have additional local data requirements.

## Data and Reproducibility

The repository contains notebooks, scripts, experiment outputs, and deployment code. Large raw datasets, local caches, and virtual environments are excluded from version control.

Research results should be regenerated using the relevant notebook, configuration, and documented data split. The production model will be retrained on all eligible interactions only after model selection and final evaluation are complete.

## Acknowledgements

- **MovieLens:** For the movie ratings and metadata used in the research.
- **The Movie Database (TMDB):** For movie metadata and poster artwork. This product uses the TMDB API but is not endorsed or certified by TMDB.

## Roadmap

- Complete retrieval diagnostics and model comparisons.
- Finalize model selection using the established evaluation protocol.
- Evaluate the selected model on the held-out test set.
- Retrain and export the final model using all eligible interactions.
- Update the deployed app with the production model bundle.
- Continue improving the movie discovery experience and recommendation quality.
