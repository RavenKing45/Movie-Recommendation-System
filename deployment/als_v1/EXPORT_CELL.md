# Export the ALS V1 bundle with optional TMDB posters

Run this as a **new code cell** in the executed `07_hybrid_recommender.ipynb` after the train-only ALS fit creates `Q_als`. Run the notebook's setup/data-loading cells first so `S`, `NAME`, `train`, `REL`, and `n_items` exist.

This exports the ALS item factors and catalogue in matching row order. It also looks for poster paths in the existing local TMDB API cache (`data/raw/tmdb_api_cache/`). It makes **no API requests**, needs no TMDB token, and keeps the posters optional: any movie without a validated cached poster will use a designed CineMatch placeholder in the app.

```python
from pathlib import Path
import json
import numpy as np
import pandas as pd

# Resolve the project root whether this notebook was launched from the root
# directory or from the notebooks/ directory.
PROJECT_ROOT = Path.cwd().resolve()
if PROJECT_ROOT.name.lower() == "notebooks":
    PROJECT_ROOT = PROJECT_ROOT.parent
if not (PROJECT_ROOT / "data").exists() and (PROJECT_ROOT.parent / "data").exists():
    PROJECT_ROOT = PROJECT_ROOT.parent

# These variables must come from the current train-only ALS fit.
assert "Q_als" in globals(), "Run the train-only ALS refit cell first."
assert "S" in globals() and "train" in globals()
assert "NAME" in globals() and "REL" in globals() and "n_items" in globals()

movies_path = PROJECT_ROOT / "data" / "interim" / NAME / "movies.parquet"
links_path = PROJECT_ROOT / "data" / "raw" / "ml-32m" / "links.csv"
tmdb_cache_dir = PROJECT_ROOT / "data" / "raw" / "tmdb_api_cache"

movies = pd.read_parquet(movies_path)
links = pd.read_csv(links_path, usecols=["movieId", "tmdbId"])

item_meta = (
    S["item_map"][["movieId", "i"]]
    .merge(movies[["movieId", "title", "genres"]], on="movieId", how="left", validate="one_to_one")
    .merge(links, on="movieId", how="left", validate="one_to_one")
    .sort_values("i")
    .reset_index(drop=True)
)

assert len(item_meta) == n_items == Q_als.shape[0]
assert np.array_equal(item_meta["i"].to_numpy(dtype=np.int64), np.arange(n_items))
assert np.isfinite(Q_als).all()

# Popularity fallback uses positive interactions from the training split only.
positive_train = train.loc[train["rating"] >= REL, "i"].to_numpy(dtype=np.int64)
positive_counts = np.bincount(positive_train, minlength=n_items).astype(np.int32)

# Read poster_path only from a local cache record whose TMDB ID and expected
# enriched payload structure validate. This loop does not access the network.
poster_by_tmdb_id = {}
tmdb_values = pd.to_numeric(item_meta["tmdbId"], errors="coerce")
valid_tmdb_ids = sorted(
    tmdb_values.loc[tmdb_values.gt(0) & (tmdb_values % 1 == 0)]
    .astype("int64").unique().tolist()
)

for tmdb_id in valid_tmdb_ids:
    cache_path = tmdb_cache_dir / f"{tmdb_id}.json"
    poster_url = ""
    try:
        with cache_path.open("r", encoding="utf-8") as f:
            payload = json.load(f)
        poster_path = payload.get("poster_path") if isinstance(payload, dict) else None
        if (
            isinstance(payload, dict)
            and payload.get("id") == int(tmdb_id)
            and isinstance(payload.get("credits"), dict)
            and isinstance(payload.get("keywords"), dict)
            and isinstance(poster_path, str)
            and poster_path.startswith("/")
            and not poster_path.startswith("//")
        ):
            poster_url = f"https://image.tmdb.org/t/p/w500{poster_path}"
    except (OSError, json.JSONDecodeError, TypeError, ValueError):
        pass
    poster_by_tmdb_id[int(tmdb_id)] = poster_url

poster_urls = []
for value in tmdb_values:
    if pd.isna(value) or float(value) <= 0 or not float(value).is_integer():
        poster_urls.append("")
    else:
        poster_urls.append(poster_by_tmdb_id.get(int(value), ""))

bundle_path = PROJECT_ROOT / "deployment" / "als_v1" / "models" / "als_v1_bundle.npz"
bundle_path.parent.mkdir(parents=True, exist_ok=True)
np.savez_compressed(
    bundle_path,
    Q=Q_als.astype(np.float32),
    movie_ids=item_meta["movieId"].to_numpy(dtype=np.int64),
    titles=item_meta["title"].fillna("Unknown title").astype(str).to_numpy(dtype=str),
    genres=item_meta["genres"].fillna("").astype(str).to_numpy(dtype=str),
    positive_counts=positive_counts,
    poster_urls=np.asarray(poster_urls, dtype=str),
)

print(f"Saved: {bundle_path.resolve()}")
print(f"Item factors: {Q_als.shape}; catalogue rows: {len(item_meta):,}")
print(f"Cached poster URLs: {sum(bool(x) for x in poster_urls):,} / {len(poster_urls):,}")
print(f"Bundle size: {bundle_path.stat().st_size / (1024**2):.2f} MiB")
```

Before publishing, confirm `Q_als` came from the intended **train-only** fit (ratings >= `REL`, 64 factors, alpha 5, regularization 80, 10 iterations). Keep the test set sealed until final evaluation; this export cell must not refit the model or load validation/test interactions.
