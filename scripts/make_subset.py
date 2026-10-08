"""User-level subsample of MovieLens 32M -> data/interim/<name>/ (parquet).

Sampling users (not rows) keeps each sampled user's full rating history.
"""
import argparse
from pathlib import Path

import numpy as np
import pandas as pd

RAW = Path("data/raw/ml-32m")
INTERIM = Path("data/interim")


def main(frac: float, seed: int, name: str) -> None:
    out = INTERIM / name
    out.mkdir(parents=True, exist_ok=True)

    ratings = pd.read_csv(
        RAW / "ratings.csv",
        dtype={"userId": "int32", "movieId": "int32",
               "rating": "float32", "timestamp": "int64"},
    )

    users = np.sort(ratings["userId"].unique())
    rng = np.random.default_rng(seed)
    keep = rng.choice(users, size=int(len(users) * frac), replace=False)

    sub = ratings[ratings["userId"].isin(keep)].reset_index(drop=True)
    sub.to_parquet(out / "ratings.parquet", index=False)

    # tags belong to users, so filter them the same way
    tags = pd.read_csv(RAW / "tags.csv")
    tags[tags["userId"].isin(keep)].to_parquet(out / "tags.parquet", index=False)

    # movies/links describe items and are small, so keep them whole
    pd.read_csv(RAW / "movies.csv").to_parquet(out / "movies.parquet", index=False)
    pd.read_csv(RAW / "links.csv").to_parquet(out / "links.parquet", index=False)

    print(f"{name}: {len(keep):,} of {len(users):,} users | "
          f"{len(sub):,} ratings | {sub['movieId'].nunique():,} movies rated")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--frac", type=float, default=0.10)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--name", default="sample_10pct")
    a = p.parse_args()
    main(a.frac, a.seed, a.name)