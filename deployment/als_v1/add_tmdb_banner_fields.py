"""Add banner fields (wide backdrop, overview, rating) to the ALS bundle.

Reads only the local TMDB cache (data/raw/tmdb_api_cache/<tmdbId>.json),
makes no API requests, and leaves all existing bundle arrays unchanged.
"""
from __future__ import annotations

import csv
import json
import os
import shutil
import tempfile
from pathlib import Path

import numpy as np


APP_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = APP_DIR.parent.parent

BUNDLE_PATH = APP_DIR / "models" / "als_v1_bundle.npz"
LINKS_PATH = PROJECT_ROOT / "data" / "raw" / "ml-32m" / "links.csv"
CACHE_DIR = PROJECT_ROOT / "data" / "raw" / "tmdb_api_cache"

BACKDROP_BASE = "https://image.tmdb.org/t/p/w1280"


def positive_int(value: object) -> int | None:
    try:
        number = float(str(value).strip())
    except (TypeError, ValueError):
        return None
    if not np.isfinite(number) or number <= 0 or not number.is_integer():
        return None
    return int(number)


def safe_path(value: object) -> str:
    if not isinstance(value, str):
        return ""
    if not value.startswith("/") or value.startswith("//") or ".." in value.split("/"):
        return ""
    return value


def main() -> None:
    for path, label in [(BUNDLE_PATH, "bundle"), (LINKS_PATH, "links.csv"), (CACHE_DIR, "TMDB cache")]:
        if not path.exists():
            raise SystemExit(f"{label} not found: {path}")

    with np.load(BUNDLE_PATH, allow_pickle=False) as data:
        bundle = {key: np.asarray(data[key]) for key in data.files}

    movie_ids = np.asarray(bundle["movie_ids"], dtype=np.int64)
    n_items = len(movie_ids)

    tmdb_by_movie: dict[int, int] = {}
    with LINKS_PATH.open("r", encoding="utf-8-sig", newline="") as file:
        for row in csv.DictReader(file):
            movie_id = positive_int(row.get("movieId"))
            tmdb_id = positive_int(row.get("tmdbId"))
            if movie_id is not None and tmdb_id is not None:
                tmdb_by_movie[movie_id] = tmdb_id

    fields: dict[int, tuple[str, str, float]] = {}
    for tmdb_id in {tmdb_by_movie[int(m)] for m in movie_ids if int(m) in tmdb_by_movie}:
        cache_path = CACHE_DIR / f"{tmdb_id}.json"
        if not cache_path.is_file():
            continue
        try:
            payload = json.loads(cache_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            continue
        if not isinstance(payload, dict) or positive_int(payload.get("id")) != tmdb_id:
            continue

        backdrop = safe_path(payload.get("backdrop_path"))
        backdrop_url = BACKDROP_BASE + backdrop if backdrop else ""
        overview = str(payload.get("overview") or "").strip()
        try:
            rating = float(payload.get("vote_average") or 0.0)
        except (TypeError, ValueError):
            rating = 0.0
        fields[tmdb_id] = (backdrop_url, overview, rating if np.isfinite(rating) else 0.0)

    backdrops, overviews, ratings = [], [], []
    for movie_id in movie_ids:
        tmdb_id = tmdb_by_movie.get(int(movie_id), -1)
        b, o, r = fields.get(tmdb_id, ("", "", 0.0))
        backdrops.append(b)
        overviews.append(o)
        ratings.append(r)

    backdrops_arr = np.asarray(backdrops, dtype=str)
    if not np.any(backdrops_arr != ""):
        raise SystemExit("No backdrops found in the cache. The bundle has not been modified.")

    backup = Path(tempfile.gettempdir()) / "als_v1_bundle_pre_banner.npz"
    shutil.copy2(BUNDLE_PATH, backup)

    bundle["backdrop_urls"] = backdrops_arr
    bundle["overviews"] = np.asarray(overviews, dtype=str)
    bundle["vote_averages"] = np.asarray(ratings, dtype=np.float32)

    temp_path = BUNDLE_PATH.with_name("als_v1_bundle_with_banner.tmp.npz")
    try:
        np.savez_compressed(temp_path, **bundle)
        os.replace(temp_path, BUNDLE_PATH)
    finally:
        if temp_path.exists():
            temp_path.unlink()

    with np.load(BUNDLE_PATH, allow_pickle=False) as saved:
        assert np.array_equal(saved["movie_ids"], movie_ids)
        assert np.array_equal(saved["Q"], bundle["Q"])
        has_backdrop = int(np.count_nonzero(saved["backdrop_urls"] != ""))

    print("BANNER FIELDS ADDED")
    print(f"Catalogue movies: {n_items:,}")
    print(f"Movies with backdrops: {has_backdrop:,} / {n_items:,}")
    print(f"Backup: {backup}")
    print("ALS factors unchanged. No API requests made.")


if __name__ == "__main__":
    main()
