from __future__ import annotations

import csv
import json
import os
import shutil
import tempfile
from pathlib import Path

import numpy as np


# Paths
APP_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = APP_DIR.parent.parent

BUNDLE_PATH = APP_DIR / "models" / "als_v1_bundle.npz"
LINKS_PATH = PROJECT_ROOT / "data" / "raw" / "ml-32m" / "links.csv"
CACHE_DIR = PROJECT_ROOT / "data" / "raw" / "tmdb_api_cache"

IMAGE_BASE = "https://image.tmdb.org/t/p/w500"


def positive_integer(value: object) -> int | None:
    """Convert a valid positive integer-like value safely."""
    if value is None:
        return None

    try:
        number = float(str(value).strip())
    except (TypeError, ValueError):
        return None

    if not np.isfinite(number) or number <= 0 or not number.is_integer():
        return None

    return int(number)


def main() -> None:
    # Verify required files and directories.
    for path, description in [
        (BUNDLE_PATH, "ALS model bundle"),
        (LINKS_PATH, "MovieLens links.csv"),
        (CACHE_DIR, "TMDB cache directory"),
    ]:
        if not path.exists():
            raise SystemExit(f"{description} not found: {path}")

    # Load all existing bundle arrays without retraining ALS.
    with np.load(BUNDLE_PATH, allow_pickle=False) as data:
        bundle = {
            key: np.asarray(data[key])
            for key in data.files
        }

    required = (
        "Q",
        "movie_ids",
        "titles",
        "genres",
        "positive_counts",
    )

    for key in required:
        if key not in bundle:
            raise SystemExit(f"Bundle is missing {key!r}.")

    movie_ids = np.asarray(bundle["movie_ids"], dtype=np.int64)
    n_items = len(movie_ids)

    if len(np.unique(movie_ids)) != n_items:
        raise SystemExit("Duplicate movie IDs found. Bundle unchanged.")

    for key in ("Q", "titles", "genres", "positive_counts"):
        if len(bundle[key]) != n_items:
            raise SystemExit(
                f"Length mismatch in {key!r}. Bundle unchanged."
            )

    # Map MovieLens movieId -> TMDB tmdbId.
    tmdb_by_movie: dict[int, int] = {}

    with LINKS_PATH.open(
        "r",
        encoding="utf-8-sig",
        newline="",
    ) as file:
        reader = csv.DictReader(file)

        if not reader.fieldnames or not {
            "movieId", "tmdbId"
        }.issubset(reader.fieldnames):
            raise SystemExit(
                "links.csv must contain movieId and tmdbId."
            )

        for row in reader:
            movie_id = positive_integer(row.get("movieId"))
            tmdb_id = positive_integer(row.get("tmdbId"))

            if movie_id is not None and tmdb_id is not None:
                tmdb_by_movie[movie_id] = tmdb_id

    # Read existing cache files only. No API requests.
    tmdb_ids = {
        tmdb_by_movie[int(movie_id)]
        for movie_id in movie_ids
        if int(movie_id) in tmdb_by_movie
    }

    poster_by_tmdb: dict[int, str] = {}
    cache_files_found = 0
    valid_poster_paths = 0

    for tmdb_id in tmdb_ids:
        cache_path = CACHE_DIR / f"{tmdb_id}.json"

        if not cache_path.is_file():
            poster_by_tmdb[tmdb_id] = ""
            continue

        cache_files_found += 1

        try:
            with cache_path.open(
                "r",
                encoding="utf-8",
            ) as file:
                payload = json.load(file)
        except (OSError, json.JSONDecodeError):
            poster_by_tmdb[tmdb_id] = ""
            continue

        poster_path = (
            payload.get("poster_path")
            if isinstance(payload, dict)
            else None
        )

        if (
            isinstance(payload, dict)
            and positive_integer(payload.get("id")) == tmdb_id
            and isinstance(poster_path, str)
            and poster_path.startswith("/")
            and not poster_path.startswith("//")
            and ".." not in poster_path.split("/")
        ):
            poster_by_tmdb[tmdb_id] = IMAGE_BASE + poster_path
            valid_poster_paths += 1
        else:
            poster_by_tmdb[tmdb_id] = ""

    # Preserve the exact ordering of movie_ids in the model bundle.
    poster_urls = np.asarray(
        [
            poster_by_tmdb.get(
                tmdb_by_movie.get(int(movie_id), -1),
                "",
            )
            for movie_id in movie_ids
        ],
        dtype=str,
    )

    if not np.any(poster_urls != ""):
        raise SystemExit(
            "No posters matched. The bundle has not been modified."
        )

    # Create a backup outside the repository.
    backup_path = (
        Path(tempfile.gettempdir())
        / "als_v1_bundle_pre_posters.npz"
    )
    shutil.copy2(BUNDLE_PATH, backup_path)

    # Add poster URLs while retaining every existing array.
    bundle["poster_urls"] = poster_urls

    temp_path = BUNDLE_PATH.with_name(
        "als_v1_bundle_with_posters.tmp.npz"
    )

    try:
        np.savez_compressed(temp_path, **bundle)
        os.replace(temp_path, BUNDLE_PATH)
    finally:
        if temp_path.exists():
            temp_path.unlink()

    # Verify the saved bundle.
    with np.load(BUNDLE_PATH, allow_pickle=False) as saved:
        if len(saved["poster_urls"]) != n_items:
            raise SystemExit(
                "Verification failed. Restore the backup if necessary."
            )

        if not np.array_equal(saved["movie_ids"], movie_ids):
            raise SystemExit(
                "Movie ordering changed unexpectedly."
            )

        if not np.array_equal(saved["Q"], bundle["Q"]):
            raise SystemExit(
                "ALS factors changed unexpectedly."
            )

        saved_urls = np.asarray(saved["poster_urls"], dtype=str)

    # Report coverage.
    matched = int(np.count_nonzero(saved_urls != ""))

    print("\nTMDB POSTER METADATA COMPLETE")
    print("=" * 42)
    print(f"Catalogue movies: {n_items:,}")
    print(f"Distinct TMDB IDs: {len(tmdb_ids):,}")
    print(f"Cache files found: {cache_files_found:,}")
    print(f"Valid poster paths: {valid_poster_paths:,}")
    print(f"Movies with posters: {matched:,} / {n_items:,}")
    print(f"Poster coverage: {matched / n_items:.1%}")
    print(f"\nUpdated bundle: {BUNDLE_PATH}")
    print(f"Backup: {backup_path}")
    print("\nALS factors unchanged. No API requests made.")


if __name__ == "__main__":
    main()