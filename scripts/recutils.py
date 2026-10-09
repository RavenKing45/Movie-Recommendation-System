## Shared helpers: load the split, rank items, compute top-K metrics.
## Plain module, no packaging. Notebooks import it via sys.path
from pathlib import Path

import numpy as np
import pandas as pd


def load_split(name: str = "sample_10pct") -> dict:
    root = Path(__file__).resolve().parents[1] / "data" / "processed" / name
    keys = ("train", "val", "test", "user_map", "item_map")
    return {k: pd.read_parquet(root / f"{k}.parquet") for k in keys}


def to_sets(df: pd.DataFrame) -> dict:
    """user index -> set of item indices."""
    return df.groupby("u")["i"].agg(lambda s: set(s.tolist())).to_dict()


def rank_topk(score_fn, users, seen_csr, k: int = 10, batch: int = 2000) -> dict:
    """Top-k unseen items per user.

    score_fn(batch_of_users) -> (batch, n_items) score array, higher = better.
    seen_csr: sparse (n_users, n_items) matrix; its nonzeros are masked out.
    """
    users = np.asarray(list(users))
    ranked = {}
    for s in range(0, len(users), batch):
        ub = users[s:s + batch]
        scores = np.asarray(score_fn(ub), dtype=np.float32)
        seen = seen_csr[ub].tocoo()
        scores[seen.row, seen.col] = -np.inf
        part = np.argpartition(-scores, k, axis=1)[:, :k]
        order = np.argsort(-np.take_along_axis(scores, part, axis=1), axis=1)
        top = np.take_along_axis(part, order, axis=1)
        for u, row in zip(ub, top):
            ranked[int(u)] = row.tolist()
    return ranked


def evaluate_topk(ranked: dict, relevant: dict, n_items: int, k: int = 10) -> dict:
    discounts = 1 / np.log2(np.arange(2, k + 2))
    recalls, ndcgs, hits, shown = [], [], [], set()
    for u, rel in relevant.items():
        top = ranked[u][:k]
        shown.update(top)
        gains = np.array([m in rel for m in top], dtype=float)
        dcg = (gains * discounts[:len(top)]).sum()
        idcg = discounts[:min(len(rel), k)].sum()
        recalls.append(gains.sum() / len(rel))
        ndcgs.append(dcg / idcg)
        hits.append(float(gains.sum() > 0))
    return {f"Recall@{k}": np.mean(recalls), f"NDCG@{k}": np.mean(ndcgs),
            f"HitRate@{k}": np.mean(hits), "Coverage": len(shown) / n_items}