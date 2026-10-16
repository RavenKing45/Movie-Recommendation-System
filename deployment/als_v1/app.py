from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import streamlit as st


APP_DIR = Path(__file__).resolve().parent
BUNDLE_PATH = APP_DIR / "models" / "als_v1_bundle.npz"
ALPHA = 5.0
REGULARIZATION = 80.0
TOP_K = 10

st.set_page_config(
    page_title="CineMatch · ALS Movie Recommendations",
    page_icon="🎬",
    layout="wide",
)


@st.cache_resource(show_spinner="Loading the ALS recommendation model…")
def load_bundle(path_string: str) -> dict[str, Any]:
    path = Path(path_string)
    if not path.exists():
        raise FileNotFoundError(
            f"ALS bundle not found at {path}. Run the export cell in "
            "EXPORT_CELL.md, then place als_v1_bundle.npz in the models/ folder."
        )

    with np.load(path, allow_pickle=False) as data:
        q = np.asarray(data["Q"], dtype=np.float32)
        movie_ids = np.asarray(data["movie_ids"], dtype=np.int64)
        titles = np.asarray(data["titles"], dtype=str)
        genres = np.asarray(data["genres"], dtype=str)
        positive_counts = np.asarray(data["positive_counts"], dtype=np.int32)

    n_items = q.shape[0]
    if q.ndim != 2 or q.shape[1] != 64:
        raise ValueError(f"Expected Q with shape (n_items, 64); received {q.shape}.")
    if any(len(x) != n_items for x in (movie_ids, titles, genres, positive_counts)):
        raise ValueError("The ALS factors and catalogue arrays do not have matching lengths.")
    if not np.isfinite(q).all():
        raise ValueError("The ALS item-factor matrix contains NaN or infinite values.")

    norms = np.linalg.norm(q, axis=1)
    has_factor = norms > 1e-8
    gram = q.astype(np.float64).T @ q.astype(np.float64)

    return {
        "Q": q,
        "movie_ids": movie_ids,
        "titles": titles,
        "genres": genres,
        "positive_counts": positive_counts,
        "has_factor": has_factor,
        "gram": gram,
    }


def movie_label(item_idx: int, model: dict[str, Any]) -> str:
    title = str(model["titles"][int(item_idx)])
    genres = str(model["genres"][int(item_idx)]).strip()
    return f"{title}  ·  {genres or 'Genres unavailable'}"


def popularity_recommendations(
    model: dict[str, Any],
    exclude: list[int] | None = None,
    k: int = TOP_K,
) -> list[int]:
    counts = model["positive_counts"]
    exclude_set = {int(i) for i in (exclude or [])}
    order = np.argsort(-counts, kind="stable")
    return [int(i) for i in order if int(i) not in exclude_set][:k]


def fold_in_user(
    liked_item_indices: list[int],
    model: dict[str, Any],
    alpha: float = ALPHA,
    regularization: float = REGULARIZATION,
) -> np.ndarray | None:
    """Solve the ALS user-factor update for a new profile of positively liked items.

    The item factors Q are learned by the train-only ALS fit. For each selected
    movie, the implicit preference is p_ui=1 and the extra confidence is alpha.
    This performs a user-factor fold-in; it does not retrain the item factors.
    """
    unique_likes = sorted({int(i) for i in liked_item_indices})
    usable = [i for i in unique_likes if bool(model["has_factor"][i])]
    if not usable:
        return None

    q = model["Q"].astype(np.float64)
    liked_factors = q[np.asarray(usable, dtype=np.int64)]
    identity = np.eye(q.shape[1], dtype=np.float64)

    # This matches the user half-step used by the project's ALS training cell:
    # A = Q'Q + alpha * Q_liked'Q_liked + lambda*I
    # b = (alpha+1) * sum(Q_liked)
    a = model["gram"] + alpha * (liked_factors.T @ liked_factors) + regularization * identity
    b = (alpha + 1.0) * liked_factors.sum(axis=0)
    return np.linalg.solve(a, b)


def als_recommendations(
    liked_item_indices: list[int],
    model: dict[str, Any],
    k: int = TOP_K,
) -> tuple[list[int], str]:
    user_factor = fold_in_user(liked_item_indices, model)
    if user_factor is None:
        return popularity_recommendations(model, exclude=liked_item_indices, k=k), "popularity fallback"

    scores = model["Q"].astype(np.float64) @ user_factor
    eligible = model["has_factor"].copy()
    if liked_item_indices:
        eligible[np.asarray(liked_item_indices, dtype=np.int64)] = False
    scores[~eligible] = -np.inf

    order = np.argsort(-scores, kind="stable")
    results = [int(i) for i in order if np.isfinite(scores[i])][:k]

    # Defensive fallback in case fewer than k factor-bearing candidates remain.
    if len(results) < k:
        already = set(results) | {int(i) for i in liked_item_indices}
        for item_idx in popularity_recommendations(model, exclude=list(already), k=k):
            results.append(item_idx)
            already.add(item_idx)
            if len(results) >= k:
                break
        return results, "ALS with popularity fill"

    return results, "ALS user-factor fold-in"


try:
    model = load_bundle(str(BUNDLE_PATH))
except (FileNotFoundError, ValueError, KeyError) as exc:
    st.title("🎬 CineMatch")
    st.error("The ALS deployment bundle is missing or invalid.")
    st.markdown(
        "Run the export cell described in `EXPORT_CELL.md`, then place the resulting "
        "`als_v1_bundle.npz` file in `models/als_v1_bundle.npz`."
    )
    st.code(str(exc))
    st.stop()

st.markdown(
    """
    <style>
      .hero { padding: 1.35rem 1.5rem; border-radius: 18px;
              background: linear-gradient(115deg, #111827, #273449);
              color: #f9fafb; margin-bottom: 1.2rem; }
      .hero h1 { margin: 0 0 .35rem 0; color: #ffffff; }
      .hero p { margin: 0; color: #d1d5db; }
      .small-note { color: #6b7280; font-size: .88rem; }
    </style>
    <div class="hero">
      <h1>🎬 CineMatch</h1>
      <p>Movie recommendations powered by an ALS collaborative-filtering model.</p>
    </div>
    """,
    unsafe_allow_html=True,
)

st.caption(
    "V1 · ALS-only scoring · MovieLens 32M user-level 10% research sample · "
    "17,241-movie catalogue"
)

if "liked_items" not in st.session_state:
    st.session_state.liked_items = []
if "recommendations" not in st.session_state:
    st.session_state.recommendations = None
if "recommendation_method" not in st.session_state:
    st.session_state.recommendation_method = ""

with st.sidebar:
    st.subheader("Build your taste profile")
    st.write("Search for movies you genuinely liked. We use those choices to estimate a user factor from the trained ALS item factors.")

    search_query = st.text_input(
        "Search movie titles",
        placeholder="e.g. The Matrix, Toy Story, Inception",
    ).strip()

    matching_items: list[int] = []
    if search_query:
        q_lower = search_query.casefold()
        matching_items = [
            i for i, title in enumerate(model["titles"])
            if q_lower in str(title).casefold()
        ]
        matching_items.sort(
            key=lambda i: (-int(model["positive_counts"][i]), str(model["titles"][i]).casefold())
        )
        matching_items = matching_items[:40]

    selected_item: int | None = None
    if matching_items:
        selected_item = st.selectbox(
            "Matching movies",
            options=matching_items,
            format_func=lambda i: movie_label(int(i), model),
            key="movie_search_selection",
        )
        if st.button("＋ Add movie", use_container_width=True):
            if int(selected_item) not in st.session_state.liked_items:
                st.session_state.liked_items.append(int(selected_item))
                st.session_state.recommendations = None
                st.rerun()
            else:
                st.info("That movie is already in your profile.")
    elif search_query:
        st.caption("No titles matched that search. Try a shorter title or alternate spelling.")
    else:
        st.caption("Type a title to find movies to add.")

    st.divider()
    st.markdown(f"**Movies in your profile: {len(st.session_state.liked_items)}**")
    if st.session_state.liked_items:
        for idx in st.session_state.liked_items:
            st.caption(f"♥ {model['titles'][idx]}")
        left, right = st.columns(2)
        with left:
            if st.button("Undo last", use_container_width=True):
                st.session_state.liked_items.pop()
                st.session_state.recommendations = None
                st.rerun()
        with right:
            if st.button("Clear all", use_container_width=True):
                st.session_state.liked_items = []
                st.session_state.recommendations = None
                st.rerun()

    recommend_clicked = st.button(
        "Get recommendations",
        type="primary",
        use_container_width=True,
    )
    if recommend_clicked:
        picks, method = als_recommendations(st.session_state.liked_items, model)
        st.session_state.recommendations = picks
        st.session_state.recommendation_method = method

if st.session_state.recommendations is None:
    results = popularity_recommendations(model, k=TOP_K)
    method = "popular in the training sample"
    st.info("Start with popular picks, or add a few movies you like in the sidebar for personalized ALS recommendations.")
else:
    results = st.session_state.recommendations
    method = st.session_state.recommendation_method

st.subheader("Recommended for you" if st.session_state.liked_items else "Popular movie picks")
st.caption(f"Ranking method: {method}. Genres are displayed as catalogue information; they are not model features in this ALS-only version.")

if not results:
    st.warning("No recommendations were available for this profile.")
else:
    cols = st.columns(2, gap="large")
    for rank, item_idx in enumerate(results, start=1):
        col = cols[(rank - 1) % 2]
        title = str(model["titles"][item_idx])
        genre_text = str(model["genres"][item_idx]).strip()
        count = int(model["positive_counts"][item_idx])
        with col:
            st.markdown(f"**{rank}. {title}**")
            st.caption(genre_text or "Genres unavailable")
            st.caption(f"Positive training interactions: {count:,}")
            st.divider()

st.markdown(
    '<p class="small-note">Your selections are kept in this app session and are not saved to a user account. '
    'This demo uses a train-only ALS item-factor bundle; no model training happens while the app is running.</p>',
    unsafe_allow_html=True,
)
