from __future__ import annotations

from html import escape
from pathlib import Path
from typing import Any

import numpy as np
import streamlit as st
import streamlit.components.v1 as components

APP_DIR = Path(__file__).resolve().parent
BUNDLE_PATH = APP_DIR / "models" / "als_v1_bundle.npz"
ALPHA = 5.0
REGULARIZATION = 80.0
TOP_K = 10

st.set_page_config(
    page_title="CineMatch+ | Discover your next favourite",
    page_icon="🎬",
    layout="wide",
    initial_sidebar_state="collapsed",
)


@st.cache_resource(show_spinner="Loading the CineMatch recommendation engine…")
def load_bundle(path_string: str) -> dict[str, Any]:
    path = Path(path_string)
    if not path.exists():
        raise FileNotFoundError(
            f"ALS bundle not found at {path}. Run the export cell described in "
            "EXPORT_CELL.md and place als_v1_bundle.npz in the models/ folder."
        )

    with np.load(path, allow_pickle=False) as data:
        q = np.asarray(data["Q"], dtype=np.float32)
        movie_ids = np.asarray(data["movie_ids"], dtype=np.int64)
        titles = np.asarray(data["titles"], dtype=str)
        genres = np.asarray(data["genres"], dtype=str)
        positive_counts = np.asarray(data["positive_counts"], dtype=np.int32)
        # Optional while supporting the current prototype bundle. Populate this
        # field by rerunning the updated export cell to enable real TMDB posters.
        poster_urls = (
            np.asarray(data["poster_urls"], dtype=str)
            if "poster_urls" in data.files
            else np.full(len(titles), "", dtype=str)
        )

    n_items = q.shape[0]
    if q.ndim != 2 or q.shape[1] != 64:
        raise ValueError(f"Expected Q with shape (n_items, 64); received {q.shape}.")
    if any(len(x) != n_items for x in (movie_ids, titles, genres, positive_counts, poster_urls)):
        raise ValueError("ALS factors and catalogue arrays do not have matching lengths.")
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
        "poster_urls": poster_urls,
        "has_factor": has_factor,
        "gram": gram,
    }


def movie_label(item_idx: int, model: dict[str, Any]) -> str:
    title = str(model["titles"][int(item_idx)])
    genres = str(model["genres"][int(item_idx)]).strip()
    return f"{title}  ·  {genres or 'Genres unavailable'}"


def popularity_recommendations(
    model: dict[str, Any], exclude: list[int] | None = None, k: int = TOP_K
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
    """Estimate a new user's ALS factor from their positively liked movies."""
    unique_likes = sorted({int(i) for i in liked_item_indices})
    usable = [i for i in unique_likes if bool(model["has_factor"][i])]
    if not usable:
        return None

    q = model["Q"].astype(np.float64)
    liked_factors = q[np.asarray(usable, dtype=np.int64)]
    identity = np.eye(q.shape[1], dtype=np.float64)
    a = model["gram"] + alpha * (liked_factors.T @ liked_factors) + regularization * identity
    b = (alpha + 1.0) * liked_factors.sum(axis=0)
    return np.linalg.solve(a, b)


def als_recommendations(
    liked_item_indices: list[int], model: dict[str, Any], k: int = TOP_K
) -> tuple[list[int], str]:
    user_factor = fold_in_user(liked_item_indices, model)
    if user_factor is None:
        return popularity_recommendations(model, exclude=liked_item_indices, k=k), "popular picks fallback"

    scores = model["Q"].astype(np.float64) @ user_factor
    eligible = model["has_factor"].copy()
    if liked_item_indices:
        eligible[np.asarray(liked_item_indices, dtype=np.int64)] = False
    scores[~eligible] = -np.inf
    order = np.argsort(-scores, kind="stable")
    results = [int(i) for i in order if np.isfinite(scores[i])][:k]

    if len(results) < k:
        already = set(results) | {int(i) for i in liked_item_indices}
        for item_idx in popularity_recommendations(model, exclude=list(already), k=k):
            results.append(item_idx)
            already.add(item_idx)
            if len(results) >= k:
                break
        return results, "ALS recommendations + popular picks"
    return results, "ALS personalised ranking"


def add_to_profile(item_idx: int) -> None:
    item_idx = int(item_idx)
    if item_idx not in st.session_state.liked_items:
        st.session_state.liked_items.append(item_idx)
        st.session_state.recommendations = None
        st.session_state.recommendation_method = ""


# ---------- Cinematic styling: black, white, cinema red and bold sans-serif ----------
st.markdown(
    r"""
    <style>
      @import url('https://fonts.googleapis.com/css2?family=Montserrat:wght@400;500;600;700;800;900&display=swap');

      :root {
        --cm-bg: #0b0b0c;
        --cm-panel: #121214;
        --cm-panel-raised: #19191c;
        --cm-line: #2a2a2e;
        --cm-white: #f7f7f8;
        --cm-muted: #a5a5ad;
        --cm-red: #e51d43;
        --cm-red-hot: #ff3157;
      }
      html, body, [class*="css"], [data-testid="stAppViewContainer"] {
        font-family: 'Montserrat', 'Arial', sans-serif;
      }
      .stApp { background: var(--cm-bg); color: var(--cm-white); }
      /* Streamlit's native top bar was overlapping the custom navigation. */
      [data-testid="stHeader"] { display: none !important; }
      [data-testid="stToolbar"], #MainMenu, footer { visibility: hidden; }
      [data-testid="stDecoration"] { display: none; }
      .block-container { max-width: 1600px; padding-top: 1.45rem; padding-bottom: 3rem; }
      p, label, .stCaption, [data-testid="stMarkdownContainer"] { color: var(--cm-white); }
      [data-testid="stCaptionContainer"] p { color: var(--cm-muted); }
      h1, h2, h3, h4 { font-family: 'Montserrat', 'Arial', sans-serif !important; letter-spacing: -.035em; }
      h1 { font-weight: 900 !important; }
      h2, h3 { font-weight: 800 !important; }
      div[data-testid="stTextInput"] input,
      div[data-testid="stSelectbox"] div[data-baseweb="select"] > div {
        background: #19191c !important; color: #f7f7f8 !important; font-size:.95rem !important;
        border: 1px solid #35353a !important; border-radius: 10px !important; padding-top:.65rem !important; padding-bottom:.65rem !important;
      }
      div[data-testid="stTextInput"] input:focus,
      div[data-testid="stSelectbox"] div[data-baseweb="select"] > div:focus-within {
        border-color: var(--cm-red) !important; box-shadow: 0 0 0 1px var(--cm-red) !important;
      }
      [data-testid="stSelectbox"] svg { fill: #f7f7f8 !important; }
      .stButton button {
        font-family: 'Montserrat', 'Arial', sans-serif !important;
        font-size:.88rem !important; font-weight:800 !important; border-radius:9px !important;
        min-height:2.75rem; padding:.55rem .85rem !important;
        border:1px solid #3a3a40 !important; transition:all .16s ease;
      }
      .stButton button[kind="primary"] {
        background: var(--cm-red) !important; border-color: var(--cm-red) !important;
        color: white !important;
      }
      .stButton button[kind="primary"]:hover {
        background: var(--cm-red-hot) !important; border-color: var(--cm-red-hot) !important;
        box-shadow: 0 6px 22px rgba(229,29,67,.20);
      }
      .stButton button[kind="secondary"] { background: #19191c !important; color: #f7f7f8 !important; }
      .stButton button[kind="secondary"]:hover { border-color: var(--cm-red) !important; color: white !important; }
      hr { border-color: var(--cm-line) !important; }
      [data-testid="stAlert"] { background: #181214; border: 1px solid #58202d; color: #f7f7f8; }

      .cm-nav {
        display:flex; align-items:center; justify-content:space-between; gap:1.2rem;
        min-height:64px; box-sizing:border-box; padding:.55rem 0 1rem;
        border-bottom:1px solid #29292d; margin-bottom:1.35rem;
        position:relative; z-index:5; overflow:visible;
      }
      .cm-nav, .cm-nav * { line-height:1.45 !important; overflow:visible; box-sizing:border-box; }
      .cm-nav .cm-brand-icon { line-height:1 !important; }
      [data-testid="stMarkdownContainer"] .cm-nav { margin-top:.1rem; }
      .cm-brand { display:flex; align-items:center; gap:.55rem; font-size:1.08rem; font-weight:900; letter-spacing:-.035em; color:#fff; white-space:nowrap; flex-shrink:0; }
      .cm-brand-icon { width:29px; height:29px; border:2px solid #f4f4f5; border-radius:50%; display:inline-flex; align-items:center; justify-content:center; font-size:.83rem; }
      .cm-brand-plus { color:var(--cm-red); }
      .cm-nav-links { display:flex; align-items:center; gap:clamp(1rem,2vw,2rem); color:#dddde0; font-size:.76rem; font-weight:800; letter-spacing:.035em; text-transform:uppercase; white-space:nowrap; }
      .cm-status { border:1px solid #35353a; padding:.55rem .78rem; border-radius:5px; font-size:.7rem; font-weight:800; letter-spacing:.045em; text-transform:uppercase; color:#e7e7e9; white-space:nowrap; flex-shrink:0; }
      .cm-status span { color:var(--cm-red); }
      .cm-hero {
        position:relative; overflow:hidden; padding:clamp(1.7rem,4vw,3.2rem); border:1px solid #242428;
        border-radius:4px; background:
          radial-gradient(ellipse at 93% 20%, rgba(229,29,67,.19), transparent 34%),
          linear-gradient(110deg, #151517 0%, #101012 54%, #1a0c11 100%);
        margin-bottom:1rem;
      }
      .cm-hero:after { content:'+'; position:absolute; right:4%; bottom:-5rem; color:rgba(229,29,67,.08); font-weight:900; font-size:24rem; line-height:1; pointer-events:none; }
      .cm-kicker { color:var(--cm-red-hot); font-size:.69rem; font-weight:900; letter-spacing:.16em; text-transform:uppercase; margin-bottom:1rem; position:relative; z-index:1; }
      .cm-hero h1 { max-width:850px; color:#fff; font-size:clamp(2.15rem,5.5vw,4.8rem); line-height:.99; margin:0 0 1.1rem; text-transform:uppercase; position:relative; z-index:1; }
      .cm-hero h1 em { color:var(--cm-red); font-style:normal; }
      .cm-hero p { max-width:740px; color:#c4c4ca; font-size:.91rem; line-height:1.8; margin:0; position:relative; z-index:1; }
      .cm-stat-strip { display:grid; grid-template-columns:repeat(3,minmax(0,1fr)); background:#141416; border:1px solid #29292d; margin:0 0 1.8rem; }
      .cm-stat { padding:1.1rem 1.25rem; border-right:1px solid #29292d; }
      .cm-stat:last-child { border-right:0; }
      .cm-stat strong { display:block; font-size:1.42rem; font-weight:900; color:#fff; letter-spacing:-.04em; }
      .cm-stat strong span { color:var(--cm-red); }
      .cm-stat small { display:block; color:#a5a5ad; margin-top:.3rem; font-size:.65rem; font-weight:700; letter-spacing:.06em; text-transform:uppercase; }
      .cm-section-heading { display:flex; align-items:baseline; justify-content:space-between; gap:1rem; margin:.3rem 0 1rem; }
      .cm-section-heading h2 { margin:0; font-size:1.6rem; text-transform:uppercase; }
      .cm-section-heading h2 span { color:var(--cm-red); }
      .cm-section-heading small { color:#a5a5ad; text-transform:uppercase; letter-spacing:.1em; font-size:.7rem; font-weight:800; }
      .cm-panel { background:#121214; border:1px solid #29292d; border-radius:4px; padding:1.55rem; margin-bottom:1.2rem; }
      .cm-panel-title { font-size:1.12rem; font-weight:900; text-transform:uppercase; margin:0 0 .5rem; }
      .cm-panel-title span { color:var(--cm-red); }
      .cm-panel-copy { font-size:.94rem; line-height:1.85; color:#b4b4bc; margin-bottom:.9rem; }
      .cm-profile-count { display:inline-block; color:#fff; background:#271117; border-left:3px solid var(--cm-red); padding:.65rem .85rem; font-size:.86rem; font-weight:800; letter-spacing:.03em; margin:.3rem 0 .9rem; }
      .cm-chip { display:inline-block; margin:0 .27rem .48rem 0; padding:.5rem .68rem; border:1px solid #3a252b; border-radius:3px; background:#1c1215; color:#f7f7f8; font-size:.86rem; font-weight:700; line-height:1.5; }
      .cm-chip:before { content:'♥ '; color:var(--cm-red); }
      .cm-movie-title { color:#f8f8f9; font-weight:850; font-size:1.12rem; line-height:1.4; margin:.75rem 0 .55rem; min-height:3.1rem; }
      .cm-genre { color:#e1e1e4; background:#252528; border-radius:2px; display:inline-block; font-size:.76rem; line-height:1.5; font-weight:850; padding:.38rem .52rem; margin:0 .25rem .38rem 0; text-transform:uppercase; }
      .cm-meta { color:#b7b7bf; font-size:.82rem; line-height:1.7; }
      .cm-rank { color:var(--cm-red); font-size:.82rem; font-weight:900; letter-spacing:.12em; text-transform:uppercase; margin:.9rem 0 .45rem; }
      .cm-placeholder { aspect-ratio:2/3; width:100%; box-sizing:border-box; display:flex; flex-direction:column; justify-content:space-between; padding:1.1rem; overflow:hidden; border:1px solid #36363b; background:linear-gradient(155deg,#27272b,#141416 55%,#321019); position:relative; }
      .cm-placeholder:after { content:'+'; position:absolute; right:-.25rem; bottom:-2.5rem; color:rgba(229,29,67,.17); font-weight:900; font-size:8rem; line-height:1; }
      .cm-placeholder .cm-ph-top { color:#e4e4e7; font-size:.68rem; font-weight:900; letter-spacing:.13em; text-transform:uppercase; position:relative; z-index:1; }
      .cm-placeholder .cm-ph-title { color:white; font-weight:900; font-size:1.45rem; line-height:1.08; text-transform:uppercase; overflow-wrap:anywhere; position:relative; z-index:1; }
      .cm-placeholder .cm-ph-bottom { color:#f14a67; font-size:.68rem; letter-spacing:.11em; font-weight:900; text-transform:uppercase; position:relative; z-index:1; }
      .cm-footer { border-top:1px solid #29292d; padding-top:1rem; margin-top:2.1rem; color:#87878f; font-size:.63rem; line-height:1.8; }
      .cm-footer strong { color:#e7e7e9; }
      @media (max-width: 900px) {
        .cm-nav { gap:.6rem; }
        .cm-nav-links { display:none; }
        .cm-hero { padding:1.45rem 1.2rem; }
        .cm-hero h1 { font-size:2.2rem; }
        .cm-stat { padding:.8rem .6rem; }
        .cm-stat strong { font-size:1rem; }
        .cm-stat small { font-size:.52rem; }
        .cm-section-heading h2 { font-size:1.2rem; }
        .cm-status { font-size:.57rem; padding:.4rem .5rem; }
      }
    </style>
    """,
    unsafe_allow_html=True,
)

try:
    model = load_bundle(str(BUNDLE_PATH))
except (FileNotFoundError, ValueError, KeyError) as exc:
    st.title("CineMatch+")
    st.error("The ALS deployment bundle is missing or invalid.")
    st.markdown("Run the export cell described in `EXPORT_CELL.md`, then put `als_v1_bundle.npz` in the `models/` folder.")
    st.code(str(exc))
    st.stop()

# ---------- Session state ----------
if "liked_items" not in st.session_state:
    st.session_state.liked_items = []
if "recommendations" not in st.session_state:
    st.session_state.recommendations = None
if "recommendation_method" not in st.session_state:
    st.session_state.recommendation_method = ""

POPULAR_COMPONENT_DIR = APP_DIR / "popular_row"
_popular_row = components.declare_component("cinematch_popular_row", path=str(POPULAR_COMPONENT_DIR))

FEATURED_COUNT = 5        # banner slides, rotated by CSS
POPULAR_TOTAL = 18        # popular titles available in the row
POPULAR_PAGE_SIZE = 6    # posters visible at once
SLIDE_SECONDS = 5

st.markdown(
    """
    <style>
      .cm-banner { position:relative; height:470px; overflow:hidden; border-radius:6px; background:#0b0b0c; border:1px solid #242428; margin-bottom:1.6rem; }
      .cm-slide { position:absolute; inset:0; opacity:0; background-color:#1a0c11; background-size:cover; background-position:center 25%;
                  animation: cm-fade __DURATION__s linear infinite; }
      .cm-slide:before { content:''; position:absolute; inset:0;
                  background: linear-gradient(90deg, rgba(11,11,12,.96) 0%, rgba(11,11,12,.72) 42%, rgba(11,11,12,.15) 100%),
                              linear-gradient(0deg, #0b0b0c 0%, rgba(11,11,12,0) 45%); }
      @keyframes cm-fade { 0% {opacity:0} 4% {opacity:1} 20% {opacity:1} 24% {opacity:0} 100% {opacity:0} }
      .cm-slide-copy { position:absolute; left:clamp(1.2rem,4vw,3.5rem); bottom:clamp(1.5rem,5vw,3.2rem); max-width:640px; z-index:1; }
      .cm-slide-title { color:#fff; font-weight:900; font-size:clamp(2rem,4.6vw,3.9rem); line-height:1.0; letter-spacing:-.03em; text-transform:uppercase; margin:.5rem 0 .8rem; }
      .cm-slide-genres { color:#e1e1e4; font-size:.8rem; font-weight:800; letter-spacing:.12em; text-transform:uppercase; }
      .cm-slide-count { color:#a5a5ad; font-size:.8rem; font-weight:700; margin-top:.4rem; }
      .cm-begin { display:inline-block; margin-top:1.3rem; background:#e51d43; color:#fff !important; padding:.8rem 1.8rem; border-radius:9px;
                  font-weight:800; font-size:.88rem; letter-spacing:.06em; text-transform:uppercase; text-decoration:none; }
      .cm-begin:hover { background:#ff3157; box-shadow:0 6px 22px rgba(229,29,67,.25); }
      .cm-row-title { color:#f8f8f9; font-size:.8rem; font-weight:800; line-height:1.3; margin:.5rem 0 .4rem; height:2.6rem; overflow:hidden; display:-webkit-box; -webkit-line-clamp:2; -webkit-box-orient:vertical; overflow-wrap:anywhere; }
      .cm-poster { width:100%; aspect-ratio:2/3; overflow:hidden; border-radius:4px; background:#19191c; }
      .cm-poster img { width:100%; height:100%; object-fit:cover; display:block; }
      .st-key-popular_prev, .st-key-popular_next { margin-top:8.5rem; }
      .cm-in-profile { color:#f04a66; font-size:.72rem; font-weight:800; letter-spacing:.06em; text-transform:uppercase; margin-top:.3rem; }
    </style>
    """.replace("__DURATION__", str(FEATURED_COUNT * SLIDE_SECONDS)),
    unsafe_allow_html=True,
)

# ---------- Header / nav ----------
st.markdown(
    """
    <div class="cm-nav">
      <div class="cm-brand"><span class="cm-brand-icon">◉</span> CINE<span style="color:#fff">MATCH</span><span class="cm-brand-plus">+</span></div>
      <div class="cm-nav-links"><span>Discover</span><span>Your taste</span><span>How it works</span></div>
      <div class="cm-status"><span>●</span>&nbsp; ALS-powered prototype</div>
    </div>
    """,
    unsafe_allow_html=True,
)

# ---------- Rotating banner (featured = most popular, rotates through top few) ----------
featured = popularity_recommendations(model, k=FEATURED_COUNT)
slides_html = []
for pos, idx in enumerate(featured):
    url = str(model["poster_urls"][idx]).strip()
    bg = f"background-image:url('{url}');" if url.startswith("https://image.tmdb.org/t/p/") else ""
    genres = " · ".join(g.strip() for g in str(model["genres"][idx]).split("|") if g.strip()) or "Genres unavailable"
    count = int(model["positive_counts"][idx])
    slides_html.append(
        f'<div class="cm-slide" style="animation-delay:{pos * SLIDE_SECONDS}s;{bg}">'
        f'<div class="cm-slide-copy">'
        f'<div class="cm-kicker">Most loved in the sample</div>'
        f'<div class="cm-slide-title">{escape(str(model["titles"][idx]))}</div>'
        f'<div class="cm-slide-genres">{escape(genres)}</div>'
        f'<div class="cm-slide-count">♥ {count:,} positive interactions</div>'
        f'<a class="cm-begin" href="#taste-profile">Begin</a>'
        f'</div></div>'
    )
st.markdown('<div class="cm-banner">' + "".join(slides_html) + "</div>", unsafe_allow_html=True)

# ---------- Popular movies row (custom component: 6 at a time, arrows + hover paging) ----------
st.markdown(
    '<div class="cm-section-heading"><h2>Popular <span>movies</span></h2><small>00 / trending</small></div>',
    unsafe_allow_html=True,
)

popular_items = []
for idx in popularity_recommendations(model, k=POPULAR_TOTAL):
    url = str(model["poster_urls"][idx]).strip()
    popular_items.append({
        "idx": int(idx),
        "title": str(model["titles"][idx]),
        "poster": url if url.startswith("https://image.tmdb.org/t/p/") else "",
        "inProfile": int(idx) in st.session_state.liked_items,
    })

popular_event = _popular_row(
    items=popular_items,
    pageSize=POPULAR_PAGE_SIZE,
    key="cinematch_popular_row",
    default=None,
)

if isinstance(popular_event, dict) and popular_event.get("n") != st.session_state.get("popular_last_event"):
    st.session_state.popular_last_event = popular_event.get("n")
    movie_idx = int(popular_event["add"])
    if movie_idx not in st.session_state.liked_items:
        add_to_profile(movie_idx)
        st.rerun()

st.markdown('<div id="taste-profile"></div>', unsafe_allow_html=True)


# ---------- Taste profile panel ----------
st.markdown(
    '<div class="cm-section-heading"><h2>Build your <span>taste profile</span></h2><small>01 / personalise</small></div>',
    unsafe_allow_html=True,
)
profile_col, results_col = st.columns([1.2, 2.0], gap="large")

with profile_col:
    st.markdown(
        '<div class="cm-panel"><div class="cm-panel-title">Start with movies <span>you love</span></div><div class="cm-panel-copy">Search the catalogue and add a few favourites. The more representative your choices, the more useful the recommendations can be.</div></div>',
        unsafe_allow_html=True,
    )
    search_query = st.text_input(
        "Search catalogue",
        placeholder="Try The Matrix, Toy Story…",
        label_visibility="collapsed",
        key="cm_search_query",
    ).strip()

    matching_items: list[int] = []
    if search_query:
        q_lower = search_query.casefold()
        matching_items = [i for i, title in enumerate(model["titles"]) if q_lower in str(title).casefold()]
        matching_items.sort(key=lambda i: (-int(model["positive_counts"][i]), str(model["titles"][i]).casefold()))
        matching_items = matching_items[:40]

    selected_item: int | None = None
    if matching_items:
        selected_item = st.selectbox(
            "Matching titles",
            options=matching_items,
            format_func=lambda i: movie_label(int(i), model),
            label_visibility="collapsed",
            key="cm_movie_search_selection",
        )
        if st.button("＋  Add to my profile", type="primary", use_container_width=True, key="cm_add_search"):
            if int(selected_item) not in st.session_state.liked_items:
                add_to_profile(int(selected_item))
                st.rerun()
            else:
                st.info("That movie is already in your profile.")
    elif search_query:
        st.caption("No titles matched. Try a shorter title or another spelling.")
    else:
        st.caption("Start typing a movie title to find a match.")

    st.markdown(
        f'<div class="cm-profile-count">♥ &nbsp; YOUR PROFILE &nbsp;·&nbsp; {len(st.session_state.liked_items)} MOVIES</div>',
        unsafe_allow_html=True,
    )
    if st.session_state.liked_items:
        for idx in list(st.session_state.liked_items):
            c1, c2 = st.columns([5, 1], gap="small")
            with c1:
                st.markdown(f'<div class="cm-chip">{escape(str(model["titles"][idx]))}</div>', unsafe_allow_html=True)
            with c2:
                if st.button("×", key=f"cm_remove_{idx}", help="Remove this movie", use_container_width=True):
                    st.session_state.liked_items.remove(idx)
                    st.session_state.recommendations = None
                    st.session_state.recommendation_method = ""
                    st.rerun()
        c1, c2 = st.columns(2, gap="small")
        with c1:
            if st.button("Undo last", use_container_width=True, key="cm_undo"):
                st.session_state.liked_items.pop()
                st.session_state.recommendations = None
                st.session_state.recommendation_method = ""
                st.rerun()
        with c2:
            if st.button("Clear all", use_container_width=True, key="cm_clear"):
                st.session_state.liked_items = []
                st.session_state.recommendations = None
                st.session_state.recommendation_method = ""
                st.rerun()
    else:
        st.markdown('<div class="cm-panel-copy">Your favourites will appear here. We never show your selected films as recommendations.</div>', unsafe_allow_html=True)

    st.markdown("<div style='height:.35rem'></div>", unsafe_allow_html=True)
    if st.button("GET MY RECOMMENDATIONS  →", type="primary", use_container_width=True, key="cm_recommend"):
        picks, method = als_recommendations(st.session_state.liked_items, model)
        st.session_state.recommendations = picks
        st.session_state.recommendation_method = method
        st.rerun()

    st.markdown(
        '<div class="cm-panel-copy" style="margin-top:.8rem">No sign-up. No saved account. Your selections stay in this browser session.</div>',
        unsafe_allow_html=True,
    )

# ---------- Recommendation gallery ----------
if st.session_state.recommendations is None:
    results = popularity_recommendations(model, k=TOP_K)
    method = "popular in the training sample"
else:
    results = st.session_state.recommendations
    method = st.session_state.recommendation_method or "ALS personalised ranking"

with results_col:
    st.markdown(
        '<div class="cm-section-heading"><h2>Curated for <span>you</span></h2><small>02 / discover</small></div>',
        unsafe_allow_html=True,
    )
    if st.session_state.recommendations is None:
        st.markdown("<div class='cm-panel-copy'>A first look at popular picks. Add a few favourites to create your personal recommendations.</div>", unsafe_allow_html=True)
    else:
        st.markdown(f"<div class='cm-panel-copy'>Ranked with {escape(method)}. Your profile movies are excluded from these results.</div>", unsafe_allow_html=True)

    if not results:
        st.warning("No recommendations were available for this profile.")
    else:
        # Three larger poster cards per row reduce visual density without changing ranking.
        for row_start in range(0, len(results), 3):
            row_items = results[row_start:row_start + 3]
            cols = st.columns(3, gap="large")
            for offset, (col, item_idx) in enumerate(zip(cols, row_items)):
                title = str(model["titles"][item_idx])
                genre_text = str(model["genres"][item_idx]).strip()
                genres = [g.strip() for g in genre_text.split("|") if g.strip()]
                count = int(model["positive_counts"][item_idx])
                poster_url = str(model["poster_urls"][item_idx]).strip()
                with col:
                    st.markdown(f'<div class="cm-rank">PICK {row_start + offset + 1:02d}</div>', unsafe_allow_html=True)
                    if poster_url.startswith("https://image.tmdb.org/t/p/"):
                        st.image(poster_url, use_container_width=True)
                    else:
                        short_title = escape(title.split("(")[0].strip())
                        st.markdown(
                            f'<div class="cm-placeholder"><div class="cm-ph-top">CINEMATCH + &nbsp; / &nbsp; FILM</div><div class="cm-ph-title">{short_title}</div><div class="cm-ph-bottom">A STORY WAITING TO BE FOUND</div></div>',
                            unsafe_allow_html=True,
                        )
                    st.markdown(f'<div class="cm-movie-title">{escape(title)}</div>', unsafe_allow_html=True)
                    genre_html = "".join(f'<span class="cm-genre">{escape(g)}</span>' for g in genres[:2])
                    if not genre_html:
                        genre_html = '<span class="cm-genre">GENRE N/A</span>'
                    st.markdown(genre_html, unsafe_allow_html=True)
                    st.markdown(f'<div class="cm-meta">♥ &nbsp; {count:,} positive training interactions</div>', unsafe_allow_html=True)
                    if item_idx not in st.session_state.liked_items:
                        if st.button("＋ Like", key=f"cm_like_{item_idx}", use_container_width=True):
                            add_to_profile(item_idx)
                            st.rerun()
                    else:
                        st.markdown('<div class="cm-meta" style="color:#f04a66;font-weight:800;margin-top:.35rem">♥ IN YOUR PROFILE</div>', unsafe_allow_html=True)

st.markdown(
    '<div class="cm-footer"><strong>CINEMATCH<span style="color:#e51d43">+</span></strong> &nbsp; · &nbsp; MOVIE DISCOVERY, REIMAGINED.<br>Prototype V1 · ALS collaborative filtering · MovieLens 32M research sample. The current model was trained on a 10% sample; this is a research prototype, not the final full-data model. Poster images and related metadata are provided via TMDB. This product uses the TMDB API but is not endorsed or certified by TMDB.</div>',
    unsafe_allow_html=True,
)
