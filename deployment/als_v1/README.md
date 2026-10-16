# ALS Movie Recommender — V1 deployment starter

This starter packages a minimal Streamlit demo around the project's **train-only implicit-feedback ALS item factors**. The app contains no training code and does not use TMDB, LightFM, or content-based scores.

## What the app does

- Searches the movie catalogue and lets a visitor select movies they liked.
- Estimates a new user factor with the same ALS user-update equation, holding the trained item factors fixed (fold-in).
- Ranks eligible items by the dot product between that user factor and the trained ALS item factors.
- Uses positive-interaction popularity as a fallback when no selected movie has a usable ALS factor.
- Excludes the visitor's selected movies from the recommendation list.

Movie genres are shown as descriptive catalogue information only; the model does not use them for scoring.

## Files needed

```text
app.py
requirements.txt
models/
  als_v1_bundle.npz
```

The notebook export cell that creates `models/als_v1_bundle.npz` is in `EXPORT_CELL.md`. Run it in the executed `07_hybrid_recommender` notebook **after the train-only ALS fit**. It writes one compact runtime asset and does not export evaluation CSVs/JSON.

## Test locally

From the project root, after exporting the bundle:

```bash
python -m pip install -r requirements.txt
streamlit run app.py
```

Open the local URL printed by Streamlit. Try a few liked movie titles, then click **Get recommendations**. With no selected movies, the app displays the positive-interaction popularity fallback.

## Deploy on Streamlit Community Cloud

1. Place `app.py`, `requirements.txt`, and `models/als_v1_bundle.npz` in the GitHub repository root (or keep the same relative structure and choose the correct entrypoint).
2. Commit the app code and the **runtime bundle only**; keep the source MovieLens files, TMDB API cache, sparse feature matrices, notebook-only reports, and validation/test data out of the deployment commit.
3. Go to https://share.streamlit.io/, connect GitHub, select the repository and branch, choose `app.py` as the entrypoint, and deploy.
4. Use Python 3.11 or 3.12 for the first deployment. The app only needs Streamlit and NumPy at runtime.

The bundle is a required application asset, unlike evaluation CSVs. If you decide not to version model binaries in Git, put the bundle in a suitable model-artifact store and update `BUNDLE_PATH` loading accordingly.

## Important scope / limitations

- The underlying research model uses the project `sample_10pct` subset of MovieLens 32M: 20,093 users and 17,241 catalogue items, not a model retrained on all MovieLens users.
- New profiles are folded into the *fixed* item-factor space. The app does not retain user selections in a database or retrain the model.
- This is an initial portfolio demo, not a production account system. It has no authentication, database, event logging, or persistent preference storage.
- Keep the test set sealed. Do not run test-set comparisons just to prepare the application.
