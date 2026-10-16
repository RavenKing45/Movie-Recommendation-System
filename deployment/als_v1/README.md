# CineMatch+ UI Update v3

This update makes the recommendation gallery less dense and gives the taste-profile panel more width.

## Changes
- Three movie cards per row instead of four.
- Wider left taste-profile column.
- Larger movie titles, genres, ranks, interaction counts, and card spacing.
- Keeps the black/white/red theme and the existing ALS recommendation logic.

## Apply
Replace `deployment/als_v1/app.py` with this folder's `app.py`. The existing model bundle remains compatible. No notebook export or model retraining is required for these layout changes.
