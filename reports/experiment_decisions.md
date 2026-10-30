# Movie Recommendation System — Experiment Decisions

This file records why the main notebook sequence was reduced to five notebooks and where the other experiments went. The archived notebooks are retained in `archive/original_notebooks/`; they have not been deleted.

## Main research path

| New notebook | Source | Role |
|---|---|---|
| `01_eda.ipynb` | Old 01 | Understand the ratings, catalogue, sparsity, time range and metadata. |
| `02_split_and_baselines.ipynb` | Old 02 | Build the filtered dataset, chronological splits and ranking baselines. |
| `03_collaborative_filtering.ipynb` | Old 03 + 04 | Compare explicit MF with implicit-feedback ALS. |
| `04_content_based.ipynb` | Old 05 | Build genre/decade features and select the frozen content-v1 representation. |
| `05_final_hybrid_evaluation.ipynb` | Old 10 | Compare ALS and LightFM, select the rank-gated warm-tail reranker on validation, and report the final test evaluation. |

The new sequence keeps the experiments that explain the final system. It is not intended to imply that every explored method was successful.

## What was simplified

### 01 — EDA

The notebook is already short. Its opening text and explanations were tightened, while rating distribution, user/movie activity, sparsity, date coverage and metadata checks were retained.

### 02 — Data preparation and baselines

The train/validation/test creation and its temporal and user-item overlap checks remain intact. The notebook now imports `to_sets` and `evaluate_topk` from the project's existing `scripts/recutils.py` instead of defining a second copy of those helpers. Its small global-ranking helper remains local because it is specific to these simple baselines.

### 03 — Collaborative filtering

The old explicit MF and implicit ALS notebooks are combined because they use the same split, item/user indexing, ranking helpers and test comparison.

- Explicit MF remains as a benchmark. It predicts numerical ratings with user/item factors and biases.
- Implicit ALS remains the selected collaborative-filtering model. The validation-selected setup uses positive interactions (`rating >= 4`), 64 factors, `alpha=5`, `lambda=80`, and 10 iterations.
- The two update equations remain separate: explicit-rating ALS and implicit-feedback ALS optimize different objectives.
- Repeated setup and item-count calculations were consolidated. The two implicit-ALS sweep stages now run through one configuration loop.
- The final train + validation refit does not calculate a validation RMSE, because validation rows are part of that refit.

The historical test comparison in this notebook uses all positive test ratings. It is a different evaluation window from the later hybrid experiment, which restricts targets to dates on or after 1 January 2016 UTC.

### 04 — Content-based recommendation

The original content model and its feature-ablation pipeline are presented as one implementation. Feature construction, user profiles, scoring and paired user-level evaluation are shared across the ablation configurations.

- Retains genre and release-decade features, rarity slices, duplicate-feature diagnostics, the ten-configuration ablation, and paired confidence intervals.
- Keeps the selected content-v1 feature matrix export at `data/interim/hybrid_features/content_v1.npz`, which the final hybrid notebook loads.
- Removes a second, redundant construction of the original IDF-weighted matrix that was used only to compare against the configurable feature builder.
- Removes an unused report wrapper.
- The selected content-v1 configuration uses equal genre and decade weights with IDF disabled. Its paired interval against the original IDF-weighted configuration crosses zero, so this is a validation selection, not a statistically established improvement.
- The appended TMDB experiment was removed from this notebook and its findings are recorded below.

### 05 — Final hybrid evaluation

The main hybrid notebook retains the comparison needed to explain the selected approach:

- ALS-only reference.
- LightFM with identity features.
- LightFM with identity plus content-v1 metadata.
- Full-catalogue evaluation, paired user-level confidence intervals and the 98% overall-NDCG retention guardrail.
- Selective warm-tail reranking using the Content-v1 candidate rank.
- Refit on train + validation, followed by the frozen test evaluation.

The TMDB LightFM arm was removed from the main execution path because it did not change the model decision. Repeated imports and the second calculation of pre-test popularity scores were removed. The repeated introspection, split-reconstruction and debug cells were removed. The separate reserved-slot implementation was not merged because it is a different method with a different failure mode; its conclusion is recorded below.

The final test cell still contains a substantial evaluation routine. It was kept because it performs the actual frozen test comparison, applies the same seen-item exclusion and tie-breaking policy, and calculates paired uncertainty. Shrinking that cell by deleting intermediate steps would make the final result harder to audit.

## Archived experiments and findings

### Old 06 — Content enrichment

**Disposition:** archive the implementation; keep the findings here.

- **Title TF-IDF:** content-v1 outperformed the title-enhanced variants in the recorded validation experiment.
- **External MovieLens tags:** no tested tag configuration beat the frozen content-v1 control.
- **Wikidata:** the 2.5% metadata-share candidate had the best observed point estimates, but the paired intervals crossed zero. NDCG@10 changed from 0.004986 to 0.006524 (difference +0.001538; 95% CI [-0.000193, 0.003640]). Recall@10 changed from 0.009608 to 0.012036 (difference +0.002428; 95% CI [-0.000352, 0.005779]). This is inconclusive, not a confirmed improvement.

The full notebook also contains external-data feasibility checks, API queries, feature construction, weight sweeps and repeated evaluation plumbing. The complete history remains in the archive.

### Old 07 — Rank-percentile hybrid

**Disposition:** archive the implementation; do not carry its scorer into the main notebook.

Rank-percentile blending did not improve on ALS in the recorded full-catalogue comparison. The recorded overall View A NDCG@10 values were approximately 0.081462 for ALS, 0.027435 for the ALS/content-v1 blend and 0.030514 for the ALS/TMDB blend. The blending experiment therefore did not provide a suitable replacement for ALS.

### Old 08 — TMDB enrichment

**Disposition:** archive the development notebook.

It documents identifier matching, API pilots, retries, request recovery, cache checks and several iterations of the metadata download process. This is a data-acquisition worklog, not a model-comparison notebook. The final TMDB feature artifacts and the archived code are retained. The main hybrid notebook no longer trains a TMDB LightFM arm.

### Old 09 — Reserved-slot hybrid

**Disposition:** archive as a separate negative experiment; do not merge it into the selected rank-gated method.

This method reserved one or more final recommendation positions for a content-selected tail movie. It improved tail exposure and some tail metrics, but every candidate failed the 98% overall-retention guardrail. For the best reserved-slot control, `ALS_tail_r1`, overall NDCG@10 changed from 0.081462 to 0.078520; the paired difference was -0.002942 (95% CI [-0.003479, -0.002435]). Increasing the number of reserved slots caused larger overall losses.

This is a different approach from the selected rank-gated replacement method in the final notebook.

### Old 11 — Anonymous-user methods

**Disposition:** archive the implementation; keep the negative finding.

The notebook compares leakage-safe fold-in methods for query users, including ALS fold-in, LightFM centroid/ridge approaches and content-to-ALS factor imputation. Across the five-fold development comparison, none produced warm-tail top-10 discovery. The content-to-factor mapper improved some top-500/top-1,000 target-rank statistics but did not improve top-10 recommendations. Tail-only AUC did not establish an advantage over original ALS; duplicate metadata rows also produced large score-tie groups.

These experiments are useful research history, but they did not justify another inference method in the final system.

## Final interpretation

The final research choice is **ALS with a selective LightFM Content-v1 warm-tail reranker**, not a standalone LightFM model and not a general ALS/content score blend.

On the final post-2016 test window, the 500-rank gate produced:

| Metric | ALS | 500-rank gate |
|---|---:|---:|
| Overall NDCG@10 | 0.076251 | 0.076092 |
| Overall Recall@10 | 0.097556 | 0.097151 |
| Warm-tail NDCG@10 | 0.000000 | 0.001685 |
| Warm-tail Recall@10 | 0.000000 | 0.004891 |

The test paired interval for overall NDCG change crossed zero (mean change -0.000159; 95% CI [-0.000399, 0.000071]). The warm-tail recall gain was positive (mean +0.004891; 95% CI [+0.001858, +0.008727]). This is a selective trade-off: warm-tail retrieval improves while overall ranking changes only slightly in this test sample.

The deployed Streamlit application remains ALS-only. The 500-rank gate is a research result and has not been integrated into the live app.
