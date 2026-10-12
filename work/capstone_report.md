# Capstone Report — Refresh / Content Opportunity Scoring

- **Author:** Ameya Chopade
- **Lane:** Refresh / Content Opportunity Scoring
- **Repo:** [https://github.com/ameyachopade/1](https://github.com/ameyachopade/1)
- **Deployed Paper:** [https://ameyachopade.github.io/1/](https://ameyachopade.github.io/1/)
- **Date:** October 2026

---

## 0. Abstract

This research addresses the problem of identifying and prioritizing decaying organic search content across enterprise domains before traffic collapse occurs. Using a multi-client dataset of 30,000 anonymized web pages across 32 clients, we framed organic decay detection as a risk scoring and ranking task evaluated at Precision@K. We engineered a transparent heuristic baseline combining historical search visibility, content staleness, and ranking vulnerability, and compared it against learned models (Logistic Regression, Decision Tree, and Random Forest) using a strict Grouped Client-Holdout split. Standardized Logistic Regression achieved a Precision@20 of 80.0% and Precision@50 of 80.0% on completely unseen clients—a substantial +28.9 percentage point lift over the test set base rate of 51.1% and beating the rule baseline's 50.0% Precision@20. Error analysis reveals that linear feature weighting avoids the client-specific overfitting seen in tree ensembles, though newly indexed pages with sparse initial clicks remain a key false-positive challenge. The resulting action engine outputs ranked content refresh recommendations with human-interpretable reason codes to direct editorial bandwidth toward the highest-leverage decay opportunities.

---

## 1. Problem Framing

Content decay—the gradual loss of organic search impressions and clicks over time—is a primary driver of lost traffic on enterprise websites. Editorial and SEO teams face an operational bottleneck: while thousands of pages may be months old, only a fraction require immediate editorial intervention. Reviewing every page manually is infeasible, while blind scheduled refreshes waste valuable editorial resources on stable or non-strategic content. 

This work provides a decision-support system that ranks candidate pages by predicted organic decline risk and attaches transparent reason codes (`stale_striking_low_ctr`, `stale_striking_distance`, `stale_low_ctr`, etc.) and suggested action labels (`review_snippet_and_ctr`, `content_refresh`, `monitor`). 
- **Unit of Analysis:** A single URL / content item over a trailing 90-day historical window.
- **Output:** A calibrated decay probability score, an integer rank, a primary reason code, and an action label.
- **Action Taken:** Editorial teams prioritize snippet/title optimization or comprehensive factual refreshes for the top $K$ pages each week.
- **Cost of a Wrong Call:** False positives waste 2–5 hours of editorial time per page; false negatives allow high-volume revenue pages to slip out of striking distance into organic obscurity where recovery requires months of link acquisition.

---

## 2. Data Safety

We used the FlyRank starter dataset containing 30,000 content items across 32 pseudonymized client organizations. 
- **Deliberately Excluded Features (Leakage Prevention):**
  - All forward-looking 30-day window comparison columns (`impressions_last_30d`, `clicks_last_30d`, `sessions_last_30d`, `impressions_prev_30d`, `clicks_prev_30d`, `sessions_prev_30d`).
  - Target-derived fields: `trend_direction` and `trend_pct` were excluded from feature matrices as the target label (`is_declining_label = (trend_direction == "down")`) is directly computed from them.
  - Identification keys: `content_id` and `client_id` were reserved exclusively for grouping and splits, never used as predictive features.
- **Data Anonymization:** No client names, domain URLs, raw queries, or personal identifiable information exist in the dataset or any exported artifacts.

---

## 3. Baseline

Before training machine learning algorithms, we constructed a deterministic rule-based baseline in plain words:
> *"Prioritize a page for refresh review if it demonstrates significant historical search demand (`impressions_90d`), has aged past 90 days without an update (`days_since_last_update >= 90`), and occupies a vulnerable ranking position (`avg_position` between 4 and 20, or underperforming CTR relative to position)."*

Mathematically formulated as:
$$\text{Baseline Score} = \ln(1 + \text{impressions\_90d}) \times (0.50 \cdot \mathbb{I}_{\text{stale}} + 0.30 \cdot \mathbb{I}_{\text{striking}} + 0.20 \cdot \mathbb{I}_{\text{low\_ctr}})$$

Evaluated on the held-out test client set (6,163 rows, 7 unseen clients), the baseline achieved:
- **Precision@10:** 50.0%
- **Precision@20:** 50.0%
- **Precision@50:** 42.0%
- **Precision@100:** 46.0%
- **ROC-AUC:** 0.519 | **PR-AUC:** 0.519 (against a test base rate of 51.1%)

---

## 4. Model & Analysis

We evaluated three learned model architectures against the baseline:
1. **Logistic Regression (Standardized):** Linear log-odds model providing calibrated probabilities and interpretable coefficients.
2. **Decision Tree (depth=5, min_samples_leaf=50):** Non-linear threshold model testing whether discrete rule boundaries capture interactions.
3. **Random Forest (100 trees, depth=10, min_samples_leaf=20):** Ensemble classifier testing bagged feature subspace interactions.

**Feature Matrix (27 features):**
- Log-transformed volume: `log_impressions_90d`, `log_clicks_90d`, `log_sessions_90d`.
- Search & Engagement rates: `avg_position`, `ctr`, `engagement_rate`, `scroll_rate`, `ai_traffic_pct`.
- Temporal: `content_age_days`, `days_since_last_update`, `days_with_impressions`.
- Competition & Content: `search_volume`, `competition`, `cpc`, `word_count`, one-hot encoded `competition_level`, `content_type`, and `main_intent`.

---

## 5. Evaluation

### Grouped Client-Holdout Split
To ensure real-world generalization, we implemented `GroupShuffleSplit` on `client_id` (80% train / 20% test). Content from the 7 test clients never appeared during training:
- **Training Set:** 23,837 rows across 25 clients (Base Rate: 55.0%).
- **Held-Out Test Set:** 6,163 rows across 7 clients (Base Rate: 51.1%).

### Performance Comparison Table

| Model / Strategy | Base Rate | P@10 | P@20 | P@50 | P@100 | ROC-AUC | PR-AUC |
|:---|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| **Week 4 Baseline (Rule)** | 51.1% | 50.0% | 50.0% | 42.0% | 46.0% | 0.519 | 0.519 |
| **Logistic Regression** | 51.1% | **70.0%** | **80.0%** | **80.0%** | **75.0%** | **0.628** | **0.624** |
| **Decision Tree (depth=5)** | 51.1% | 70.0% | 65.0% | 62.0% | 61.0% | 0.609 | 0.588 |
| **Random Forest** | 51.1% | 40.0% | 55.0% | 64.0% | 65.0% | 0.625 | 0.616 |

Logistic Regression established clear superiority in early ranking precision, achieving **80.0% Precision@20** and **80.0% Precision@50** on unseen clients (+28.9pp above the base rate).

---

## 6. Interpretation & Error Analysis

### Learned Feature Drivers
Inspecting the standardized Logistic Regression coefficients reveals what the model relies on:
1. `log_impressions_90d` ($+1.347$): High impression volume is the strongest positive driver of decline risk, as high-visibility assets face the greatest competitive pressure and algorithmic exposure.
2. `log_clicks_90d` ($-0.706$): Active click volume is protective; healthy CTR signals continuous relevance.
3. `content_age_days` ($-0.373$): Older pages that survived historical updates exhibit evergreen stability.
4. `avg_position` ($-0.241$): Pages near the top have the furthest to drop.

### Error Analysis
- **False Positives (Predicted Decay, Actually Stable):** Newly indexed pages (age ~90–100d, e.g. `content_7be5f150dc65`) with zero or sparse clicks misled the model into diagnosing low CTR as decay, even though impressions were just beginning to accrue.
- **False Negatives (Predicted Stable, Actually Declined):** Enterprise flagship content with $>500\text{k}$ impressions and solid positions (e.g. `content_5fe46e04994d`) appeared invulnerable to historical metrics, yet experienced macro SERP volatility (e.g. AI Overviews or aggressive competitor ad packs) that static 90-day totals could not foresee.

---

## 7. Recommendation: Editorial Action Playbook

Editorial teams should adopt the following operational workflow:
1. **Tier 1 (Score $\ge 0.75$, Reason: `stale_striking_low_ctr`):** Immediate SERP Snippet Audit (`review_snippet_and_ctr`). Test updated H1 titles and meta descriptions to restore click capture before position rank decays further.
2. **Tier 2 (Score $\ge 0.60$, Reason: `stale_striking_distance`):** Full Content Refresh (`content_refresh`). Expand thin sections, refresh outdated statistics, and add missing semantic entities to defend positions 4–20.
3. **Tier 3 (Score $< 0.50$, Reason: `stable_monitoring`):** Passive Monitoring (`monitor`). Avoid premature edits that induce indexing instability on stable evergreen articles.

---

## 8. Reproducibility

- **Repository:** [https://github.com/ameyachopade/1](https://github.com/ameyachopade/1)
- **Re-run Instructions:**
  ```bash
  git clone https://github.com/ameyachopade/1.git
  cd 1
  pip install -r requirements.txt
  python -c "import pandas, sklearn; print('Ready')"
  ```
- **Notebooks:**
  - Baseline: `work/notebooks/w04_baseline_score.ipynb`
  - Modeling: `work/notebooks/w05_model.ipynb`
  - Capstone: `work/notebooks/capstone.ipynb`
- **Random Seeds:** Fixed at `seed=42` across all splits, models, and evaluations.
- **Receipts:** Committed to `work/outputs/model_results.json` and `work/outputs/baseline_metrics.json`.

---

## 9. Acknowledgments & Data Credit

Built on the **FlyRank ML Internship dataset** ([https://flyrank.ai](https://flyrank.ai)). 
Special thanks to Haris and the FlyRank team for providing real-world search intelligence warehouse access and industry mentoring.
