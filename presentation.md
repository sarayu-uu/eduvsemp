# Education-Employment Mismatch in India
Quantifying trends, hotspots, and structural drivers

---

## Slide 1 - Problem Statement
India's education attainment is rising faster than employment opportunities.  
The mismatch is not uniform across regions.

Goal:
- quantify mismatch over time
- locate where it is worst (state/district)
- identify key drivers for policy targeting

**Presenter notes:**
- This is a diagnostic + explanation system, not a forecasting product.

---

## Slide 2 - What This Project Does
- Integrates education, employment, economic, and job-market datasets
- Builds comparable mismatch indicators (`gap`, `gap_ratio`)
- Maps hotspot regions
- Trains interpretable ML models to explain mismatch differences

**Presenter notes:**
- Emphasize "single view from scattered public data."

---

## Slide 3 - Data Sources
- AISHE (education enrollment)
- PLFS CPERV1 (employment, skills, informality, sector structure)
- Economic indicators (poverty, per-capita, youth unemployment)
- Job-market postings (demand proxy)

**Presenter notes:**
- Mention all sources are standardized and merged before modeling.

---

## Slide 4 - Pipeline Overview
1. Clean and standardize states/years
2. Build merged state-year base table
3. Engineer CPERV1 state and district features
4. Compute mismatch indicators
5. Train and evaluate regression models
6. Serve insights in Streamlit dashboard

**Presenter notes:**
- Code references: `cleaning/cleaning.py`, `eda/eda.py`, `app.py`.

---

## Slide 5 - Core Feature Engineering
- `education_index`: min-max normalized education metric
- `employment_index`: min-max normalized employment metric
- `gap = education_index - employment_index`
- `gap_ratio = education_index / employment_rate`
- district predictors: `skill_rate`, `informal_rate`, `sector_share_*`

**Presenter notes:**
- `gap_ratio` is the district model target.

---

## Slide 6 - Dashboard (Current)
Active sections:
- Home
- Time Trend
- Gap Trend (State)
- Model Insights
- Drivers (Feature Importance)

Recent updates:
- sidebar switched to option list (`radio`)
- redesigned visual Home page
- Home KPI now auto-shows best district R2
- Model Insights auto-selects current best model

---

## Slide 7 - National Trend Snapshot
- Education and employment are index-normalized for comparability
- Gap chart tracks divergence over time
- Current data has limited years, so normalized values can appear extreme

**Presenter notes:**
- With 2 years, min-max normalization can produce `-1` to `+1` gap jumps.

---

## Slide 8 - Regional Hotspot Logic
- State map uses centroid points
- Color encodes mismatch intensity (`gap_ratio`)
- Helps quickly identify high-mismatch regions for targeting

**Presenter notes:**
- Red means higher mismatch; green means better alignment.

---

## Slide 9 - Modeling Setup (District)
Models trained:
- Linear Regression
- Random Forest Regressor
- Gradient Boosting Regressor

Target:
- `y = gap_ratio`

Features:
- `X = [skill_rate, informal_rate, sector_share_*]`

Split and metrics:
- 80/20 train-test split (`random_state=42`)
- R2 and RMSE

---

## Slide 10 - Latest District Results
From `data/eda/model_metrics_district.txt`:
- LR: **R2 = 0.7770**, RMSE = 1.8491
- RF: R2 = 0.7639, RMSE = 1.9026
- GBR: R2 = 0.7620, RMSE = 1.9103

Best model by R2:
- **Linear Regression (0.7770)**

---

## Slide 11 - Baseline Comparison
Baseline model:
- predict mean of training target for all test points

RMSE:
- Baseline: 3.9162
- LR: 1.8491
- RF: 1.9026
- GBR: 1.9103

Interpretation:
- all trained models are ~51-53% better than baseline RMSE

---

## Slide 12 - Model Quality Read
Train vs test behavior:
- LR generalizes best (small train-test gap)
- RF/GBR show higher train scores and lower test gains (more overfit)

Meaning:
- current feature set captures strong signal
- simpler model (LR) is most stable on this split

---

## Slide 13 - Practical Value
- Creates a reproducible mismatch diagnostic from public data
- Surfaces where intervention is needed most
- Identifies likely structural drivers (skills, informality, sector composition)
- Supports evidence-based skilling and employment policy design

---

## Slide 14 - Limitations and Next Steps
Limitations:
- national index trend has few years
- district names unavailable in CPERV1 feature file (codes shown)
- some economic/job proxies are coarse

Next steps:
- add richer time coverage
- add district-name mapping
- add state-specific socio-economic covariates
- test cross-validation and regularized models
