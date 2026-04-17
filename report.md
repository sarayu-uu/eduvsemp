# Education-Employment Mismatch in India - Project Report

## Tech Stack
- Python
- pandas, NumPy
- scikit-learn
- Plotly, Matplotlib, Seaborn
- Streamlit

## Problem Statement
Education levels are rising faster than employment opportunities in India, creating a mismatch.  
This project quantifies that mismatch over time, identifies high-mismatch regions, and explains structural drivers using interpretable ML.

## Data Sources
Raw datasets in `data/raw/`:
- AISHE enrollment and university files
- PLFS CPERV1 person-level data (`cperv1.csv`)
- Unemployment indicators
- Economic indicators
- Job market postings
- State/UT mapping file

## Data Cleaning and Integration
Cleaning pipeline: `cleaning/cleaning.py`
- Standardized state names and year formats
- Applied missing-value policy (column/row drop thresholds + median/mode imputation)
- Built employment, skill, informal, and sector-share features from CPERV1
- Aggregated and merged education, employment, economic, and job-demand inputs

Main merged outputs:
- `data/cleaned/final_merged.csv` (state-year merged base)
- `data/cleaned/cperv1_features_by_state.csv`
- `data/cleaned/cperv1_features_by_district.csv`

## Feature Engineering
Core engineered signals:
- `education_index`: min-max normalized education metric
- `employment_index`: min-max normalized employment metric
- `gap = education_index - employment_index`
- `gap_ratio = education_index / employment_rate` (district/state mismatch signal)
- district features: `skill_rate`, `informal_rate`, and `sector_share_*`

## Modeling
District and state regression workflows in `eda/eda.py`:
- Linear Regression
- Random Forest Regressor
- Gradient Boosting Regressor

District model setup:
- Target (`y`): `gap_ratio`
- Features (`X`): `skill_rate`, `informal_rate`, `sector_share_*`
- Train/test split: 80/20 (`random_state=42`)
- Metrics: R2 and RMSE

## Latest Model Metrics
From `data/eda/model_metrics_district.txt`:
- LR: R2 = 0.7770, RMSE = 1.8491
- RF: R2 = 0.7639, RMSE = 1.9026
- GBR: R2 = 0.7620, RMSE = 1.9103

Current best district model by R2: **Linear Regression (0.7770)**.

From `data/eda/model_metrics.txt` (state-level residualized targets):
- Model A (`gap_resid`): best R2 = 0.2016 (RF)
- Model B (`gap_ratio_resid`): best R2 = 0.2185 (RF)

## Streamlit App (Current)
Main app: `app.py`

Active sections:
- Home
- Time Trend
- Model Insights
- Gap Trend (State)
- Drivers (Feature Importance)

Recent UI and logic updates:
- Sidebar changed from dropdown to radio option selection.
- Home page redesigned into a visual overview (hero, KPI cards, trend snapshot, hotspot map).
- Home model KPI now shows **best district R2** parsed from metrics file (not fixed RF).
- Socio-economic segment section removed from active UI.
- Model Insights now auto-selects the **best-performing model** for:
  - Actual vs Predicted scatter
  - Residual distribution
  - Top error districts
- Feature-importance chart remains Random Forest-based (explicitly labeled).

## Interpretation Summary
- District-level modeling is strong relative to baseline and captures substantial mismatch variance.
- Highest-signal explanatory factors are skill coverage, informality, and sector composition.
- State-level residualized models show weaker signal than district models, consistent with higher aggregation and lower variance.

## Notes and Limitations
- National index-based gap is min-max normalized; with very few years, values can appear extreme.
- Model performance depends on quality and coverage of CPERV1-derived employment and sector features.
- Feature importance indicates association strength for prediction, not causal effect.
