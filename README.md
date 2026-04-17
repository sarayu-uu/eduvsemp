# Education-Employment Mismatch in India

## Problem Statement
Education attainment in India is rising faster than employment opportunities.  
This project quantifies the mismatch trend, identifies regional hotspots, and models structural drivers (skills, informality, sector composition) using public datasets.

## Data Sources
Raw files are stored in `data/raw/` and include:
- AISHE enrollment data
- PLFS CPERV1 person-level data (`cperv1.csv`)
- Unemployment and economic indicators
- Job-market postings data
- State/UT mapping file

## Project Structure
- `cleaning/cleaning.py`: data cleaning, harmonization, merged dataset creation
- `eda/eda.py`: feature engineering, charts, model training, metric exports
- `app.py`: Streamlit dashboard (interactive analysis)
- `data/eda/model_metrics.txt`: state-level model metrics
- `data/eda/model_metrics_district.txt`: district-level model metrics

## How to Run
```bash
pip install -r requirements.txt
python cleaning/cleaning.py
python eda/eda.py
streamlit run app.py
```

## Dashboard Sections (Current)
- Home
- Time Trend
- Gap Trend (State)
- Model Insights
- Drivers (Feature Importance)

Notes:
- Sidebar uses direct option selection (`radio`), not dropdown.
- Home page KPI auto-displays the best district model R2 from the latest metrics file.
- Model Insights auto-selects the best-performing model for prediction/error plots.

## Modeling Summary
District model setup:
- Target: `gap_ratio = education_index / employment_rate`
- Features: `skill_rate`, `informal_rate`, `sector_share_*`
- Models: Linear Regression, Random Forest, Gradient Boosting
- Split: 80/20 train-test (`random_state=42`)
- Metrics: R2, RMSE

## Latest District Results
From `data/eda/model_metrics_district.txt`:
- LR: R2 = **0.7770**, RMSE = 1.8491
- RF: R2 = 0.7639, RMSE = 1.9026
- GBR: R2 = 0.7620, RMSE = 1.9103

Current best model by R2: **Linear Regression (0.7770)**.

Baseline comparison (mean predictor RMSE on same split):
- Baseline RMSE = 3.9162
- All trained models improve RMSE by ~51-53%.

## Important Caveats
- National `education_index` and `employment_index` are min-max normalized; with very few years, trend-gap values can look extreme.
- Feature importance reflects predictive association, not causal effect.
