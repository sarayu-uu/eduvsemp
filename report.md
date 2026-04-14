# Education?Employment Mismatch in India ? Project Report

## Tech Stack
- **Python**: data cleaning, feature engineering, modeling, and plotting
- **Pandas**: data manipulation and joins
- **Seaborn / Matplotlib**: EDA plots and static charts
- **Plotly**: interactive map and segment bar charts in Streamlit
- **Scikit?learn**: regression models and feature importance
- **Streamlit**: dashboard/UI

## Problem Statement
Education levels in India are rising faster than employment opportunities, creating a widening mismatch. The goal of this project is to quantify that mismatch over time, identify where it is most severe, and explain which socio?economic and job?market factors drive it.

**How the project answers the PS**
- Builds education and employment indexes and tracks the gap over time
- Identifies mismatch hotspots by state and district
- Quantifies how socio?economic segments (poverty/income) relate to mismatch
- Explains drivers using feature importance from machine?learning models

## Data Collection
**Raw datasets (in `data/raw/`)**
- **AISHE**: `aishe_enrollment.csv.csv`, `aishe_university_type.csv.csv`
- **PLFS CPERV1 (person file)**: `cperv1.csv`
- **Unemployment**: `unemployment_india.csv.csv`
- **Economic indicators**: `economic_factors.csv.csv`
- **Job market postings**: `job_market.csv`
- **State code mapping**: `4. Indian_States_and_UTs_Code  Name.xlsx`

These sources supply education, employment, socio?economic indicators, and job?market structure.

## Data Cleaning & Preprocessing
All cleaning logic is in `cleaning/cleaning.py`.

### Standardization
- **State names** are normalized using a mapping in `STATE_MAP`.
- **Year** values are parsed from strings like `2019-20` using `year_from_string()`.

### Missing Value Policy
The function `apply_missing_policy()` is applied to multiple datasets:
- **Drop columns** with > 40% missing values
- **Drop rows** with > 50% missing values
- **Fill remaining missing values**:
  - Numeric columns: **median**
  - Categorical columns: **mode**

> This keeps features stable while minimizing bias from heavy missingness.

### Dataset?Specific Cleaning
- **Unemployment**: monthly data is aggregated to yearly, standardized by state.
- **Economic factors**: columns renamed to consistent names (`per_capita`, `poverty_rate`, etc.).
- **AISHE**: multi?row headers are reconstructed; data is reshaped to long format.
- **Job market**: sector inferred from `tagsAndSkills`; median salary calculated.
- **CPERV1**:
  - `Principal_Status_Code` ? **employed_flag** (11?51), **unemployed_flag** (81?82)
  - **skill_flag** from vocational training & recent training fields
  - **informal_flag** from contract type and social security indicators
  - **industry_section** derived from NIC codes, then converted into sector shares

Cleaned outputs are written to `data/cleaned/`.

## Data Integration (Where and How Datasets Are Combined)
Two main integration points:

### 1) `build_base_and_merge()` ? `cleaning/cleaning.py`
Combines:
- **Unemployment (base table)** by `State`, `Year`
- **Economic indicators** by `Year`
- **AISHE national education totals** by `Year`
- **Job market aggregates** by `Year`

**Output**: `data/cleaned/final_merged.csv`

### 2) `build_state_year_dataset()` ? `eda/eda.py`
Merges:
- `final_merged.csv` (state?year table)
- CPERV1 state features (`cperv1_features_by_state.csv`) by `State`

**Output**: in?memory `state_year` used for modeling, trends, and segmentation

## Feature Engineering
Key engineered features:

- **education_index**: normalized education metric (AISHE or CPERV1)
- **employment_index**: normalized employment metric
- **gap**: `education_index - employment_index`
- **gap_ratio**: `education_index / employment_index`
- **gap_simple**: `education_index / employment_rate` (CPERV1)
- **job_demand_adjusted**: job demand scaled by year trend
- **Residualized features** (`*_resid`): remove time trend using linear regression vs Year

**District?level features** (from CPERV1):
- `skill_rate`, `informal_rate`
- `sector_share_*` for NIC industry sections

## EDA Performed
EDA is implemented in `eda/eda.py`.

### Trend Analysis
- National mismatch trend (`gap_trend.png`)
- Education vs employment trend (`education_vs_employment_trend.png`)

### Segmentation Analysis
- Poverty segment comparison (`gap_by_poverty_segment.png`)
- Income segment comparison (`gap_by_income_segment.png`)

### District / State Insights
- Top/Bottom 10 districts by mismatch
- Top/Bottom 10 states by mismatch

## Modeling & Feature Importance
Models trained in `train_models()`:
- **Linear Regression** (scaled features)
- **Random Forest Regressor** (non?linear, feature importances)
- **Gradient Boosting Regressor**

**Targets** evaluated:
- `gap_resid` (gap residualized vs Year)
- `gap_ratio_resid` (ratio residualized vs Year)

Feature importance is captured via Random Forest for both **state** and **district** models.
Top drivers often include:
- `skill_rate`, `informal_rate`
- `youth_unemployment_rate`
- `sector_share_A_Agriculture`
- `sector_share_P_Education`

## Visualizations in the Streamlit App
The dashboard (`app.py`) presents:

### Home
- Problem statement and how the project answers it
- India mismatch map (state points colored by gap ratio)

### Gap Trend (State)
- National mismatch trend over time

### Education vs Employment
- Education vs employment index trend

### Drivers (Feature Importance)
- Top drivers in state/district models with label explanations

### Socio?Economic Segments
- Poverty and income segment comparisons
- Live computed segment stats shown in the UI

### District Insights
- Top/Bottom mismatch districts and states
- Explanation of CPERV1 codes and calculation logic

## How the Segment Bars Are Calculated
In the app:
- `gap_simple = education_index / employment_rate` (CPERV1)
- Split data into **High vs Low** groups using **median** poverty_rate or per_capita
- Bar height = **mean gap_simple** for each group

## Summary
This project connects education, employment, socio?economic factors, and job?market structure to quantify and explain mismatch in India. It combines multiple government datasets, builds interpretable metrics, and presents results through an interactive dashboard focused on both trends and regional disparities.
