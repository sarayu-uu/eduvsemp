from pathlib import Path

import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt
import plotly.express as px
from eda.eda import (
    add_mismatch_index,
    add_time_features,
    build_national_trends,
    build_state_year_dataset,
    load_base,
)
from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler


BASE_DIR = Path(".")
EDA_DIR = BASE_DIR / "data" / "eda"
CLEAN_DIR = BASE_DIR / "data" / "cleaned"


def file_exists(path: Path) -> bool:
    return path.exists() and path.is_file()


def load_state_mismatch() -> pd.DataFrame:
    path = CLEAN_DIR / "cperv1_features_by_state.csv"
    if not path.exists():
        return pd.DataFrame()
    df = pd.read_csv(path)
    if "education_index" in df.columns and "employment_rate" in df.columns:
        denom = df["employment_rate"].replace(0, pd.NA)
        df["gap_ratio"] = df["education_index"] / denom
    return df


def state_centroids() -> dict[str, tuple[float, float]]:
    # Approximate state/UT centroids for India (lat, lon).
    return {
        "Andhra Pradesh": (15.9129, 79.7400),
        "Arunachal Pradesh": (28.2180, 94.7278),
        "Assam": (26.2006, 92.9376),
        "Bihar": (25.0961, 85.3131),
        "Chhattisgarh": (21.2787, 81.8661),
        "Goa": (15.2993, 74.1240),
        "Gujarat": (22.2587, 71.1924),
        "Haryana": (29.0588, 76.0856),
        "Himachal Pradesh": (31.1048, 77.1734),
        "Jharkhand": (23.6102, 85.2799),
        "Karnataka": (15.3173, 75.7139),
        "Kerala": (10.8505, 76.2711),
        "Madhya Pradesh": (22.9734, 78.6569),
        "Maharashtra": (19.7515, 75.7139),
        "Manipur": (24.6637, 93.9063),
        "Meghalaya": (25.4670, 91.3662),
        "Mizoram": (23.1645, 92.9376),
        "Nagaland": (26.1584, 94.5624),
        "Odisha": (20.9517, 85.0985),
        "Punjab": (31.1471, 75.3412),
        "Rajasthan": (27.0238, 74.2179),
        "Sikkim": (27.5330, 88.5122),
        "Tamil Nadu": (11.1271, 78.6569),
        "Telangana": (18.1124, 79.0193),
        "Tripura": (23.9408, 91.9882),
        "Uttar Pradesh": (26.8467, 80.9462),
        "Uttarakhand": (30.0668, 79.0193),
        "West Bengal": (22.9868, 87.8550),
        "Delhi": (28.7041, 77.1025),
        "Puducherry": (11.9416, 79.8083),
        "Jammu and Kashmir": (33.7782, 76.5762),
        "Andaman and Nicobar Islands": (11.7401, 92.6586),
        "Dadra and Nagar Haveli and Daman and Diu": (20.3974, 72.8328),
    }


def plot_mismatch_map(state_df: pd.DataFrame):
    if state_df.empty or "State" not in state_df.columns or "gap_ratio" not in state_df.columns:
        return None
    centers = state_centroids()
    plot_df = state_df.copy()
    plot_df["lat"] = plot_df["State"].map(lambda s: centers.get(s, (None, None))[0])
    plot_df["lon"] = plot_df["State"].map(lambda s: centers.get(s, (None, None))[1])
    plot_df = plot_df.dropna(subset=["lat", "lon", "gap_ratio"])

    fig = px.scatter_geo(
        plot_df,
        lat="lat",
        lon="lon",
        color="gap_ratio",
        color_continuous_scale=["#2E7D32", "#F9A825", "#C62828"],
        hover_name="State",
        size="gap_ratio",
        size_max=18,
        projection="natural earth",
    )
    fig.update_geos(
        scope="asia",
        showcountries=True,
        showland=True,
        landcolor="#F5F5F5",
        center={"lat": 22.5, "lon": 80.0},
        lataxis_range=[5, 37.5],
        lonaxis_range=[65, 100.0],
    )
    fig.update_layout(
        margin=dict(l=0, r=0, t=0, b=0),
        coloraxis_colorbar=dict(title="Gap Ratio"),
    )
    return fig


def segment_stats() -> dict[str, float]:
    try:
        df = load_base()
    except Exception:
        return {}
    df = add_mismatch_index(df)
    df = add_time_features(df)
    state_year = build_state_year_dataset(df)

    if "education_index_state" in state_year.columns and "cperv1_employment_rate" in state_year.columns:
        denom = state_year["cperv1_employment_rate"].replace(0, pd.NA)
        state_year["gap_simple"] = state_year["education_index_state"] / denom
    else:
        state_year["gap_simple"] = (
            state_year["gap_ratio"] if "gap_ratio" in state_year.columns else state_year["gap"]
        )

    out: dict[str, float] = {}
    if "poverty_rate" in state_year.columns:
        poverty_med = state_year["poverty_rate"].median()
        state_year["poverty_segment"] = state_year["poverty_rate"].apply(
            lambda x: "High Poverty" if x >= poverty_med else "Low Poverty"
        )
        seg = state_year.groupby("poverty_segment")["gap_simple"].mean().to_dict()
        out["poverty_high_mean"] = float(seg.get("High Poverty", float("nan")))
        out["poverty_low_mean"] = float(seg.get("Low Poverty", float("nan")))

    if "per_capita" in state_year.columns:
        income_med = state_year["per_capita"].median()
        state_year["income_segment"] = state_year["per_capita"].apply(
            lambda x: "High Income" if x >= income_med else "Low Income"
        )
        seg = state_year.groupby("income_segment")["gap_simple"].mean().to_dict()
        out["income_high_mean"] = float(seg.get("High Income", float("nan")))
        out["income_low_mean"] = float(seg.get("Low Income", float("nan")))

    return out


def segment_data() -> tuple[pd.DataFrame, pd.DataFrame]:
    try:
        df = load_base()
    except Exception:
        return pd.DataFrame(), pd.DataFrame()
    df = add_mismatch_index(df)
    df = add_time_features(df)
    state_year = build_state_year_dataset(df)

    if "education_index_state" in state_year.columns and "cperv1_employment_rate" in state_year.columns:
        denom = state_year["cperv1_employment_rate"].replace(0, pd.NA)
        state_year["gap_simple"] = state_year["education_index_state"] / denom
    else:
        state_year["gap_simple"] = (
            state_year["gap_ratio"] if "gap_ratio" in state_year.columns else state_year["gap"]
        )

    poverty_df = pd.DataFrame()
    income_df = pd.DataFrame()

    if "poverty_rate" in state_year.columns:
        poverty_med = state_year["poverty_rate"].median()
        state_year["poverty_segment"] = state_year["poverty_rate"].apply(
            lambda x: "High Poverty" if x >= poverty_med else "Low Poverty"
        )
        poverty_df = (
            state_year.groupby("poverty_segment")["gap_simple"]
            .mean()
            .reset_index()
            .rename(columns={"gap_simple": "avg_gap"})
        )

    if "per_capita" in state_year.columns:
        income_med = state_year["per_capita"].median()
        state_year["income_segment"] = state_year["per_capita"].apply(
            lambda x: "High Income" if x >= income_med else "Low Income"
        )
        income_df = (
            state_year.groupby("income_segment")["gap_simple"]
            .mean()
            .reset_index()
            .rename(columns={"gap_simple": "avg_gap"})
        )

    return poverty_df, income_df


@st.cache_data(show_spinner=False)
def district_model_insights() -> dict[str, pd.DataFrame | float]:
    path = CLEAN_DIR / "cperv1_features_by_district.csv"
    if not path.exists():
        return {}
    df = pd.read_csv(path)
    if "education_index" not in df.columns or "employment_rate" not in df.columns:
        return {}

    df["gap_ratio"] = df["education_index"] / df["employment_rate"].replace(0, pd.NA)
    df = df.dropna(subset=["gap_ratio"]).copy()
    feature_cols = ["skill_rate", "informal_rate"] + [
        c for c in df.columns if c.startswith("sector_share_")
    ]
    if not feature_cols:
        return {}

    X = df[feature_cols].copy()
    y = df["gap_ratio"].copy()
    district_label = (
        df["State"].fillna("State?")
        + " - "
        + df["District_Code"].astype(str)
        if "State" in df.columns and "District_Code" in df.columns
        else pd.Series([f"D{i}" for i in range(len(df))], index=df.index)
    )

    X_train, X_test, y_train, y_test, d_train, d_test = train_test_split(
        X, y, district_label, test_size=0.2, random_state=42
    )

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    lr = LinearRegression().fit(X_train_scaled, y_train)
    rf = RandomForestRegressor(random_state=42, n_estimators=300).fit(X_train, y_train)
    gbr = GradientBoostingRegressor(
        n_estimators=200, learning_rate=0.05, random_state=42
    ).fit(X_train, y_train)

    pred_lr = lr.predict(X_test_scaled)
    pred_rf = rf.predict(X_test)
    pred_gbr = gbr.predict(X_test)

    comparison = pd.DataFrame(
        [
            {
                "Model": "Linear Regression",
                "R2": r2_score(y_test, pred_lr),
                "RMSE": mean_squared_error(y_test, pred_lr) ** 0.5,
            },
            {
                "Model": "Random Forest",
                "R2": r2_score(y_test, pred_rf),
                "RMSE": mean_squared_error(y_test, pred_rf) ** 0.5,
            },
            {
                "Model": "Gradient Boosting",
                "R2": r2_score(y_test, pred_gbr),
                "RMSE": mean_squared_error(y_test, pred_gbr) ** 0.5,
            },
        ]
    )

    eval_df = pd.DataFrame(
        {
            "district_label": d_test.values,
            "actual_gap_ratio": y_test.values,
            "pred_gap_ratio_rf": pred_rf,
        }
    )
    eval_df["residual_rf"] = eval_df["actual_gap_ratio"] - eval_df["pred_gap_ratio_rf"]
    eval_df["abs_error_rf"] = eval_df["residual_rf"].abs()
    top_error = eval_df.nlargest(10, "abs_error_rf").copy()

    importance_df = (
        pd.DataFrame({"feature": X.columns, "importance": rf.feature_importances_})
        .sort_values("importance", ascending=False)
        .head(10)
    )

    return {
        "comparison": comparison,
        "eval": eval_df,
        "top_error": top_error,
        "importance": importance_df,
    }


@st.cache_data(show_spinner=False)
def time_trend_frames() -> tuple[pd.DataFrame, pd.DataFrame]:
    try:
        base = load_base()
    except Exception:
        return pd.DataFrame(), pd.DataFrame()
    base = add_mismatch_index(base)
    base = add_time_features(base)

    national = build_national_trends(base).copy()
    state_year = build_state_year_dataset(base).copy()

    if "Year" in national.columns:
        national["Year"] = national["Year"].astype(int)
    if "Year" in state_year.columns:
        state_year["Year"] = state_year["Year"].astype(int)
    return national, state_year


def main() -> None:
    st.set_page_config(page_title="Education–Employment Mismatch", layout="wide")
    st.markdown(
        """
        <style>
            .stApp { background-color: #FFFFFF; }
            h1, h2, h3, p, li, label, div { color: #000000; }
            section[data-testid="stSidebar"] {
                background-color: #F2F2F2;
            }
            section[data-testid="stSidebar"] * {
                color: #000000;
            }
            .stCode, .stTextInput, .stSelectbox, .stTextArea, .stMarkdown code {
                background-color: #F2F2F2 !important;
                color: #000000 !important;
            }
            div[data-baseweb="select"] > div {
                background-color: #F2F2F2 !important;
                color: #000000 !important;
            }
            code, pre {
                background-color: #F2F2F2 !important;
                color: #000000 !important;
            }
            .stSelectbox div[role="listbox"] {
                background-color: #FFFFFF !important;
                color: #000000 !important;
            }
            .stSelectbox div[role="option"] {
                color: #000000 !important;
            }
            ul[role="listbox"] {
                background-color: #FFFFFF !important;
                color: #000000 !important;
            }
            li[role="option"] {
                color: #000000 !important;
            }
        </style>
        """,
        unsafe_allow_html=True,
    )
    st.title("Education–Employment Mismatch in India")
    st.caption("Education is increasing faster than employment, creating a widening mismatch.")

    st.sidebar.header("Sections")
    graph_options = [
        "Home",
        "Time Trend",
        "Model Insights",
        "Gap Trend (State)",
        "Drivers (Feature Importance)",
        "Socio-Economic Segments",
    ]
    choice = st.sidebar.selectbox("Section", graph_options)

    if choice == "Home":
        st.header("Problem Statement")
        st.markdown(
            """
India's education levels are rising faster than employment opportunities, creating a growing mismatch.
The goal of this project is to quantify that mismatch over time and identify which socio-economic and
job-market factors explain it.
"""
        )
        st.markdown(
            """
**How this project answers the problem statement:**  
- Builds education and employment indexes and tracks the gap over time.  
- Shows where mismatch is highest (district/state).  
- Identifies drivers (skills, informality, sector mix, poverty, income).  
"""
        )

        # Insight cards
        c1, c2, c3 = st.columns(3)
        with c1:
            st.markdown("**Trend**\n\nMismatch is increasing over time")
        with c2:
            st.markdown("**Model Strength**\n\nDistrict model explains most variation (R² ~0.76)")
        with c3:
            st.markdown("**Key Drivers**\n\nSkills and informal employment matter most")

        # One strong visual
        st.header("Mismatch Hotspots Across India")
        state_df = load_state_mismatch()
        fig = plot_mismatch_map(state_df)
        if fig is not None:
            st.plotly_chart(fig, use_container_width=True)
            st.markdown(
                """
**How to read this map:**  
- Red points indicate states with the highest mismatch (education rising faster than jobs).  
- Green points indicate states where education and employment are more aligned.  
"""
            )
        else:
            st.info("State-level CPERV1 features not found. Run `python cleaning/cleaning.py` first.")

        st.header("Data Sources")
        st.markdown(
            """
- **Education:** AISHE  
- **Employment:** PLFS CPERV1  
- **Economic Indicators:** National statistics  
- **Job Market Structure:** Sector shares + job demand  
"""
        )

        with st.expander("Trend Analysis (State-Level) — details"):
            st.markdown(
                """
Purpose: explain mismatch after removing time trend.  
Target: gap_ratio_resid (education_index / employment_index), residualized by Year.  
Features: per_capita, poverty_rate, youth_unemployment_rate, skill_rate, informal_rate.  
Interpretation: moderate signal; time trend still dominates.
"""
            )
            st.caption("Feature guide for labels mentioned above:")
            st.markdown(
                """
- **per_capita**: per-capita income (economic indicator).  
- **poverty_rate**: share of population below the poverty line.  
- **youth_unemployment_rate**: unemployment rate among youth.  
- **skill_rate**: share with vocational training or recent training (CPERV1).  
- **informal_rate**: share in informal work (no contract or no social security).  
- **education_index**: average years of formal education (CPERV1).  
- **employment_index / employment_rate**: share employed (CPERV1, Principal_Status_Code 11-51).  
- **gap_ratio_resid**: (education_index / employment_index) after removing the time trend.
"""
            )

        st.header("Why the mismatch is high or low (interpretation)")
        st.markdown(
            """
High mismatch (red) typically appears where education levels rise but job creation is slower, especially in
informal or low-productivity sectors. Low mismatch (green) tends to appear where employment growth keeps pace
with education and where sector mix includes manufacturing or professional services that absorb educated workers.
"""
        )


    if choice == "Gap Trend (State)":
        st.header("Gap Trend (State)")
        national, _ = time_trend_frames()
        if national.empty:
            st.info("Missing trend inputs. Run `python cleaning/cleaning.py` and `python eda/eda.py` first.")
            return

        trend_long = national.melt(
            id_vars=["Year"],
            value_vars=["education_index", "employment_index"],
            var_name="series",
            value_name="index_value",
        )
        trend_long["series"] = trend_long["series"].map(
            {
                "education_index": "Education Index",
                "employment_index": "Employment Index",
            }
        )
        fig_dual = px.line(
            trend_long,
            x="Year",
            y="index_value",
            color="series",
            markers=True,
            title="Education vs Employment Index Over Time",
        )
        fig_dual.update_layout(yaxis_title="Normalized Index (0 to 1)", legend_title="")
        st.plotly_chart(fig_dual, use_container_width=True)
        st.markdown(
            """
**What this first chart means:**  
- Blue line = education trend over time.  
- Green line = employment trend over time.  
- If blue rises faster than green, mismatch is widening.
"""
        )

        gap_df = national[["Year", "gap"]].copy()
        gap_df["gap_sign"] = gap_df["gap"].apply(
            lambda x: "Education > Employment" if x >= 0 else "Employment > Education"
        )
        fig_gap = px.bar(
            gap_df,
            x="Year",
            y="gap",
            color="gap_sign",
            color_discrete_map={
                "Education > Employment": "#C62828",
                "Employment > Education": "#2E7D32",
            },
            title="Gap by Year (education_index - employment_index)",
        )
        fig_gap.add_hline(y=0, line_dash="dash", line_color="black")
        fig_gap.update_layout(yaxis_title="Gap Value", legend_title="")
        st.plotly_chart(fig_gap, use_container_width=True)
        st.markdown(
            """
**What this second chart means:**  
- This is the exact difference: **gap = education_index - employment_index**.  
- **Above 0 (red):** education index is higher than employment index.  
- **Below 0 (green):** employment index is higher than education index.  
- Bigger bars mean a larger mismatch in that year.
"""
        )
        st.caption("Feature guide:")
        st.markdown(
            """
- **education_index**: normalized education metric (higher = more education).  
- **employment_index**: normalized employment metric (higher = more jobs).  
- **gap**: education_index - employment_index.
"""
        )

    if choice == "Drivers (Feature Importance)":
        st.header("Drivers of Mismatch")
        district_plot = EDA_DIR / "rf_feature_importance_district.png"
        if file_exists(district_plot):
            st.image(str(district_plot), caption="Top drivers in district model")
        else:
            st.image(
                str(EDA_DIR / "rf_feature_importance.png"),
                caption="Top drivers (state model)",
            )
        st.markdown(
            """
**How to read this:**  
- Each bar is a district-level driver used by the model.  
- Longer bar = stronger contribution to mismatch differences across districts.  
"""
        )
        st.caption("Plain-English driver guide (what it means + what to do):")
        st.markdown(
            """
`District` here means each CPERV1 district unit (shown as State + District Code) across all districts in the dataset, not one single district.
"""
        )
        st.markdown(
            """
- **Farm Jobs** (`sector_share_A_Agriculture`): district has many agriculture workers.  
  Meaning: education may rise faster than local non-farm jobs.  
  Action: create non-farm pathways (agro-processing, logistics, rural services).
- **Education Jobs** (`sector_share_P_Education`): district has many workers in education services.  
  Meaning: local system can absorb education-related skills better.  
  Action: expand teacher training, ed-tech support, and allied roles.
- **Construction Jobs** (`sector_share_F_Construction`): district has many construction workers.  
  Meaning: jobs exist, but often low-formal and skill mismatch can remain.  
  Action: certify workers, improve safety, and link training to certified contractors.
- **Factory Jobs** (`sector_share_C_Manufacturing`): district has manufacturing base.  
  Meaning: better potential to absorb trained youth if skills match demand.  
  Action: align ITI/polytechnic curricula with local factory skill needs.
- **Professional Jobs** (`sector_share_M_Professional`): more technical/professional services.  
  Meaning: stronger knowledge-economy absorption.  
  Action: scale advanced digital, analytical, and service-sector training.
- **Trade Jobs** (`sector_share_G_Trade`): many wholesale/retail jobs.  
  Meaning: large job volume but quality/skill alignment may vary.  
  Action: upskill for supply-chain, digital retail, and formal sales roles.
- **Hospitality Jobs** (`sector_share_I_Accommodation`): accommodation/food-service concentration.  
  Meaning: service jobs available but often seasonal/informal.  
  Action: train for hospitality standards, language, and formal placement channels.
- **Unknown Sector** (`sector_share_Unknown`): missing or unclear industry coding share.  
  Meaning: weaker data quality can hide real mismatch causes.  
  Action: improve local labor-data capture before policy targeting.
- **Training Coverage** (`skill_rate`): share of people with vocational/recent training.  
  Meaning: indicates readiness of workforce for skilled jobs.  
  Action: expand demand-linked training where this is low.
- **Informal Work** (`informal_rate`): share of workers without contract/social security.  
  Meaning: high informality usually means weaker quality employment absorption.  
  Action: incentivize formal hiring, apprenticeships, and social-security-linked jobs.
"""
        )
        st.success(
            "How to use this: focus first on districts with the tallest bars, then apply the matching action above for those dominant drivers."
        )

    if False and choice == "Drivers (Feature Importance)":
        st.header("Drivers of Mismatch")
        district_plot = EDA_DIR / "rf_feature_importance_district.png"
        if file_exists(district_plot):
            st.image(str(district_plot), caption="Top drivers in district model")
        else:
            st.image(
                str(EDA_DIR / "rf_feature_importance.png"),
                caption="Top drivers (state model)",
            )
        st.markdown(
            """
**How to read this:**  
- Each bar is a feature used by the model.  
- Longer bar = stronger influence on the mismatch.  

**Label meanings:**  
- **skill_rate_resid**: % of people with training (de‑trended)  
- **informal_rate_resid**: % in informal work (de‑trended)  
- **youth_unemployment_resid**: youth unemployment signal (de‑trended)
"""
        )

        st.caption("Residualized vs raw features:")
        st.markdown(
            """
- ***_resid** features remove the time trend (effect of Year), so they show deviations from the overall trend.  
- **informal_rate_resid** is the informal-rate signal after removing time effects.  
- **informal_rate** is the raw share in informal work without removing the time trend.  
- **skill_rate_resid** and **youth_unemployment_resid** are the same idea: raw rates with the time trend removed.  
If both appear, **_resid** captures short-term or state/district-specific differences, while the raw rate reflects the overall level.
"""
        )

        st.caption("More label explanations (district model bars):")
        st.markdown(
            """
- **sector_share_***: share of workers in that NIC industry section within the district.  
- **sector_share_A_Agriculture**: share employed in agriculture.  
- **sector_share_P_Education**: share employed in education sector.  
- **sector_share_F_Construction**: share employed in construction.  
- **sector_share_C_Manufacturing**: share employed in manufacturing.  
- **sector_share_M_Professional**: share employed in professional/technical services.  
- **sector_share_G_Trade**: share employed in trade (wholesale/retail).  
- **sector_share_I_Accommodation**: share employed in accommodation/food services.  
- **sector_share_Unknown**: share with missing/unknown industry code.  
- **skill_rate**: share with vocational/recent training.  
- **informal_rate**: share in informal work (no contract or no social security).
"""
        )

    if choice == "Socio-Economic Segments":
        st.header("Gap by Poverty Segment (mean gap_simple = education_index / employment_rate)")
        poverty_df, income_df = segment_data()
        if not poverty_df.empty:
            fig_pov = px.bar(
                poverty_df,
                x="poverty_segment",
                y="avg_gap",
                color="poverty_segment",
                color_discrete_map={"High Poverty": "#C62828", "Low Poverty": "#2E7D32"},
            )
            fig_pov.update_layout(showlegend=False, yaxis_title="Average Gap (education / employment)")
            st.plotly_chart(fig_pov, use_container_width=True)
        else:
            st.info("Missing data to compute poverty segment chart.")
        st.markdown("Taller bars mean mismatch is worse in poorer regions.")
        st.caption("Feature guide:")
        st.markdown(
            """
- **gap_simple**: education_index / employment_rate (district/state-level CPERV1).  
- **poverty_rate**: share of population below the poverty line.
"""
        )

        st.header("Gap by Income Segment (mean gap_simple = education_index / employment_rate)")
        if not income_df.empty:
            fig_inc = px.bar(
                income_df,
                x="income_segment",
                y="avg_gap",
                color="income_segment",
                color_discrete_map={"High Income": "#2E7D32", "Low Income": "#C62828"},
            )
            fig_inc.update_layout(showlegend=False, yaxis_title="Average Gap (education / employment)")
            st.plotly_chart(fig_inc, use_container_width=True)
        else:
            st.info("Missing data to compute income segment chart.")
        st.markdown("Shorter bars mean mismatch is smaller in richer regions.")
        st.caption("Feature guide:")
        st.markdown(
            """
- **gap_simple**: education_index / employment_rate.  
- **per_capita**: per-capita income (higher = richer).
"""
        )

        stats = segment_stats()
        if stats:
            st.markdown(
                f"""
**How the bars are calculated:**  
- **gap_simple** = education_index / employment_rate (both from CPERV1).  
- **High vs Low** groups are created using the **median** poverty_rate or per_capita across rows.  
- The bar height is the **mean gap_simple** within each group.

**Poverty segment (current data):**  
- High Poverty group mean gap (approx): **{stats.get('poverty_high_mean', float('nan')):.2f}**  
- Low Poverty group mean gap (approx): **{stats.get('poverty_low_mean', float('nan')):.2f}**

**Income segment (current data):**  
- High Income group mean gap (approx): **{stats.get('income_high_mean', float('nan')):.2f}**  
- Low Income group mean gap (approx): **{stats.get('income_low_mean', float('nan')):.2f}**

**How this was analyzed (data pipeline):**  
These numbers are computed from `data/cleaned/final_merged.csv` merged with
`data/cleaned/cperv1_features_by_state.csv`, following the same logic used in `eda/eda.py`.
"""
            )
        else:
            st.info("Missing cleaned data files required to compute segment stats.")

    # Legacy section kept for reference; disabled after replacing with "Time Trend".
    if False and choice == "District Insights":
        st.header("District Insights (Detailed + Grouped)")

        st.subheader("Top/Bottom Districts (with codes)")
        top_path = EDA_DIR / "district_top10_gap.png"
        bottom_path = EDA_DIR / "district_bottom10_gap.png"
        if file_exists(top_path):
            st.image(str(top_path), caption="Top 10 districts by mismatch (State - District Code)")
            st.markdown("Higher bars = more education per employed person (bigger mismatch).")
        if file_exists(bottom_path):
            st.image(str(bottom_path), caption="Bottom 10 districts by mismatch (State - District Code)")
            st.markdown("Lower bars = education and employment are closer (smaller mismatch).")

        st.subheader("State Summary (averaged across districts)")
        top_state = EDA_DIR / "state_top10_gap.png"
        bottom_state = EDA_DIR / "state_bottom10_gap.png"
        if file_exists(top_state):
            st.image(str(top_state), caption="Top 10 states by mismatch (average across districts)")
            st.markdown("State averages can hide extreme districts.")
        if file_exists(bottom_state):
            st.image(str(bottom_state), caption="Bottom 10 states by mismatch (average across districts)")
            st.markdown("Lower bars = better match on average.")

        st.subheader("What the labels mean")
        st.markdown(
            """
**State - District Code** = official CPERV1 district code within each state.  
We do not have district names, so codes are shown.  
**Gap Ratio** = education_index / employment_rate.  
Higher values mean worse mismatch.
"""
        )
        st.caption("Feature guide:")
        st.markdown(
            """
- **Years_Formal_Education**: years of formal education per person (CPERV1).  
- **education_index**: average Years_Formal_Education in the district/state.  
- **Principal_Status_Code**: employment status code (CPERV1).  
  Employed = 11-51, Unemployed = 81-82.  
- **employment_rate**: share of people with Principal_Status_Code 11-51.  
- **Gap Ratio**: education_index / employment_rate.
"""
        )

        st.subheader("Sources and Calculation")
        st.markdown(
            """
**Source dataset:** PLFS CPERV1 person file (`data/raw/cperv1.csv`).  
**How it’s calculated:**  
1. Compute **education_index** = average `Years_Formal_Education` per district.  
2. Compute **employment_rate** = share of people with `Principal_Status_Code` in 11–51.  
3. **Gap Ratio** = education_index / employment_rate.  
4. District gaps are averaged to get state-level bars.
"""
        )

    if choice == "Time Trend":
        st.header("Time Trend")
        national, state_year = time_trend_frames()
        if national.empty:
            st.info("Missing trend inputs. Run `python cleaning/cleaning.py` and `python eda/eda.py` first.")
            return

        # 1) State mismatch over time.
        if "gap_ratio" in state_year.columns:
            trend_ratio = (
                state_year.groupby("Year", dropna=False)["gap_ratio"]
                .mean()
                .reset_index()
                .dropna()
            )
            fig1 = px.line(
                trend_ratio,
                x="Year",
                y="gap_ratio",
                markers=True,
                title="Mismatch Over Time (Average State Gap Ratio)",
            )
            fig1.update_layout(yaxis_title="Average Gap Ratio")
            st.plotly_chart(fig1, use_container_width=True)

        # 7) State-wise small multiples (top 12 by average mismatch).
        if "State" in state_year.columns and "gap_ratio" in state_year.columns:
            state_trend = (
                state_year.groupby(["State", "Year"], dropna=False)["gap_ratio"]
                .mean()
                .reset_index()
                .dropna(subset=["State", "gap_ratio"])
            )
            top_states = (
                state_trend.groupby("State")["gap_ratio"]
                .mean()
                .sort_values(ascending=False)
                .head(12)
                .index.tolist()
            )
            small = state_trend[state_trend["State"].isin(top_states)].copy()
            fig7 = px.line(
                small,
                x="Year",
                y="gap_ratio",
                facet_col="State",
                facet_col_wrap=4,
                title="State-wise Mismatch Trends (Top 12 by Average Gap Ratio)",
            )
            fig7.update_layout(showlegend=False, yaxis_title="Gap Ratio", height=900)
            st.plotly_chart(fig7, use_container_width=True)

        # 10) National gap with milestones.
        fig10 = px.line(
            national,
            x="Year",
            y="gap",
            markers=True,
            title="National Gap with Milestones",
        )
        fig10.update_layout(yaxis_title="Education Index - Employment Index")
        years = national["Year"].dropna().astype(int).tolist()
        if years:
            mid_year = years[len(years) // 2]
            milestone_years = [years[0], mid_year, years[-1]]
            for yr in milestone_years:
                fig10.add_vline(x=yr, line_dash="dot", line_color="gray")
        st.plotly_chart(fig10, use_container_width=True)

        st.markdown(
            """
**How to read this section:**  
- These are descriptive trend visuals only (no driver-causality claims).  
- They directly support the problem statement by showing how mismatch changes over time.
"""
        )

    if choice == "Model Insights":
        st.header("Model Insights (District Model)")
        st.markdown(
            """
This section shows how the district model performs in predicting mismatch.
It directly supports the problem statement by showing whether the model can reliably identify districts with higher education-employment mismatch.
"""
        )

        insights = district_model_insights()
        if not insights:
            st.info("Missing district data. Run `python cleaning/cleaning.py` and `python eda/eda.py` first.")
            return

        comparison = insights["comparison"]
        eval_df = insights["eval"]
        top_error = insights["top_error"]
        importance_df = insights["importance"]

        c1, c2 = st.columns(2)
        with c1:
            fig_r2 = px.bar(
                comparison,
                x="Model",
                y="R2",
                color="Model",
                title="Test R2 by Model (District)",
            )
            fig_r2.update_layout(showlegend=False)
            st.plotly_chart(fig_r2, use_container_width=True)
        with c2:
            fig_rmse = px.bar(
                comparison,
                x="Model",
                y="RMSE",
                color="Model",
                title="Test RMSE by Model (District)",
            )
            fig_rmse.update_layout(showlegend=False)
            st.plotly_chart(fig_rmse, use_container_width=True)

        fig_scatter = px.scatter(
            eval_df,
            x="actual_gap_ratio",
            y="pred_gap_ratio_rf",
            title="Random Forest: Actual vs Predicted (District Test Set)",
            hover_name="district_label",
            opacity=0.7,
        )
        lim = max(
            float(eval_df["actual_gap_ratio"].max()),
            float(eval_df["pred_gap_ratio_rf"].max()),
        )
        fig_scatter.add_shape(
            type="line",
            x0=0,
            y0=0,
            x1=lim,
            y1=lim,
            line=dict(color="black", dash="dash"),
        )
        st.plotly_chart(fig_scatter, use_container_width=True)

        fig_resid = px.histogram(
            eval_df,
            x="residual_rf",
            nbins=30,
            title="Random Forest Residual Distribution (District Test Set)",
        )
        st.plotly_chart(fig_resid, use_container_width=True)

        c3, c4 = st.columns(2)
        with c3:
            fig_imp = px.bar(
                importance_df.sort_values("importance"),
                x="importance",
                y="feature",
                orientation="h",
                title="Top 10 Random Forest Feature Importances",
            )
            st.plotly_chart(fig_imp, use_container_width=True)
        with c4:
            fig_err = px.bar(
                top_error.sort_values("abs_error_rf"),
                x="abs_error_rf",
                y="district_label",
                orientation="h",
                title="Top 10 Districts with Largest Prediction Error",
            )
            st.plotly_chart(fig_err, use_container_width=True)

        st.markdown(
            """
**How to interpret these visuals:**  
- Higher **R2** and lower **RMSE** indicate better predictive quality.  
- In the Actual vs Predicted chart, points closer to the diagonal are better predictions.  
- Residuals centered near zero indicate fewer systematic errors.  
- Feature importance shows which district characteristics contribute most to mismatch prediction.
"""
        )



if __name__ == "__main__":
    main()
