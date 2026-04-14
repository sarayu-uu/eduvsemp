from pathlib import Path

import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt
import plotly.express as px
from eda.eda import add_mismatch_index, add_time_features, build_state_year_dataset, load_base


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
        "District Insights",
        "Gap Trend (State)",
        "Education vs Employment",
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
        st.header("Gap Trend (National, Averaged Across States)")
        st.image(
            str(EDA_DIR / "gap_trend.png"),
            caption="Average gap over time (state averages combined)",
        )
        st.markdown(
            """
This shows whether mismatch is increasing over time (national trend).  
Higher values mean education is growing faster than jobs.
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

    if choice == "Education vs Employment":
        st.header("Education vs Employment")
        st.image(
            str(EDA_DIR / "education_vs_employment_trend.png"),
            caption="Education and employment indexes over time",
        )
        st.markdown(
            """
Education rising faster than employment means mismatch is increasing.
"""
        )
        st.caption("Feature guide:")
        st.markdown(
            """
- **education_index**: normalized education metric (higher = more education).  
- **employment_index**: normalized employment metric (higher = more jobs).
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

    if choice == "District Insights":
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



if __name__ == "__main__":
    main()
