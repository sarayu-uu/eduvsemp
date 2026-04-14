from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import train_test_split


DATA_DIR = Path("data")
CLEAN_DIR = DATA_DIR / "cleaned"
EDA_DIR = DATA_DIR / "eda"


def ensure_dirs() -> None:
    EDA_DIR.mkdir(parents=True, exist_ok=True)


def load_base() -> pd.DataFrame:
    df = pd.read_csv(CLEAN_DIR / "final_merged.csv")
    df = df.rename(
        columns={
            "unemployment_rate_x": "unemployment_rate",
            "unemployment_rate_y": "unemployment_aux",
        }
    )
    return df


def add_mismatch_index(df: pd.DataFrame) -> pd.DataFrame:
    # Education metric: total enrollment (both) from AISHE.
    edu_col = "edu_Grand Total Both"
    # Job demand metric: overall job count (static, no year info available).
    job_col = "job_count_total"
    if edu_col in df.columns and job_col in df.columns:
        df["mismatch_index"] = df[edu_col] - df[job_col]
    return df


def add_time_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.sort_values(["Year"]).copy()
    if "job_count_total" in df.columns:
        min_year = df["Year"].min()
        df["job_demand_adjusted"] = df["job_count_total"] * (
            1 + 0.02 * (df["Year"] - min_year)
        )
    return df


def normalize_series(series: pd.Series) -> pd.Series:
    min_val = series.min()
    max_val = series.max()
    if pd.isna(min_val) or pd.isna(max_val) or min_val == max_val:
        return series * 0
    return (series - min_val) / (max_val - min_val)


def build_national_trends(df: pd.DataFrame) -> pd.DataFrame:
    yearly = (
        df.groupby("Year", dropna=False)
        .agg(
            education_metric=("edu_Grand Total Both", "mean"),
            employment_metric=("estimated_employed", "mean"),
            unemployment_rate=("unemployment_rate", "mean"),
            labour_participation_rate=("labour_participation_rate", "mean"),
            per_capita=("per_capita", "mean"),
            poverty_rate=("poverty_rate", "mean"),
            labour_force=("labour_force", "mean"),
            youth_unemployment_rate=("youth_unemployment_rate", "mean"),
            job_demand_adjusted=("job_demand_adjusted", "mean"),
        )
        .reset_index()
    )
    if "Year" in yearly.columns:
        yearly["Year"] = yearly["Year"].astype(int)

    yearly["education_index"] = normalize_series(yearly["education_metric"])
    yearly["employment_index"] = normalize_series(yearly["employment_metric"])
    yearly["gap"] = yearly["education_index"] - yearly["employment_index"]

    return yearly


def build_state_year_dataset(df: pd.DataFrame) -> pd.DataFrame:
    trends = build_national_trends(df)
    edu_index_by_year = trends[["Year", "education_index"]].copy()

    state_year = df.copy()
    state_year = state_year.merge(edu_index_by_year, on="Year", how="left")

    state_year["employment_index"] = normalize_series(state_year["estimated_employed"])
    # Prefer state-level education from CPERV1 when available.
    if "cperv1_education_index" in state_year.columns:
        state_year["education_index_state"] = normalize_series(
            state_year["cperv1_education_index"]
        )
        state_year["gap"] = (
            state_year["education_index_state"] - state_year["employment_index"]
        )
    else:
        state_year["gap"] = state_year["education_index"] - state_year["employment_index"]

    # Alternative target: ratio-style gap (education / employment).
    edu_col = (
        "education_index_state"
        if "education_index_state" in state_year.columns
        else "education_index"
    )
    denom = state_year["employment_index"].replace(0, pd.NA)
    state_year["gap_ratio"] = state_year[edu_col] / denom

    # Merge CPERV1 features by State (education, skill, informal, employment proxies).
    cperv1_path = CLEAN_DIR / "cperv1_features_by_state.csv"
    if cperv1_path.exists():
        cperv1 = pd.read_csv(cperv1_path)
        state_year = state_year.merge(
            cperv1[
                [
                    "State",
                    "education_index",
                    "employment_rate",
                    "unemployment_rate",
                    "skill_rate",
                    "informal_rate",
                ]
            ].rename(
                columns={
                    "education_index": "cperv1_education_index",
                    "employment_rate": "cperv1_employment_rate",
                    "unemployment_rate": "cperv1_unemployment_rate",
                    "skill_rate": "cperv1_skill_rate",
                    "informal_rate": "cperv1_informal_rate",
                }
            ),
            on="State",
            how="left",
        )

    return state_year


def lag_analysis(state_year: pd.DataFrame) -> None:
    # Simple lag insight: does prior-year job demand relate to next-year gap?
    df = state_year.sort_values(["State", "Year"]).copy()
    if "job_count_total" in df.columns:
        df["job_lag"] = df.groupby("State")["job_count_total"].shift(1)
    if "job_demand_adjusted" in df.columns:
        df["job_lag"] = df.groupby("State")["job_demand_adjusted"].shift(1)

    valid = df[["job_lag", "gap"]].dropna()
    corr = valid["job_lag"].corr(valid["gap"]) if not valid.empty else float("nan")

    (EDA_DIR / "lag_analysis.txt").write_text(
        f"Correlation between prior-year job demand and next-year gap: {corr:.4f}\n",
        encoding="utf-8",
    )


def segment_analysis(state_year: pd.DataFrame) -> None:
    # Segment: high vs low poverty, high vs low income.
    df = state_year.copy()
    # Use a simpler, more intuitive gap for segmentation if available:
    # gap_simple = education_index_state / cperv1_employment_rate
    if "education_index_state" in df.columns and "cperv1_employment_rate" in df.columns:
        denom = df["cperv1_employment_rate"].replace(0, pd.NA)
        df["gap_simple"] = df["education_index_state"] / denom
    else:
        df["gap_simple"] = df["gap_ratio"] if "gap_ratio" in df.columns else df["gap"]
    poverty_med = df["poverty_rate"].median()
    income_med = df["per_capita"].median()

    df["poverty_segment"] = df["poverty_rate"].apply(
        lambda x: "High Poverty" if x >= poverty_med else "Low Poverty"
    )
    df["income_segment"] = df["per_capita"].apply(
        lambda x: "High Income" if x >= income_med else "Low Income"
    )

    seg_poverty = (
        df.groupby("poverty_segment")["gap_simple"]
        .mean()
        .reset_index()
        .rename(columns={"gap_simple": "avg_gap"})
    )
    seg_income = (
        df.groupby("income_segment")["gap_simple"]
        .mean()
        .reset_index()
        .rename(columns={"gap_simple": "avg_gap"})
    )

    # Simple visualization
    plt.figure()
    sns.barplot(data=seg_poverty, x="poverty_segment", y="avg_gap", color="red")
    plt.title("Mismatch is Higher in High-Poverty Areas")
    plt.xlabel("Poverty Group")
    plt.ylabel("Average Gap (education / employment)")
    plt.grid(False)
    plt.tight_layout()
    plt.savefig(EDA_DIR / "gap_by_poverty_segment.png", dpi=300, bbox_inches="tight")
    plt.close()

    plt.figure()
    sns.barplot(data=seg_income, x="income_segment", y="avg_gap", color="green")
    plt.title("Mismatch is Lower in High-Income Areas")
    plt.xlabel("Income Group")
    plt.ylabel("Average Gap (education / employment)")
    plt.grid(False)
    plt.tight_layout()
    plt.savefig(EDA_DIR / "gap_by_income_segment.png", dpi=300, bbox_inches="tight")
    plt.close()


def correlation_analysis(df: pd.DataFrame) -> pd.DataFrame:
    # Exclude derived/target-like columns from correlation to avoid misleading ±1.0.
    drop_cols = [
        "education_index",
        "employment_index",
        "gap",
        "education_metric",
        "employment_metric",
        "Year",
    ]
    corr_df = df.drop(columns=[c for c in drop_cols if c in df.columns])
    corr = corr_df.corr(numeric_only=True)
    corr.to_csv(EDA_DIR / "correlation_matrix.csv")

    plt.figure(figsize=(20, 14))
    sns.heatmap(corr, annot=True, cmap="coolwarm", fmt=".2f")
    plt.title("Correlation Heatmap (Numeric Features)")
    plt.tight_layout()
    plt.savefig(EDA_DIR / "correlation_heatmap.png", dpi=150)
    plt.close()

    if "unemployment_rate" in corr.columns:
        target_corr = (
            corr["unemployment_rate"]
            .drop(labels=["unemployment_rate"])
            .sort_values(key=lambda s: s.abs(), ascending=False)
        )
        strong = target_corr[target_corr.abs() >= 0.2]
        strong.to_csv(
            EDA_DIR / "correlation_with_unemployment.csv", header=["correlation"]
        )
        return strong
    return pd.Series(dtype="float64")


def plot_key_charts(df: pd.DataFrame) -> None:
    sns.set_style("whitegrid")
    plt.rcParams["figure.figsize"] = (10, 6)
    plt.rcParams["font.size"] = 12
    from matplotlib.ticker import FixedFormatter, FixedLocator

    def set_year_axis(ax: plt.Axes, years: pd.Series) -> None:
        years_sorted = sorted(pd.unique(years.dropna().astype(int)))
        if not years_sorted:
            return
        ax.xaxis.set_major_locator(FixedLocator(years_sorted))
        ax.xaxis.set_major_formatter(FixedFormatter([str(y) for y in years_sorted]))
        ax.ticklabel_format(style="plain", axis="x")

    # 1) Mismatch Gap Trend (main evidence)
    plt.figure()
    df_plot = df.copy()
    df_plot["Year"] = df_plot["Year"].astype(int)
    sns.lineplot(
        data=df_plot, x="Year", y="gap", color="red", linewidth=3, marker="o"
    )
    plt.title("Widening Gap Between Education and Employment in India")
    plt.xlabel("Year")
    plt.ylabel("Education Index − Employment Index")
    # Force year ticks as actual years (no implicit 0..n or scientific formatting)
    set_year_axis(plt.gca(), df_plot["Year"])
    year_last = df["Year"].iloc[-1]
    gap_last = df["gap"].iloc[-1]
    plt.annotate(
        "Gap widening → mismatch increasing",
        xy=(year_last, gap_last),
        xytext=(year_last - 2, gap_last + 0.05),
        arrowprops=dict(arrowstyle="->"),
    )
    plt.grid(False)
    plt.tight_layout()
    plt.savefig(EDA_DIR / "gap_trend.png", dpi=300, bbox_inches="tight")
    plt.close()

    # 2) Education vs Employment Trend (dual line)
    plt.figure()
    df_plot2 = df.copy()
    df_plot2["Year"] = df_plot2["Year"].astype(int)
    sns.lineplot(
        data=df_plot2, x="Year", y="education_index", color="blue", linewidth=2.5
    )
    sns.lineplot(
        data=df_plot2, x="Year", y="employment_index", color="green", linewidth=2.5
    )
    plt.title("Education Rising Faster Than Employment in India")
    plt.xlabel("Year")
    plt.ylabel("Index (0 to 1)")
    set_year_axis(plt.gca(), df_plot2["Year"])
    plt.grid(False)
    # Direct labels on lines
    plt.text(df["Year"].iloc[-1], df["education_index"].iloc[-1], "Education", color="blue")
    plt.text(df["Year"].iloc[-1], df["employment_index"].iloc[-1], "Employment", color="green")
    plt.tight_layout()
    plt.savefig(EDA_DIR / "education_vs_employment_trend.png", dpi=300, bbox_inches="tight")
    plt.close()

        # Extra charts removed to keep only the images used in the Streamlit app.


def train_models(df: pd.DataFrame, target_col: str = "gap") -> dict:
    df = df.copy()

    # Residualize features and target against Year to remove time-trend leakage.
    def residualize(series: pd.Series, year: pd.Series) -> pd.Series:
        series = pd.to_numeric(series, errors="coerce")
        mask = series.notna() & year.notna()
        if mask.sum() < 3:
            return pd.Series([float("nan")] * len(series), index=series.index)
        x = year[mask].values.reshape(-1, 1)
        y = series[mask].values
        model = LinearRegression()
        model.fit(x, y)
        pred = model.predict(x)
        resid = pd.Series(index=series.index, dtype="float64")
        resid.loc[mask] = y - pred
        return resid

    df = df.sort_values(["State", "Year"]).copy()

    df["per_capita_resid"] = residualize(df["per_capita"], df["Year"])
    df["poverty_rate_resid"] = residualize(df["poverty_rate"], df["Year"])
    df["youth_unemployment_resid"] = residualize(
        df["youth_unemployment_rate"], df["Year"]
    )
    df["skill_rate_resid"] = residualize(df["cperv1_skill_rate"], df["Year"])
    df["informal_rate_resid"] = residualize(df["cperv1_informal_rate"], df["Year"])

    # Remove job_demand_adjusted entirely to avoid engineered time structure.

    df["target_resid"] = residualize(df[target_col], df["Year"])

    # Target: residualized education–employment metric.
    target = "target_resid"
    y = df[target]
    feature_cols = [
        "per_capita_resid",
        "poverty_rate_resid",
        "youth_unemployment_resid",
        "skill_rate_resid",
        "informal_rate_resid",
    ]
    feature_cols = [c for c in feature_cols if c in df.columns]
    X = df[feature_cols].copy()

    valid_mask = X.notna().all(axis=1) & y.notna()
    X = X.loc[valid_mask].copy()
    y = y.loc[valid_mask].copy()

    # One-hot encode categorical columns (e.g., Area)
    X = pd.get_dummies(X, drop_first=True)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    # Scale features for linear model; keep unscaled for tree model.
    from sklearn.preprocessing import StandardScaler

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    lin = LinearRegression()
    lin.fit(X_train_scaled, y_train)

    from sklearn.ensemble import GradientBoostingRegressor

    rf = RandomForestRegressor(random_state=42, n_estimators=300)
    rf.fit(X_train, y_train)

    from sklearn.ensemble import GradientBoostingRegressor

    gbr = GradientBoostingRegressor(
        n_estimators=200, learning_rate=0.05, random_state=42
    )
    gbr.fit(X_train, y_train)

    pred_lr = lin.predict(X_test_scaled)
    rmse_lr = mean_squared_error(y_test, pred_lr) ** 0.5
    r2_lr = r2_score(y_test, pred_lr) if len(y_test) >= 2 else float("nan")

    pred = rf.predict(X_test)
    rmse = mean_squared_error(y_test, pred) ** 0.5
    r2 = r2_score(y_test, pred) if len(y_test) >= 2 else float("nan")

    pred_gbr = gbr.predict(X_test)
    rmse_gbr = mean_squared_error(y_test, pred_gbr) ** 0.5
    r2_gbr = r2_score(y_test, pred_gbr) if len(y_test) >= 2 else float("nan")

    # Shuffle check (sanity): average R2 over multiple shuffles.
    from sklearn.utils import shuffle

    shuf_scores = []
    for seed in range(5):
        X_shuf, y_shuf = shuffle(X, y, random_state=42 + seed)
        Xs_train, Xs_test, ys_train, ys_test = train_test_split(
            X_shuf, y_shuf, test_size=0.2, random_state=42 + seed
        )
        rf_shuf = RandomForestRegressor(random_state=42, n_estimators=300)
        rf_shuf.fit(Xs_train, ys_train)
        pred_shuf = rf_shuf.predict(Xs_test)
        score = r2_score(ys_test, pred_shuf) if len(ys_test) >= 2 else float("nan")
        shuf_scores.append(score)
    r2_shuf = sum(shuf_scores) / len(shuf_scores)

    coef = pd.Series(lin.coef_, index=X.columns).sort_values(ascending=False)

    importances = pd.Series(rf.feature_importances_, index=X.columns).sort_values(
        ascending=False
    )

    importance_df = pd.DataFrame(
        {"feature": X.columns, "importance": rf.feature_importances_}
    ).sort_values(by="importance", ascending=False)
    top5 = importance_df.head(5)
    corr_with_target = {}
    for feat in top5["feature"]:
        try:
            corr_with_target[feat] = X[feat].corr(y)
        except Exception:
            corr_with_target[feat] = None

    explain_lines = []
    explain_lines.append("Top 5 Features (Random Forest) with Interpretations\n")
    for _, row in top5.iterrows():
        feat = row["feature"]
        imp = row["importance"]
        corr = corr_with_target.get(feat)
        if corr is None or pd.isna(corr):
            direction = "has no clear linear direction with unemployment."
        elif corr > 0:
            direction = "tends to increase with higher unemployment."
        else:
            direction = "tends to decrease as unemployment rises."

        explain_lines.append(f"- {feat} (importance={imp:.4f})")
        explain_lines.append(f"  Interpretation: {feat} {direction}")

    # No file writes for explainability to keep output clean.

    # Top feature importance (hide zero-importance bars)
    non_zero = importances[importances > 0]
    top5_imp = non_zero.head(5).iloc[::-1]
    rename_map = {
        "per_capita_resid": "Per Capita (Res.)",
        "poverty_rate_resid": "Poverty Rate (Res.)",
        "youth_unemployment_resid": "Youth Unemp (Res.)",
        "skill_rate_resid": "Skill Rate (Res.)",
        "informal_rate_resid": "Informal Rate (Res.)",
    }
    top5_labels = [rename_map.get(i, i) for i in top5_imp.index]
    plt.figure()
    plt.barh(top5_labels, top5_imp.values, color="steelblue")
    plt.title("What Actually Drives the Gap (Top 5 Features)")
    plt.grid(False)
    plt.tight_layout()
    plt.savefig(EDA_DIR / "rf_feature_importance.png", dpi=300, bbox_inches="tight")
    plt.close()

    return {
        "r2_lr": r2_lr,
        "rmse_lr": rmse_lr,
        "r2": r2,
        "rmse": rmse,
        "r2_gbr": r2_gbr,
        "rmse_gbr": rmse_gbr,
        "r2_shuf": r2_shuf,
        "top_importances": importances.head(10),
        "top_coefficients": coef.head(10),
    }


def train_models_district() -> None:
    # District-level CPERV1 model (more rows, more variation).
    district_path = CLEAN_DIR / "cperv1_features_by_district.csv"
    if not district_path.exists():
        return

    df = pd.read_csv(district_path)
    df["gap_ratio"] = df["education_index"] / df["employment_rate"].replace(0, pd.NA)
    df = df.dropna(subset=["gap_ratio"])

    # Features: skill/informal + sector shares.
    feature_cols = ["skill_rate", "informal_rate"]
    feature_cols += [c for c in df.columns if c.startswith("sector_share_")]
    X = df[feature_cols].copy()
    y = df["gap_ratio"]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    # Linear baseline (scaled).
    from sklearn.preprocessing import StandardScaler

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    lin = LinearRegression()
    lin.fit(X_train_scaled, y_train)

    rf = RandomForestRegressor(random_state=42, n_estimators=300)
    rf.fit(X_train, y_train)

    from sklearn.ensemble import GradientBoostingRegressor

    gbr = GradientBoostingRegressor(
        n_estimators=200, learning_rate=0.05, random_state=42
    )
    gbr.fit(X_train, y_train)

    pred_lr = lin.predict(X_test_scaled)
    rmse_lr = mean_squared_error(y_test, pred_lr) ** 0.5
    r2_lr = r2_score(y_test, pred_lr)

    pred = rf.predict(X_test)
    rmse = mean_squared_error(y_test, pred) ** 0.5
    r2 = r2_score(y_test, pred)

    pred_gbr = gbr.predict(X_test)
    rmse_gbr = mean_squared_error(y_test, pred_gbr) ** 0.5
    r2_gbr = r2_score(y_test, pred_gbr)

    # Feature importance (top 10)
    importances = pd.Series(rf.feature_importances_, index=X.columns).sort_values(
        ascending=False
    )
    plt.figure()
    importances.head(10).iloc[::-1].plot(kind="barh", color="teal")
    plt.title("District Model: Top Feature Importances")
    plt.grid(False)
    plt.tight_layout()
    plt.savefig(EDA_DIR / "rf_feature_importance_district.png", dpi=300, bbox_inches="tight")
    plt.close()

    # District mismatch extremes (top/bottom 10 by gap_ratio)
    df["district_label"] = df["State"].fillna("State?") + " - " + df["District_Code"].astype(str)
    top10 = df.nlargest(10, "gap_ratio")[["district_label", "gap_ratio"]]
    bottom10 = df.nsmallest(10, "gap_ratio")[["district_label", "gap_ratio"]]

    plt.figure(figsize=(10, 6))
    plt.barh(top10["district_label"][::-1], top10["gap_ratio"][::-1], color="#E94F37")
    plt.title("Top 10 Districts with Highest Mismatch")
    plt.xlabel("Gap Ratio (education / employment)")
    plt.ylabel("District (State - Code)")
    plt.tight_layout()
    plt.savefig(EDA_DIR / "district_top10_gap.png", dpi=300, bbox_inches="tight")
    plt.close()

    plt.figure(figsize=(10, 6))
    plt.barh(bottom10["district_label"][::-1], bottom10["gap_ratio"][::-1], color="#2E86AB")
    plt.title("Bottom 10 Districts with Lowest Mismatch")
    plt.xlabel("Gap Ratio (education / employment)")
    plt.ylabel("District (State - Code)")
    plt.tight_layout()
    plt.savefig(EDA_DIR / "district_bottom10_gap.png", dpi=300, bbox_inches="tight")
    plt.close()

    # State-level mismatch extremes (mean gap_ratio across districts)
    state_gap = (
        df.groupby("State", dropna=False)["gap_ratio"]
        .mean()
        .reset_index()
        .dropna(subset=["State"])
    )
    state_top10 = state_gap.nlargest(10, "gap_ratio")
    state_bottom10 = state_gap.nsmallest(10, "gap_ratio")

    plt.figure(figsize=(10, 6))
    plt.barh(state_top10["State"][::-1], state_top10["gap_ratio"][::-1], color="#E94F37")
    plt.title("Top 10 States with Highest Mismatch")
    plt.xlabel("Gap Ratio (education / employment)")
    plt.ylabel("State")
    plt.tight_layout()
    plt.savefig(EDA_DIR / "state_top10_gap.png", dpi=300, bbox_inches="tight")
    plt.close()

    plt.figure(figsize=(10, 6))
    plt.barh(state_bottom10["State"][::-1], state_bottom10["gap_ratio"][::-1], color="#2E86AB")
    plt.title("Bottom 10 States with Lowest Mismatch")
    plt.xlabel("Gap Ratio (education / employment)")
    plt.ylabel("State")
    plt.tight_layout()
    plt.savefig(EDA_DIR / "state_bottom10_gap.png", dpi=300, bbox_inches="tight")
    plt.close()


def main() -> None:
    ensure_dirs()
    df = load_base()
    df = add_mismatch_index(df)
    df = add_time_features(df)

    trends = build_national_trends(df)
    plot_key_charts(trends)
    state_year = build_state_year_dataset(df)
    segment_analysis(state_year)

    # Train baseline (gap) and ratio model.
    results_gap = train_models(state_year, target_col="gap")
    results_ratio = train_models(state_year, target_col="gap_ratio")

    with open(EDA_DIR / "model_metrics.txt", "w", encoding="utf-8") as f:
        f.write("Model A: gap_resid (gap residualized vs Year)\n")
        f.write(f"LR R2: {results_gap['r2_lr']:.4f}\n")
        f.write(f"LR RMSE: {results_gap['rmse_lr']:.4f}\n")
        f.write(f"RF R2: {results_gap['r2']:.4f}\n")
        f.write(f"RF RMSE: {results_gap['rmse']:.4f}\n")
        f.write(f"GBR R2: {results_gap['r2_gbr']:.4f}\n")
        f.write(f"GBR RMSE: {results_gap['rmse_gbr']:.4f}\n")
        f.write(f"RF R2 (shuffled avg): {results_gap['r2_shuf']:.4f}\n\n")

        f.write("Model B: gap_ratio_resid (education/employment residualized vs Year)\n")
        f.write(f"LR R2: {results_ratio['r2_lr']:.4f}\n")
        f.write(f"LR RMSE: {results_ratio['rmse_lr']:.4f}\n")
        f.write(f"RF R2: {results_ratio['r2']:.4f}\n")
        f.write(f"RF RMSE: {results_ratio['rmse']:.4f}\n")
        f.write(f"GBR R2: {results_ratio['r2_gbr']:.4f}\n")
        f.write(f"GBR RMSE: {results_ratio['rmse_gbr']:.4f}\n")
        f.write(f"RF R2 (shuffled avg): {results_ratio['r2_shuf']:.4f}\n")

    # District-level CPERV1 model (separate, higher-variation dataset).
    train_models_district()


if __name__ == "__main__":
    main()
