import re
from pathlib import Path

import pandas as pd


# === Paths and output folders ===
DATA_DIR = Path("data")
RAW_DIR = DATA_DIR / "raw"
CLEAN_DIR = DATA_DIR / "cleaned"


def ensure_dirs() -> None:
    # Ensure output folder exists.
    CLEAN_DIR.mkdir(parents=True, exist_ok=True)


def normalize_key(text: str) -> str:
    # Normalize strings for robust state matching.
    if text is None:
        return ""
    key = str(text).strip().upper()
    key = key.replace("&", "AND")
    key = re.sub(r"[.\-]", " ", key)
    key = re.sub(r"\s+", " ", key)
    return key


# === State name standardization ===
STATE_MAP = {
    "AP": "Andhra Pradesh",
    "AR": "Arunachal Pradesh",
    "AS": "Assam",
    "BR": "Bihar",
    "CG": "Chhattisgarh",
    "CHHATTISGARH": "Chhattisgarh",
    "GA": "Goa",
    "GJ": "Gujarat",
    "HR": "Haryana",
    "HP": "Himachal Pradesh",
    "JH": "Jharkhand",
    "KA": "Karnataka",
    "KL": "Kerala",
    "MP": "Madhya Pradesh",
    "MH": "Maharashtra",
    "MN": "Manipur",
    "ML": "Meghalaya",
    "MZ": "Mizoram",
    "NL": "Nagaland",
    "OD": "Odisha",
    "OR": "Odisha",
    "ORISSA": "Odisha",
    "PB": "Punjab",
    "RJ": "Rajasthan",
    "SK": "Sikkim",
    "TN": "Tamil Nadu",
    "TS": "Telangana",
    "TR": "Tripura",
    "UK": "Uttarakhand",
    "UT": "Uttarakhand",
    "UP": "Uttar Pradesh",
    "WB": "West Bengal",
    "DL": "Delhi",
    "NCT OF DELHI": "Delhi",
    "DELHI": "Delhi",
    "PONDICHERRY": "Puducherry",
    "PUDUCHERRY": "Puducherry",
    "J AND K": "Jammu and Kashmir",
    "JAMMU AND KASHMIR": "Jammu and Kashmir",
    "JAMMU KASHMIR": "Jammu and Kashmir",
    "JAMMU KASHMIR AND LADAKH": "Jammu and Kashmir",
    "LADAKH": "Jammu and Kashmir",
    "ANDAMAN AND NICOBAR ISLANDS": "Andaman and Nicobar Islands",
    "DADRA AND NAGAR HAVELI": "Dadra and Nagar Haveli and Daman and Diu",
    "DAMAN AND DIU": "Dadra and Nagar Haveli and Daman and Diu",
}


def standardize_state(value: object) -> object:
    # Convert state abbreviations/aliases to full official names.
    if pd.isna(value):
        return value
    original = str(value).strip()
    key = normalize_key(original)
    if key in STATE_MAP:
        return STATE_MAP[key]
    return original


def year_from_string(value: object) -> object:
    # Extract YYYY from strings like "2019-20".
    if pd.isna(value):
        return value
    text = str(value).strip()
    match = re.search(r"(\d{4})", text)
    if not match:
        return value
    return int(match.group(1))


def apply_missing_policy(df: pd.DataFrame) -> pd.DataFrame:
    # Apply missing-value rules: drop columns/rows, then fill median/mode.
    if df.empty:
        return df

    col_missing = df.isna().mean()
    drop_cols = col_missing[col_missing > 0.40].index.tolist()
    df = df.drop(columns=drop_cols)

    row_missing = df.isna().mean(axis=1)
    df = df.loc[row_missing <= 0.50].copy()

    for col in df.columns:
        if pd.api.types.is_numeric_dtype(df[col]):
            median_val = df[col].median()
            if not pd.isna(median_val):
                df[col] = df[col].fillna(median_val)
        else:
            modes = df[col].mode(dropna=True)
            if not modes.empty:
                df[col] = df[col].fillna(modes.iloc[0])

    return df


def parse_aishe(file_path: Path, label_col: str) -> pd.DataFrame:
    # Dataset: AISHE tables (data/raw/aishe_enrollment.csv.csv or data/raw/aishe_university_type.csv.csv).
    # Action: rebuild multi-row headers and convert to long format.
    raw = pd.read_csv(file_path, header=None)

    header_levels = raw.iloc[1].copy()
    header_genders = raw.iloc[2].copy()

    header_levels = header_levels.ffill()
    header_genders = header_genders.ffill()

    columns = []
    for idx in range(raw.shape[1]):
        if idx == 0:
            columns.append("SlNo")
        elif idx == 1:
            columns.append(label_col)
        else:
            level = str(header_levels.iloc[idx]).strip()
            gender = str(header_genders.iloc[idx]).strip()
            if gender.lower() == "nan":
                columns.append(level)
            else:
                columns.append(f"{level} {gender}".strip())

    data = raw.iloc[4:].copy()
    data.columns = columns
    data = data.dropna(how="all")

    current_label = None
    states = []
    years = []
    year_pattern = re.compile(r"^\d{4}[-/]\d{2}$")

    for _, row in data.iterrows():
        label = row[label_col]
        label_str = "" if pd.isna(label) else str(label).strip()
        if year_pattern.match(label_str):
            states.append(current_label)
            years.append(year_from_string(label_str))
        elif label_str:
            current_label = label_str
            states.append(None)
            years.append(None)
        else:
            states.append(None)
            years.append(None)

    data[label_col] = states
    data["Year"] = years
    data = data.dropna(subset=["Year"])

    metric_cols = [c for c in data.columns if c not in {"SlNo", label_col, "Year"}]
    long_df = data.melt(
        id_vars=[label_col, "Year"],
        value_vars=metric_cols,
        var_name="Metric",
        value_name="Value",
    )
    long_df = long_df.dropna(subset=["Value"]).reset_index(drop=True)
    long_df[label_col] = long_df[label_col].apply(standardize_state)
    long_df["Year"] = long_df["Year"].astype(int)
    long_df["Value"] = pd.to_numeric(long_df["Value"], errors="coerce")
    long_df = long_df.dropna(subset=["Value"])

    return long_df


def clean_unemployment() -> pd.DataFrame:
    # Dataset: Unemployment (data/raw/unemployment_india.csv.csv).
    # Action: clean columns, convert monthly to yearly, standardize states.
    df = pd.read_csv(RAW_DIR / "unemployment_india.csv.csv")
    df.columns = [c.strip() for c in df.columns]

    df["Date"] = pd.to_datetime(df["Date"], errors="coerce", dayfirst=True)
    df["Year"] = df["Date"].dt.year
    df = df.drop(columns=["Date", "Frequency"])

    df["Region"] = df["Region"].apply(standardize_state)

    group_cols = ["Region", "Area", "Year"]
    numeric_cols = [
        c for c in df.select_dtypes(include=["number"]).columns.tolist() if c not in group_cols
    ]
    yearly = df.groupby(group_cols, dropna=False)[numeric_cols].mean().reset_index()

    yearly = apply_missing_policy(yearly)
    yearly = yearly.rename(
        columns={
            "Region": "State",
            "Estimated Unemployment Rate (%)": "unemployment_rate",
            "Estimated Employed": "estimated_employed",
            "Estimated Labour Participation Rate (%)": "labour_participation_rate",
        }
    )
    return yearly


def clean_economic_factors() -> pd.DataFrame:
    # Dataset: Economic indicators (data/raw/economic_factors.csv.csv).
    # Action: rename columns and standardize year.
    df = pd.read_csv(RAW_DIR / "economic_factors.csv.csv")
    df = df.rename(
        columns={
            "Years": "Year",
            "Unemployment rate": "unemployment_rate",
            "Per capita": "per_capita",
            "Poverty Rate": "poverty_rate",
            "Literacy Rate": "literacy_rate",
            "Labour Force": "labour_force",
            "Youth UR": "youth_unemployment_rate",
        }
    )
    df["Year"] = df["Year"].apply(year_from_string).astype(int)
    df = apply_missing_policy(df)
    return df


def clean_job_market() -> tuple[pd.DataFrame, pd.DataFrame]:
    # Dataset: Job market postings (data/raw/job_market.csv).
    # Action: derive sector + average salary, then aggregate by sector.
    df = pd.read_csv(RAW_DIR / "job_market.csv")

    df["sector"] = df["tagsAndSkills"].fillna("").apply(
        lambda x: x.split(",")[0].strip() if x.strip() else "Unknown"
    )

    df["average_salary"] = df[["minimumSalary", "maximumSalary"]].mean(axis=1)
    df["average_salary"] = df["average_salary"].fillna(df["minimumSalary"])
    df["average_salary"] = df["average_salary"].fillna(df["maximumSalary"])

    df = apply_missing_policy(df)

    sector_summary = (
        df.groupby("sector", dropna=False)
        .agg(job_count=("jobId", "count"), avg_salary=("average_salary", "median"))
        .reset_index()
    )

    return df, sector_summary


def clean_cperv1() -> tuple[pd.DataFrame, pd.DataFrame]:
    # Dataset: CPERV1 person file (data/raw/cperv1.csv).
    # Action: derive education, employment, skill, and informal rates.
    use_cols = [
        "State_UT_Code",
        "District_Code",
        "Principal_Status_Code",
        "Principal_Industry_Code",
        "Years_Formal_Education",
        "General_Education_Level",
        "Vocational_Training",
        "Training_Completed_365_Days",
        "Principal_Job_Contract_Type",
        "Principal_Social_Security",
    ]
    df = pd.read_csv(RAW_DIR / "cperv1.csv", usecols=use_cols)

    # Employment classification (per user guidance): Employed 11–51, Unemployed 81–82.
    df["employed_flag"] = df["Principal_Status_Code"].between(11, 51, inclusive="both")
    df["unemployed_flag"] = df["Principal_Status_Code"].isin([81, 82])

    # Skill training: treat Vocational_Training != 9 as trained.
    # Also include Training_Completed_365_Days == 1.
    df["skill_flag"] = (
        (df["Vocational_Training"].notna() & (df["Vocational_Training"] != 9))
        | (df["Training_Completed_365_Days"] == 1)
    )

    # Informal work proxy: no contract or no social security.
    # Contract types 1–2 indicate no/short contract.
    # Social security code 2 indicates not eligible.
    df["informal_flag"] = df["Principal_Job_Contract_Type"].isin([1, 2]) | (
        df["Principal_Social_Security"] == 2
    )

    # Map industry code to broad NIC section using first two digits (NIC 2008 style).
    def nic_section(code: object) -> str:
        if pd.isna(code):
            return "Unknown"
        try:
            val = int(str(code).split(".")[0])
        except ValueError:
            return "Unknown"
        two = val // 1000  # e.g., 84119 -> 84
        if 1 <= two <= 3:
            return "A_Agriculture"
        if 5 <= two <= 9:
            return "B_Mining"
        if 10 <= two <= 33:
            return "C_Manufacturing"
        if two == 35:
            return "D_Electricity"
        if 36 <= two <= 39:
            return "E_Water_Waste"
        if 41 <= two <= 43:
            return "F_Construction"
        if 45 <= two <= 47:
            return "G_Trade"
        if 49 <= two <= 53:
            return "H_Transport"
        if 55 <= two <= 56:
            return "I_Accommodation"
        if 58 <= two <= 63:
            return "J_Information"
        if 64 <= two <= 66:
            return "K_Finance"
        if two == 68:
            return "L_RealEstate"
        if 69 <= two <= 75:
            return "M_Professional"
        if 77 <= two <= 82:
            return "N_Admin"
        if two == 84:
            return "O_Public"
        if two == 85:
            return "P_Education"
        if 86 <= two <= 88:
            return "Q_Health"
        if 90 <= two <= 93:
            return "R_Arts"
        if 94 <= two <= 96:
            return "S_OtherServices"
        if 97 <= two <= 98:
            return "T_Households"
        if two == 99:
            return "U_Extraterritorial"
        return "Unknown"

    df["industry_section"] = df["Principal_Industry_Code"].apply(nic_section)

    agg = (
        df.groupby("State_UT_Code", dropna=False)
        .agg(
            education_index=("Years_Formal_Education", "mean"),
            employment_rate=("employed_flag", "mean"),
            unemployment_rate=("unemployed_flag", "mean"),
            skill_rate=("skill_flag", "mean"),
            informal_rate=("informal_flag", "mean"),
            count=("Principal_Status_Code", "size"),
        )
        .reset_index()
    )

    # District-level aggregation with sector shares.
    district = (
        df.groupby(["State_UT_Code", "District_Code"], dropna=False)
        .agg(
            education_index=("Years_Formal_Education", "mean"),
            employment_rate=("employed_flag", "mean"),
            unemployment_rate=("unemployed_flag", "mean"),
            skill_rate=("skill_flag", "mean"),
            informal_rate=("informal_flag", "mean"),
            count=("Principal_Status_Code", "size"),
        )
        .reset_index()
    )
    sector_counts = (
        df.groupby(["State_UT_Code", "District_Code", "industry_section"])
        .size()
        .reset_index(name="sector_count")
    )
    sector_pivot = sector_counts.pivot_table(
        index=["State_UT_Code", "District_Code"],
        columns="industry_section",
        values="sector_count",
        fill_value=0,
    ).reset_index()
    sector_totals = sector_counts.groupby(["State_UT_Code", "District_Code"])["sector_count"].sum().reset_index(name="total")
    sector_pivot = sector_pivot.merge(sector_totals, on=["State_UT_Code", "District_Code"], how="left")
    for col in sector_pivot.columns:
        if col not in {"State_UT_Code", "District_Code", "total"}:
            sector_pivot[col] = sector_pivot[col] / sector_pivot["total"]
            sector_pivot = sector_pivot.rename(columns={col: f"sector_share_{col}"})
    sector_pivot = sector_pivot.drop(columns=["total"])

    district = district.merge(sector_pivot, on=["State_UT_Code", "District_Code"], how="left")

    # Map State_UT_Code to State names using uploaded mapping file.
    state_map_path = RAW_DIR / "4. Indian_States_and_UTs_Code  Name.xlsx"
    if state_map_path.exists():
        state_map = pd.read_excel(state_map_path)
        state_map = state_map.rename(
            columns={"State Code": "State_UT_Code", "State/UT Name": "State"}
        )
        agg = agg.merge(state_map, on="State_UT_Code", how="left")
        district = district.merge(state_map, on="State_UT_Code", how="left")
    else:
        agg["State"] = agg["State_UT_Code"].astype(str)

    national = pd.DataFrame(
        {
            "education_index": [df["Years_Formal_Education"].mean()],
            "employment_rate": [df["employed_flag"].mean()],
            "unemployment_rate": [df["unemployed_flag"].mean()],
            "skill_rate": [df["skill_flag"].mean()],
            "informal_rate": [df["informal_flag"].mean()],
            "count": [len(df)],
        }
    )

    assumptions = (
        "CPERV1 assumptions used:\n"
        "- Employed: Principal_Status_Code between 11 and 51 (inclusive).\n"
        "- Unemployed: Principal_Status_Code in {81, 82}.\n"
        "- Training received: Vocational_Training != 9 OR Training_Completed_365_Days == 1.\n"
        "- Informal proxy: Principal_Job_Contract_Type in {1,2} OR Principal_Social_Security == 2.\n"
        "- Industry sections approximated from NIC 2008 2-digit ranges (sector shares by district).\n"
        "State code mapping uses the uploaded State Code -> State Name file.\n"
        "These follow the decoded CPERV1 variable definitions provided.\n"
    )

    (CLEAN_DIR / "cperv1_assumptions.txt").write_text(assumptions, encoding="utf-8")

    return agg, national, district


def build_base_and_merge(
    unemployment: pd.DataFrame,
    economic: pd.DataFrame,
    aishe_enrollment: pd.DataFrame,
    job_market_sector: pd.DataFrame,
) -> pd.DataFrame:
    # Controlled merge: unemployment base + economic + AISHE national + job demand.
    # Base: data/cleaned/unemployment_india_clean.csv
    # Adds: data/cleaned/economic_factors_clean.csv + AISHE national trend + job_market_sector_summary.csv
    base = unemployment.copy()

    merged = base.merge(economic, on="Year", how="left")

    # AISHE enrollment has no state; treat as national trend and merge by Year.
    aishe_national = (
        aishe_enrollment.groupby(["Year", "Metric"], dropna=False)["Value"]
        .sum()
        .reset_index()
    )
    aishe_wide = aishe_national.pivot(index="Year", columns="Metric", values="Value")
    aishe_wide = aishe_wide.add_prefix("edu_").reset_index()

    merged = merged.merge(aishe_wide, on="Year", how="left")

    # Job market data has no year info; apply overall sector summary to all years.
    unique_years = sorted(merged["Year"].dropna().unique())
    overall_job = pd.DataFrame(
        {
            "Year": unique_years,
            "job_count_total": job_market_sector["job_count"].sum(),
            "job_avg_salary_overall": job_market_sector["avg_salary"].median(),
        }
    )

    merged = merged.merge(overall_job, on="Year", how="left")

    return merged


def main() -> None:
    # Orchestrate all cleaning steps and write outputs.
    ensure_dirs()

    # === AISHE (Education) ===
    # Source: data/raw/aishe_enrollment.csv.csv and data/raw/aishe_university_type.csv.csv
    aishe_enrollment = parse_aishe(
        RAW_DIR / "aishe_enrollment.csv.csv", label_col="UniversityType"
    )
    aishe_university = parse_aishe(
        RAW_DIR / "aishe_university_type.csv.csv", label_col="State"
    )

    # === Unemployment (Target Base) ===
    # Source: data/raw/unemployment_india.csv.csv
    unemployment = clean_unemployment()

    # === Economic Indicators ===
    # Source: data/raw/economic_factors.csv.csv
    economic = clean_economic_factors()

    # === Job Market ===
    # Source: data/raw/job_market.csv
    job_market, job_market_sector = clean_job_market()

    # === CPERV1 Person File ===
    # Source: data/raw/cperv1.csv
    cperv1_by_state, cperv1_national, cperv1_by_district = clean_cperv1()

    aishe_enrollment.to_csv(CLEAN_DIR / "aishe_enrollment_clean.csv", index=False)
    aishe_university.to_csv(CLEAN_DIR / "aishe_university_type_clean.csv", index=False)
    unemployment.to_csv(CLEAN_DIR / "unemployment_india_clean.csv", index=False)
    economic.to_csv(CLEAN_DIR / "economic_factors_clean.csv", index=False)
    job_market.to_csv(CLEAN_DIR / "job_market_clean.csv", index=False)
    job_market_sector.to_csv(CLEAN_DIR / "job_market_sector_summary.csv", index=False)
    cperv1_by_state.to_csv(CLEAN_DIR / "cperv1_features_by_state.csv", index=False)
    cperv1_national.to_csv(CLEAN_DIR / "cperv1_features_national.csv", index=False)
    cperv1_by_district.to_csv(CLEAN_DIR / "cperv1_features_by_district.csv", index=False)

    # === Final Controlled Merge ===
    merged = build_base_and_merge(
        unemployment=unemployment,
        economic=economic,
        aishe_enrollment=aishe_enrollment,
        job_market_sector=job_market_sector,
    )
    merged.to_csv(CLEAN_DIR / "final_merged.csv", index=False)

    print("Cleaned files saved to:", CLEAN_DIR)
    print("Rows:")
    print("aishe_enrollment_clean:", len(aishe_enrollment))
    print("aishe_university_type_clean:", len(aishe_university))
    print("unemployment_india_clean:", len(unemployment))
    print("economic_factors_clean:", len(economic))
    print("job_market_clean:", len(job_market))
    print("job_market_sector_summary:", len(job_market_sector))
    print("final_merged:", len(merged))


if __name__ == "__main__":
    main()
