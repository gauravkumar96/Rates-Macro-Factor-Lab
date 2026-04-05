"""
Process repo rate, 91-day T-bill, and 10-year bond data for India.
1. Convert repo rate changes to month-end data -> repo.csv
2. Join all three datasets on dates -> merged_data.csv
3. Generate an HTML presentation with inferences
"""

import pandas as pd
import numpy as np
from datetime import datetime

# ─────────────────────────────────────────────────────
# 1. Read and process Repo Rate data
# ─────────────────────────────────────────────────────
repo_raw = pd.read_excel("repo.xlsx")
print("=== Repo Raw Data ===")
print(repo_raw.head())
print(f"Shape: {repo_raw.shape}")

# The repo rate is in decimal form (e.g., 0.05 = 5%). Convert to percentage.
repo_raw["Repo Rate (%)"] = repo_raw["Repo Rate"] * 100

# Ensure dates are datetime
repo_raw["Effective Date"] = pd.to_datetime(repo_raw["Effective Date"])

# Sort by date
repo_raw = repo_raw.sort_values("Effective Date").reset_index(drop=True)

# Create month-end repo rate series
# For each month, the repo rate at month-end is the last effective rate <= that month-end
# First, determine the date range
start_date = repo_raw["Effective Date"].min()
end_date = pd.Timestamp("2026-02-28")  # Match the other datasets

# Generate month-end dates
month_ends = pd.date_range(start=start_date, end=end_date, freq="ME")

# For each month-end, find the applicable repo rate (last change on or before that date)
repo_monthly = []
for me_date in month_ends:
    applicable = repo_raw[repo_raw["Effective Date"] <= me_date]
    if len(applicable) > 0:
        rate = applicable.iloc[-1]["Repo Rate (%)"]
        repo_monthly.append({"observation_date": me_date.strftime("%Y-%m-%d"), "repo_rate": round(rate, 4)})

repo_df = pd.DataFrame(repo_monthly)

# Convert observation_date to first-of-month format to match the other CSVs
# The other CSVs use YYYY-MM-01 format
repo_df["observation_date"] = pd.to_datetime(repo_df["observation_date"])
# Create a column with first-of-month for joining
repo_df["month_key"] = repo_df["observation_date"].dt.to_period("M")

print("\n=== Repo Monthly (first 10) ===")
print(repo_df.head(10))
print(f"Shape: {repo_df.shape}")

# Save repo.csv with month-end dates
repo_csv = repo_df[["observation_date", "repo_rate"]].copy()
repo_csv.to_csv("repo.csv", index=False)
print(f"\nSaved repo.csv with {len(repo_csv)} rows")

# ─────────────────────────────────────────────────────
# 2. Read and join all three datasets
# ─────────────────────────────────────────────────────
# Read 91-day and 10-year data
tbill_91d = pd.read_csv("91days.csv")
bond_10y = pd.read_csv("10years.csv")

# Rename columns for clarity
tbill_91d.columns = ["observation_date", "tbill_91d_rate"]
bond_10y.columns = ["observation_date", "bond_10y_rate"]

# Convert dates
tbill_91d["observation_date"] = pd.to_datetime(tbill_91d["observation_date"])
bond_10y["observation_date"] = pd.to_datetime(bond_10y["observation_date"])

# Create month keys for joining
tbill_91d["month_key"] = tbill_91d["observation_date"].dt.to_period("M")
bond_10y["month_key"] = bond_10y["observation_date"].dt.to_period("M")

print(f"\n91-day T-bill: {tbill_91d.shape}, range: {tbill_91d['observation_date'].min()} to {tbill_91d['observation_date'].max()}")
print(f"10-year Bond: {bond_10y.shape}, range: {bond_10y['observation_date'].min()} to {bond_10y['observation_date'].max()}")
print(f"Repo Rate: {repo_df.shape}, range: {repo_df['observation_date'].min()} to {repo_df['observation_date'].max()}")

# Merge all three on month_key
merged = repo_df[["month_key", "observation_date", "repo_rate"]].merge(
    tbill_91d[["month_key", "tbill_91d_rate"]], on="month_key", how="outer"
).merge(
    bond_10y[["month_key", "bond_10y_rate"]], on="month_key", how="outer"
)

# Sort by month_key
merged = merged.sort_values("month_key").reset_index(drop=True)

# Fill observation_date where missing from other sources
for idx, row in merged.iterrows():
    if pd.isna(row["observation_date"]):
        # Try to reconstruct from month_key
        merged.at[idx, "observation_date"] = row["month_key"].to_timestamp()

merged["observation_date"] = pd.to_datetime(merged["observation_date"])
merged = merged.sort_values("observation_date").reset_index(drop=True)

# Keep only the relevant columns
merged_final = merged[["observation_date", "repo_rate", "tbill_91d_rate", "bond_10y_rate"]].copy()
merged_final.to_csv("merged_data.csv", index=False)
print(f"\nSaved merged_data.csv with {len(merged_final)} rows")
print("\n=== Merged Data (first 20) ===")
print(merged_final.head(20).to_string())
print("\n=== Merged Data (last 10) ===")
print(merged_final.tail(10).to_string())

# ─────────────────────────────────────────────────────
# 3. Compute statistics for the presentation
# ─────────────────────────────────────────────────────
# Only use rows where all three rates are available
complete = merged_final.dropna().copy()
print(f"\nComplete data rows (all three rates): {len(complete)}")

# Compute spreads
complete["spread_10y_repo"] = complete["bond_10y_rate"] - complete["repo_rate"]
complete["spread_91d_repo"] = complete["tbill_91d_rate"] - complete["repo_rate"]
complete["spread_10y_91d"] = complete["bond_10y_rate"] - complete["tbill_91d_rate"]
complete["term_spread"] = complete["bond_10y_rate"] - complete["tbill_91d_rate"]

# Stats
print("\n=== Summary Statistics ===")
for col in ["repo_rate", "tbill_91d_rate", "bond_10y_rate"]:
    print(f"\n{col}:")
    print(f"  Mean: {complete[col].mean():.2f}%")
    print(f"  Std:  {complete[col].std():.2f}%")
    print(f"  Min:  {complete[col].min():.2f}% ({complete.loc[complete[col].idxmin(), 'observation_date'].strftime('%Y-%m')})")
    print(f"  Max:  {complete[col].max():.2f}% ({complete.loc[complete[col].idxmax(), 'observation_date'].strftime('%Y-%m')})")

# Correlation
corr = complete[["repo_rate", "tbill_91d_rate", "bond_10y_rate"]].corr()
print("\n=== Correlation Matrix ===")
print(corr.to_string())

# Spread statistics  
print("\n=== Spread Statistics ===")
print(f"10Y - Repo: Mean={complete['spread_10y_repo'].mean():.2f}%, Std={complete['spread_10y_repo'].std():.2f}%")
print(f"91D - Repo: Mean={complete['spread_91d_repo'].mean():.2f}%, Std={complete['spread_91d_repo'].std():.2f}%")
print(f"10Y - 91D (Term Spread): Mean={complete['term_spread'].mean():.2f}%, Std={complete['term_spread'].std():.2f}%")

# Identify monetary policy regimes
print("\n=== Monetary Policy Regimes ===")
for start, end, name in [
    ("2012-01", "2014-01", "Tightening (2012-2014)"),
    ("2015-01", "2016-12", "Easing (2015-2016)"),
    ("2017-01", "2019-12", "Mixed (2017-2019)"),
    ("2020-01", "2021-12", "COVID Easing (2020-2021)"),
    ("2022-01", "2023-12", "Post-COVID Tightening (2022-2023)"),
    ("2024-01", "2026-02", "Easing Cycle (2024-2026)")
]:
    mask = (complete["observation_date"] >= start) & (complete["observation_date"] <= end)
    subset = complete[mask]
    if len(subset) > 0:
        print(f"\n  {name}:")
        print(f"    Repo: {subset['repo_rate'].iloc[0]:.2f}% -> {subset['repo_rate'].iloc[-1]:.2f}%")
        print(f"    91D T-bill avg: {subset['tbill_91d_rate'].mean():.2f}%")
        print(f"    10Y Bond avg:   {subset['bond_10y_rate'].mean():.2f}%")
        print(f"    Avg Term Spread: {subset['term_spread'].mean():.2f}%")

print("\n✅ Processing complete!")
