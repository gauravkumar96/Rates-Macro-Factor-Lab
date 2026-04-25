# India Interest Rate Analysis

Analyzing the relationship between India's **Repo Rate**, **91-Day T-Bill Rate**, and **10-Year Government Bond Yield** using monthly data.

## What This Does :

- Converts RBI repo rate changes into a monthly time series
- Merges all three rate series on a common date index
- Computes spreads, correlations, and summary statistics across monetary policy regimes
- Generates an HTML presentation with charts and inferences

## Data Sources

| Series | Source |
|--------|--------|
| 91-Day T-Bill Rate | [FRED - INDIR3TIB01STM](https://fred.stlouisfed.org/series/INDIR3TIB01STM) |
| 10-Year Bond Yield | [FRED - INDIRLTLT01STM](https://fred.stlouisfed.org/series/INDIRLTLT01STM) |
| Repo Rate | [Shriram Finance - Historical Repo Rate Trends](https://www.shriramfinance.in/articles/deposits/2025/detailed-historical-repo-rate-trends-in-india) |

## Files

- `process_data.py` — Cleans, merges, and analyzes the data
- `generate_presentation.py` — Builds the HTML presentation
- `91days.csv` / `10years.csv` / `repo.xlsx` — Raw data
- `merged_data.csv` / `repo.csv` — Processed output
