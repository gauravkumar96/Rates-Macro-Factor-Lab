# Rates Macro Factor Lab

A systematic framework to **model, decompose, and stress-test interest rate curves** as a macro risk factor across asset classes — focused on Indian fixed-income markets.

Each numbered sub-project is self-contained with its own data, code, and write-up.

## Sub-Projects

### [01 · NSS Yield Curve Estimation](01_YieldCurveEstimation_NSS_model/)
Six-parameter Nelson-Siegel-Svensson fit of the Indian G-Sec ZCYC, with a Streamlit dashboard for interactive parameter exploration.
**Live:** https://yieldcurveestimation1nss.streamlit.app/ · **Data:** FBIL, CCIL

### [02 · Repo vs 91-Day T-Bills vs 10Y Bond](02_Repo_91dayTBills_vs_10YrBond/)
Monetary policy transmission study — aligns RBI repo, 91D T-Bill, and 10Y G-Sec on a monthly index, computes spreads and regime-conditional stats. Outputs an HTML deck and PDF report.
**Data:** FRED, RBI

### [03 · Carry Trade Analysis](03_Carry_Trade_Analysis/)
Cross-country carry trade study (USDINR, USDJPY) — decomposes P&L into rate differential vs. FX depreciation, with cumulative returns, risk/return, and inflation-FX charts. Outputs Word reports and a comparison PDF.
**Data:** USDINR & USDJPY annual/quarterly series

## Roadmap

More sub-projects planned: PCA factor decomposition (level/slope/curvature), cross-asset rate sensitivity, stress-testing, term-premium decomposition.

## Tech Stack

Python (`pandas`, `numpy`, `scipy`, `plotly`, `streamlit`), Jupyter, Streamlit. Data from FBIL, CCIL, and FRED.
