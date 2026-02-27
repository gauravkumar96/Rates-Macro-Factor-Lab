# NSS Yield Curve Visualization

https://yieldcurveestimation1nss.streamlit.app/

Interactive dashboard and notebook for exploring the **Nelson-Siegel-Svensson (NSS)** Zero Coupon Yield Curve (ZCYC) for Indian Government Securities.

## Project Structure

```
├── dashboard/                  # Streamlit dashboard app
│   ├── app.py                  # Main dashboard application
│   ├── requirements.txt        # Python dependencies
│   ├── 0_rates_india.xlsx      # Official ZCYC rates (FBIL/CCIL)
│   ├── ZCYC_param.xlsx         # NSS model parameters (β₀–β₃, τ₁, τ₂)
│   └── .streamlit/config.toml  # Theme configuration
├── NSS_Model.ipynb             # Jupyter notebook — theory + functions
└── README.md
```

## Quick Start

```bash
cd dashboard
pip install -r requirements.txt
streamlit run app.py
```

## Features

### Dashboard (`dashboard/app.py`)
- **Static yield curve** plotted from official FBIL/CCIL data
- **Interactive sliders** for all 6 NSS parameters (β₀, β₁, β₂, β₃, τ₁, τ₂)
- **Dynamic curve overlay** — see real-time impact of parameter changes
- **CSV download** — export both original and adjusted curve data
- **Model explanation page** — detailed NSS theory with academic references

### Notebook (`NSS_Model.ipynb`)
- NSS model theory and formulas in markdown
- Core functions: `nss_rate()`, `load_rates()`, `load_params()`, `plot_yield_curves()`, `export_curves()`
- Interactive Plotly chart output
- Academic references

## The NSS Model

The Nelson-Siegel-Svensson model fits the term structure with 6 parameters:

$$r(m) = \beta_0 + \beta_1 \left(\frac{1 - e^{-m/\tau_1}}{m/\tau_1}\right) + \beta_2 \left(\frac{1 - e^{-m/\tau_1}}{m/\tau_1} - e^{-m/\tau_1}\right) + \beta_3 \left(\frac{1 - e^{-m/\tau_2}}{m/\tau_2} - e^{-m/\tau_2}\right)$$

| Parameter | Role |
|-----------|------|
| β₀ | Long-run interest rate level |
| β₁ | Short-term slope component |
| β₂ | First curvature (hump at τ₁) |
| β₃ | Second curvature (hump at τ₂) |
| τ₁, τ₂ | Decay factors controlling hump locations |

## Data Sources

- **FBIL** — [fbil.org.in](https://www.fbil.org.in/)
- **CCIL** — [ccilindia.com](https://www.ccilindia.com/web/ccil/zero-coupon-yield-curve)

## References

1. Nelson & Siegel (1987) — *Parsimonious Modeling of Yield Curves*, Journal of Business
2. Svensson (1994) — *Estimating Forward Interest Rates*, NBER WP 4871
3. BIS Papers No. 25 (2005) — *Zero-Coupon Yield Curves: Technical Documentation*

