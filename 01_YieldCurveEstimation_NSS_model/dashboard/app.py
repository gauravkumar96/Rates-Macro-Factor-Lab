"""
NSS (Nelson-Siegel-Svensson) Yield Curve Dashboard
====================================================
Interactive Streamlit dashboard for visualizing and exploring
the Zero Coupon Yield Curve (ZCYC) for Indian Government Securities.

Data source: FBIL / CCIL
Model: Nelson-Siegel-Svensson (4-factor)
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from pathlib import Path
import io

# ──────────────────────────────────────────────
# Page configuration
# ──────────────────────────────────────────────
st.set_page_config(
    page_title="NSS Yield Curve Dashboard",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ──────────────────────────────────────────────
# Custom CSS
# ──────────────────────────────────────────────
st.markdown("""
<style>
    /* Main header styling */
    .main-header {
        background: linear-gradient(135deg, #0a0f1a 0%, #111827 50%, #0d1117 100%);
        padding: 1.8rem 2rem;
        border-radius: 16px;
        margin-bottom: 1.5rem;
        border: 1px solid rgba(201, 168, 76, 0.2);
        box-shadow: 0 8px 32px rgba(0, 0, 0, 0.4);
    }
    .main-header h1 {
        background: linear-gradient(90deg, #c9a84c, #e8d48b, #c9a84c);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        font-size: 2rem;
        font-weight: 800;
        margin: 0;
        letter-spacing: -0.5px;
    }
    .main-header p {
        color: #9ca3af;
        font-size: 0.95rem;
        margin: 0.4rem 0 0 0;
    }

    /* Card styling */
    .metric-card {
        background: linear-gradient(145deg, #111827, #0d1117);
        padding: 1.2rem 1.4rem;
        border-radius: 12px;
        border: 1px solid rgba(201, 168, 76, 0.1);
        margin-bottom: 0.8rem;
        transition: transform 0.2s ease, box-shadow 0.2s ease;
    }
    .metric-card:hover {
        transform: translateY(-2px);
        box-shadow: 0 6px 20px rgba(201, 168, 76, 0.12);
    }
    .metric-card .label {
        color: #6b7280;
        font-size: 0.75rem;
        text-transform: uppercase;
        letter-spacing: 1.2px;
        font-weight: 600;
    }
    .metric-card .value {
        color: #c9a84c;
        font-size: 1.5rem;
        font-weight: 700;
        margin-top: 0.2rem;
    }

    /* Sidebar styling */
    section[data-testid="stSidebar"] {
        background: linear-gradient(180deg, #0a0f1a 0%, #111827 100%);
    }
    section[data-testid="stSidebar"] .stSlider > div > div > div {
        background: #c9a84c;
    }

    /* Formula box */
    .formula-box {
        background: linear-gradient(145deg, #0d1117, #0a0f1a);
        border: 1px solid rgba(201, 168, 76, 0.2);
        border-radius: 12px;
        padding: 1.5rem;
        margin: 1rem 0;
    }

    /* Download section */
    .download-section {
        background: linear-gradient(145deg, #111827, #0d1117);
        padding: 1.2rem;
        border-radius: 12px;
        border: 1px solid rgba(201, 168, 76, 0.1);
        margin-top: 1rem;
    }

    /* Info page styling */
    .info-section {
        background: linear-gradient(145deg, #0d1117, #0a0f1a);
        padding: 1.8rem;
        border-radius: 14px;
        border: 1px solid rgba(201, 168, 76, 0.08);
        margin-bottom: 1.5rem;
    }
    .info-section h3 {
        color: #c9a84c;
        border-bottom: 2px solid rgba(201, 168, 76, 0.25);
        padding-bottom: 0.5rem;
        margin-bottom: 1rem;
    }

    /* Reference cards */
    .ref-card {
        background: linear-gradient(145deg, #111827, #0d1117);
        padding: 1rem 1.2rem;
        border-radius: 10px;
        border-left: 3px solid #c9a84c;
        margin-bottom: 0.8rem;
        transition: transform 0.2s ease;
    }
    .ref-card:hover {
        transform: translateX(4px);
    }
    .ref-card .ref-title {
        color: #e5e7eb;
        font-weight: 600;
        font-size: 0.95rem;
    }
    .ref-card .ref-detail {
        color: #9ca3af;
        font-size: 0.82rem;
        margin-top: 0.25rem;
    }

    /* Tabs */
    .stTabs [data-baseweb="tab-list"] {
        gap: 8px;
    }
    .stTabs [data-baseweb="tab"] {
        background-color: #111827;
        border-radius: 8px 8px 0 0;
        padding: 8px 20px;
        border: 1px solid rgba(201, 168, 76, 0.08);
    }
    .stTabs [data-baseweb="tab"][aria-selected="true"] {
        background-color: #0a0f1a;
        border-bottom: 2px solid #c9a84c;
    }
</style>
""", unsafe_allow_html=True)


# ──────────────────────────────────────────────
# Helper: NSS model
# ──────────────────────────────────────────────
def nss_rate(m: np.ndarray, beta0: float, beta1: float,
             beta2: float, beta3: float, tau1: float, tau2: float) -> np.ndarray:
    """Compute the Nelson-Siegel-Svensson yield for maturity array *m* (in years)."""
    # Avoid division by zero at m = 0
    m = np.where(m == 0, 1e-6, m)

    x1 = m / tau1
    x2 = m / tau2

    term1 = (1 - np.exp(-x1)) / x1
    term2 = term1 - np.exp(-x1)
    term3 = (1 - np.exp(-x2)) / x2 - np.exp(-x2)

    return beta0 + beta1 * term1 + beta2 * term2 + beta3 * term3


# ──────────────────────────────────────────────
# Data loading (cached)
# ──────────────────────────────────────────────
DATA_DIR = Path(__file__).parent


@st.cache_data
def load_rates():
    """Load the official ZCYC rates from 0_rates_india.xlsx."""
    df = pd.read_excel(DATA_DIR / "0_rates_india.xlsx")
    df.columns = ["Maturity", "Zero_Coupon_Rate"]
    return df


@st.cache_data
def load_params():
    """Load NSS parameters from ZCYC_param.xlsx."""
    df = pd.read_excel(DATA_DIR / "ZCYC_param.xlsx")
    # Normalise column names (special chars in β)
    cols = df.columns.tolist()
    rename_map = {}
    for c in cols:
        cl = c.lower().strip()
        if "0" in cl and "date" not in cl:
            rename_map[c] = "beta0"
        elif "1" in cl and "tau" not in cl:
            rename_map[c] = "beta1"
        elif "2" in cl and "tau" not in cl:
            rename_map[c] = "beta2"
        elif "3" in cl and "tau" not in cl:
            rename_map[c] = "beta3"
        elif "tau1" in cl or ("tau" in cl and "1" in cl):
            rename_map[c] = "tau1"
        elif "tau2" in cl or ("tau" in cl and "2" in cl):
            rename_map[c] = "tau2"
        elif "date" in cl:
            rename_map[c] = "date"
    df = df.rename(columns=rename_map)
    return df


# ──────────────────────────────────────────────
# Load data
# ──────────────────────────────────────────────
rates_df = load_rates()
params_df = load_params()

param_row = params_df.iloc[0]
default_params = {
    "beta0": float(param_row["beta0"]),
    "beta1": float(param_row["beta1"]),
    "beta2": float(param_row["beta2"]),
    "beta3": float(param_row["beta3"]),
    "tau1": float(param_row["tau1"]),
    "tau2": float(param_row["tau2"]),
}
curve_date = str(param_row.get("date", ""))

# ──────────────────────────────────────────────
# Navigation
# ──────────────────────────────────────────────
page = st.sidebar.radio(
    "📑 **Navigation**",
    ["📈  Yield Curve Explorer", "📖  Model Explanation"],
    index=0,
)

# ══════════════════════════════════════════════
# PAGE 1 — Yield Curve Explorer
# ══════════════════════════════════════════════
if page == "📈  Yield Curve Explorer":

    # Header
    st.markdown("""
    <div class="main-header">
        <h1>📈 NSS Yield Curve Dashboard</h1>
        <p>Interactive Nelson-Siegel-Svensson model explorer &nbsp;·&nbsp; Indian Government Securities ZCYC</p>
    </div>
    """, unsafe_allow_html=True)

    # ── Sidebar: parameter sliders ──
    st.sidebar.markdown("---")
    st.sidebar.markdown("### 🎛️ NSS Parameters")
    st.sidebar.caption("Adjust the sliders to reshape the yield curve")

    beta0 = st.sidebar.slider(
        "β₀  (long-run level)",
        min_value=0.0, max_value=20.0,
        value=default_params["beta0"], step=0.01,
        format="%.4f",
        help="Asymptotic long-term interest rate level — the rate the curve converges to at very long maturities.",
    )
    beta1 = st.sidebar.slider(
        "β₁  (short-term component)",
        min_value=-30.0, max_value=30.0,
        value=default_params["beta1"], step=0.01,
        format="%.4f",
        help="Controls the slope at the short end. Negative → upward-sloping; positive → downward-sloping.",
    )
    beta2 = st.sidebar.slider(
        "β₂  (medium-term hump 1)",
        min_value=-50.0, max_value=50.0,
        value=default_params["beta2"], step=0.01,
        format="%.4f",
        help="Creates a hump or trough at maturity ≈ τ₁. Controls the first curvature factor.",
    )
    beta3 = st.sidebar.slider(
        "β₃  (medium-term hump 2)",
        min_value=-50.0, max_value=50.0,
        value=default_params["beta3"], step=0.01,
        format="%.4f",
        help="Creates a second hump/trough at maturity ≈ τ₂. The Svensson extension factor.",
    )

    st.sidebar.markdown("---")
    st.sidebar.markdown("### ⏱️ Decay Parameters")

    tau1 = st.sidebar.slider(
        "τ₁  (first decay)",
        min_value=0.1, max_value=50.0,
        value=default_params["tau1"], step=0.01,
        format="%.4f",
        help="Decay speed for β₁ and β₂ components. Larger → slower decay, hump shifts to longer maturities.",
    )
    tau2 = st.sidebar.slider(
        "τ₂  (second decay)",
        min_value=0.1, max_value=50.0,
        value=default_params["tau2"], step=0.01,
        format="%.4f",
        help="Decay speed for β₃ component. Controls where the Svensson extension's hump peaks.",
    )

    # ── Reset button ──
    if st.sidebar.button("🔄 Reset to Original Parameters", use_container_width=True):
        st.rerun()

    # ── Compute dynamic curve ──
    maturities = rates_df["Maturity"].values
    dynamic_rates = nss_rate(maturities, beta0, beta1, beta2, beta3, tau1, tau2)

    # ── Metric cards row ──
    col1, col2, col3, col4 = st.columns(4)

    with col1:
        st.markdown(f"""
        <div class="metric-card">
            <div class="label">Curve Date</div>
            <div class="value" style="font-size:1.1rem">{curve_date}</div>
        </div>
        """, unsafe_allow_html=True)
    with col2:
        st.markdown(f"""
        <div class="metric-card">
            <div class="label">Short Rate (0.5Y)</div>
            <div class="value">{dynamic_rates[1]:.2f}%</div>
        </div>
        """, unsafe_allow_html=True)
    with col3:
        st.markdown(f"""
        <div class="metric-card">
            <div class="label">Mid Rate (10Y)</div>
            <div class="value">{dynamic_rates[np.argmin(np.abs(maturities - 10))]:.2f}%</div>
        </div>
        """, unsafe_allow_html=True)
    with col4:
        st.markdown(f"""
        <div class="metric-card">
            <div class="label">Long Rate (30Y)</div>
            <div class="value">{dynamic_rates[np.argmin(np.abs(maturities - 30))]:.2f}%</div>
        </div>
        """, unsafe_allow_html=True)

    # ── Build Plotly chart ──
    fig = go.Figure()

    # Static curve (original data)
    fig.add_trace(go.Scatter(
        x=maturities,
        y=rates_df["Zero_Coupon_Rate"].values,
        mode="lines",
        name="Original ZCYC (FBIL/CCIL)",
        line=dict(color="#c9a84c", width=3),
        hovertemplate="Maturity: %{x:.1f}Y<br>Rate: %{y:.4f}%<extra>Original</extra>",
    ))

    # Dynamic curve (user-adjusted)
    fig.add_trace(go.Scatter(
        x=maturities,
        y=dynamic_rates,
        mode="lines",
        name="User-Adjusted Curve",
        line=dict(color="#5b8def", width=2.5, dash="dash"),
        hovertemplate="Maturity: %{x:.1f}Y<br>Rate: %{y:.4f}%<extra>Adjusted</extra>",
    ))

    # Spread fill between curves
    fig.add_trace(go.Scatter(
        x=np.concatenate([maturities, maturities[::-1]]),
        y=np.concatenate([rates_df["Zero_Coupon_Rate"].values, dynamic_rates[::-1]]),
        fill="toself",
        fillcolor="rgba(91, 141, 239, 0.07)",
        line=dict(color="rgba(255,255,255,0)"),
        hoverinfo="skip",
        showlegend=False,
    ))

    fig.update_layout(
        template="plotly_dark",
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(10, 15, 26, 0.9)",
        title=dict(
            text="Zero Coupon Yield Curve — India Government Securities",
            font=dict(size=18, color="#e5e7eb"),
            x=0.5,
        ),
        xaxis=dict(
            title="Maturity (Years)",
            gridcolor="rgba(201, 168, 76, 0.08)",
            range=[0, 51],
            dtick=5,
            showspikes=True,
            spikecolor="#c9a84c",
            spikethickness=1,
        ),
        yaxis=dict(
            title="Zero Coupon Rate (%)",
            gridcolor="rgba(201, 168, 76, 0.08)",
            showspikes=True,
            spikecolor="#c9a84c",
            spikethickness=1,
        ),
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.02,
            xanchor="center",
            x=0.5,
            bgcolor="rgba(0,0,0,0.3)",
            bordercolor="rgba(255,255,255,0.1)",
            borderwidth=1,
            font=dict(size=12),
        ),
        hovermode="x unified",
        height=540,
        margin=dict(l=60, r=30, t=80, b=60),
    )

    st.plotly_chart(fig, use_container_width=True)

    # ── NSS Formula (collapsible) ──
    with st.expander("📐 NSS Formula Used", expanded=False):
        st.markdown('<div class="formula-box">', unsafe_allow_html=True)
        st.latex(
            r"r(m) = \beta_0 "
            r"+ \beta_1 \left(\frac{1 - e^{-m/\tau_1}}{m/\tau_1}\right) "
            r"+ \beta_2 \left(\frac{1 - e^{-m/\tau_1}}{m/\tau_1} - e^{-m/\tau_1}\right) "
            r"+ \beta_3 \left(\frac{1 - e^{-m/\tau_2}}{m/\tau_2} - e^{-m/\tau_2}\right)"
        )
        st.markdown("</div>", unsafe_allow_html=True)

        # Current parameters table
        st.markdown("**Current parameter values:**")
        param_table = pd.DataFrame({
            "Parameter": ["β₀", "β₁", "β₂", "β₃", "τ₁", "τ₂"],
            "Original": [default_params["beta0"], default_params["beta1"],
                         default_params["beta2"], default_params["beta3"],
                         default_params["tau1"], default_params["tau2"]],
            "User-Adjusted": [beta0, beta1, beta2, beta3, tau1, tau2],
        })
        param_table["Δ Change"] = param_table["User-Adjusted"] - param_table["Original"]
        st.dataframe(param_table, use_container_width=True, hide_index=True)

    # ── Download section ──
    st.markdown("---")
    st.markdown('<div class="download-section">', unsafe_allow_html=True)
    st.markdown("#### 💾 Download Curve Data")

    download_df = pd.DataFrame({
        "Maturity": maturities,
        "Original_Rate": rates_df["Zero_Coupon_Rate"].values,
        "User_Adjusted_Rate": np.round(dynamic_rates, 4),
        "Spread_bps": np.round((dynamic_rates - rates_df["Zero_Coupon_Rate"].values) * 100, 2),
    })

    csv_buffer = io.StringIO()
    download_df.to_csv(csv_buffer, index=False)
    csv_data = csv_buffer.getvalue()

    col_dl1, col_dl2, col_dl3 = st.columns([2, 2, 3])
    with col_dl1:
        st.download_button(
            label="📥 Download as CSV",
            data=csv_data,
            file_name=f"nss_yield_curves_{curve_date}.csv",
            mime="text/csv",
            use_container_width=True,
        )
    with col_dl2:
        st.caption(f"Contains {len(download_df)} data points  ·  {download_df.columns.tolist()}")

    st.markdown("</div>", unsafe_allow_html=True)


# ══════════════════════════════════════════════
# PAGE 2 — Model Explanation
# ══════════════════════════════════════════════
elif page == "📖  Model Explanation":

    # Header
    st.markdown("""
    <div class="main-header">
        <h1>📖 Nelson-Siegel-Svensson Model</h1>
        <p>A comprehensive guide to the yield curve model used for Indian Government Securities</p>
    </div>
    """, unsafe_allow_html=True)

    # ── Section 1: What is a Yield Curve? ──
    st.markdown('<div class="info-section">', unsafe_allow_html=True)
    st.markdown("### 🔍 What is a Yield Curve?")
    st.markdown("""
A **yield curve** plots the interest rates (yields) of bonds with equal credit quality
but differing maturity dates. For government securities, the **Zero Coupon Yield Curve (ZCYC)**
represents the term structure of interest rates — the relationship between the time to maturity
and the yield of zero-coupon bonds.

The ZCYC is a foundational tool in fixed-income markets because it:

- **Benchmarks pricing** for all fixed-income instruments (bonds, swaps, FRAs)
- **Signals monetary policy expectations** — an inverted curve often precedes recessions
- **Enables valuation** of any cash flow stream by discounting at maturity-specific rates
- **Drives risk management** through duration and key rate duration analysis
    """)
    st.markdown("</div>", unsafe_allow_html=True)

    # ── Section 2: The Nelson-Siegel Model ──
    st.markdown('<div class="info-section">', unsafe_allow_html=True)
    st.markdown("### 📊 The Nelson-Siegel Model (1987)")
    st.markdown("""
**Charles Nelson** and **Andrew Siegel** proposed a parsimonious model in 1987 to fit the
term structure using just **four parameters** (β₀, β₁, β₂, τ):
    """)
    st.latex(
        r"r(m) = \beta_0 "
        r"+ \beta_1 \left(\frac{1 - e^{-m/\tau}}{m/\tau}\right) "
        r"+ \beta_2 \left(\frac{1 - e^{-m/\tau}}{m/\tau} - e^{-m/\tau}\right)"
    )
    st.markdown("""
**Parameter interpretation:**

| Parameter | Role | Effect on Curve |
|-----------|------|-----------------|
| **β₀** | Long-run level | The asymptotic rate as maturity → ∞ |
| **β₁** | Short-term component | Determines the slope; β₀ + β₁ = instantaneous short rate |
| **β₂** | Medium-term (hump) | Creates a hump or trough; magnitude and sign control the curvature |
| **τ** | Decay speed | Where the hump peaks; larger τ → hump shifts to longer maturities |

The model elegantly captures the three classic yield curve shapes:
- **Normal** (upward-sloping): β₁ < 0
- **Inverted** (downward-sloping): β₁ > 0
- **Humped**: non-zero β₂
    """)
    st.markdown("</div>", unsafe_allow_html=True)

    # ── Section 3: Svensson Extension ──
    st.markdown('<div class="info-section">', unsafe_allow_html=True)
    st.markdown("### 🚀 The Svensson Extension (1994)")
    st.markdown("""
**Lars Svensson** extended the Nelson-Siegel model by adding a **second hump term** (β₃, τ₂),
which gives the model more flexibility to fit complex yield curve shapes — particularly when
the curve exhibits a double hump or unusual features at specific maturities.
    """)
    st.latex(
        r"r(m) = \beta_0 "
        r"+ \beta_1 \left(\frac{1 - e^{-m/\tau_1}}{m/\tau_1}\right) "
        r"+ \beta_2 \left(\frac{1 - e^{-m/\tau_1}}{m/\tau_1} - e^{-m/\tau_1}\right) "
        r"+ \beta_3 \left(\frac{1 - e^{-m/\tau_2}}{m/\tau_2} - e^{-m/\tau_2}\right)"
    )
    st.markdown("""
**Additional parameters:**

| Parameter | Role | Effect on Curve |
|-----------|------|-----------------|
| **β₃** | Second curvature factor | Creates an additional hump/trough at a different maturity |
| **τ₂** | Second decay speed | Controls where the second hump peaks |

This **six-parameter model** is used by many central banks and financial institutions
worldwide, including the **Reserve Bank of India** and **FBIL/CCIL** for constructing
the official Indian government securities yield curve.
    """)
    st.markdown("</div>", unsafe_allow_html=True)

    # ── Section 4: Indian Context ──
    st.markdown('<div class="info-section">', unsafe_allow_html=True)
    st.markdown("### 🇮🇳 ZCYC in the Indian Context")
    st.markdown("""
In India, the **Financial Benchmarks India Limited (FBIL)** — a subsidiary of the
**Clearing Corporation of India Limited (CCIL)** — publishes the official **Zero Coupon
Yield Curve (ZCYC)** daily for Indian government securities.

**Key facts:**

- **Model used:** Nelson-Siegel-Svensson (NSS) with 6 parameters
- **Data source:** Actual traded prices and quotes of Government of India securities
- **Publication:** Daily, after market close
- **Coverage:** Maturities from 0.5 years to 50 years, at 0.5-year intervals
- **Purpose:** Benchmark for fixed-income pricing, valuation, and risk management
- **Regulatory role:** Used by banks, mutual funds, and insurance companies for
  mark-to-market valuation as per RBI guidelines

The FBIL ZCYC parameters (β₀, β₁, β₂, β₃, τ₁, τ₂) are estimated by minimizing the
sum of squared price errors between the model-implied prices and actual market prices
of government securities.
    """)
    st.markdown("</div>", unsafe_allow_html=True)

    # ── Section 5: Understanding Parameters ──
    st.markdown('<div class="info-section">', unsafe_allow_html=True)
    st.markdown("### 🎓 Understanding Each Parameter")

    col1, col2 = st.columns(2)

    with col1:
        st.markdown("""
**β₀ — Long-run Level**
> The rate the curve converges to at very long maturities.
> Represents the market's expectation of the long-term equilibrium
> interest rate. Typically between 5% and 10% for Indian G-Secs.

**β₁ — Short-term Slope**
> Determines how steep the curve is at the short end.
> - β₁ < 0 → normal (upward-sloping) curve
> - β₁ > 0 → inverted curve
> - β₀ + β₁ gives the instantaneous (overnight) rate

**β₂ — First Curvature (Hump)**
> Creates a hump or trough in the curve at maturity ≈ τ₁.
> - β₂ > 0 → hump (rates peak then decline)
> - β₂ < 0 → trough (rates dip then rise)
        """)

    with col2:
        st.markdown("""
**β₃ — Second Curvature (Svensson)**
> The additional flexibility from Svensson's extension.
> Creates a second hump/trough at maturity ≈ τ₂,
> allowing the model to fit more complex curve shapes.

**τ₁ — First Decay Factor**
> Controls the speed at which the β₁ and β₂ components
> decay toward zero. Larger values shift the hump to
> longer maturities.

**τ₂ — Second Decay Factor**
> Controls the decay speed of the β₃ component.
> Determines where the Svensson extension's
> second hump peaks along the maturity axis.
        """)

    st.markdown("</div>", unsafe_allow_html=True)

    # ── Section 6: References ──
    st.markdown('<div class="info-section">', unsafe_allow_html=True)
    st.markdown("### 📚 References & Further Reading")

    references = [
        {
            "title": "Nelson, C.R. & Siegel, A.F. (1987)",
            "detail": "\"Parsimonious Modeling of Yield Curves\" — Journal of Business, Vol. 60, No. 4, pp. 473–489. The foundational paper introducing the three-factor yield curve model.",
            "url": "https://www.jstor.org/stable/2352957",
        },
        {
            "title": "Svensson, L.E.O. (1994)",
            "detail": "\"Estimating and Interpreting Forward Interest Rates: Sweden 1992–1994\" — NBER Working Paper No. 4871. Extended the Nelson-Siegel model with a second curvature term.",
            "url": "https://www.nber.org/papers/w4871",
        },
        {
            "title": "BIS Papers No. 25 (2005)",
            "detail": "\"Zero-Coupon Yield Curves: Technical Documentation\" — Bank for International Settlements. Comprehensive guide to how central banks estimate yield curves.",
            "url": "https://www.bis.org/publ/bppdf/bispap25.htm",
        },
        {
            "title": "FBIL — Valuation of Government Securities",
            "detail": "Official documentation on how Financial Benchmarks India Limited constructs the ZCYC using the NSS model for Indian government securities.",
            "url": "https://www.fbil.org.in/",
        },
        {
            "title": "CCIL — Zero Coupon Yield Curve",
            "detail": "Clearing Corporation of India Limited — Daily ZCYC data and methodology documentation for Indian G-Secs.",
            "url": "https://www.ccilindia.com/web/ccil/zero-coupon-yield-curve",
        },
        {
            "title": "Diebold, F.X. & Li, C. (2006)",
            "detail": "\"Forecasting the Term Structure of Government Bond Yields\" — Journal of Econometrics. Influential work on dynamic Nelson-Siegel models for yield curve forecasting.",
            "url": "https://doi.org/10.1016/j.jeconom.2005.03.005",
        },
        {
            "title": "Gürkaynak, R.S., Sack, B. & Wright, J.H. (2007)",
            "detail": "\"The U.S. Treasury Yield Curve: 1961 to the Present\" — Journal of Monetary Economics. Federal Reserve implementation of NSS model.",
            "url": "https://doi.org/10.1016/j.jmoneco.2007.06.029",
        },
    ]

    for ref in references:
        st.markdown(f"""
        <div class="ref-card">
            <div class="ref-title">📄 {ref['title']}</div>
            <div class="ref-detail">{ref['detail']}</div>
            <div class="ref-detail" style="margin-top:4px">
                🔗 <a href="{ref['url']}" target="_blank" style="color:#c9a84c">{ref['url']}</a>
            </div>
        </div>
        """, unsafe_allow_html=True)

    st.markdown("</div>", unsafe_allow_html=True)


# ──────────────────────────────────────────────
# Footer
# ──────────────────────────────────────────────
st.markdown("---")
st.markdown(
    '<p style="text-align:center; color:#6b7280; font-size:0.8rem">'
    'NSS Yield Curve Dashboard &nbsp;·&nbsp; Data: FBIL / CCIL &nbsp;·&nbsp; '
    'Model: Nelson-Siegel-Svensson (1994)'
    '</p>',
    unsafe_allow_html=True,
)
