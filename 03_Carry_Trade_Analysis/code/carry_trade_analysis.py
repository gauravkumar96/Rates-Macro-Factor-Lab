"""
USD/INR Carry Trade Analysis - US Investor's Perspective
=========================================================
This script analyses the carry trade from the viewpoint of a US-based investor who:
1. Borrows in USD at the US Fed Funds rate
2. Converts to INR, invests at the RBI Repo rate
3. Converts back to USD at year-end

P&L = Interest Rate Differential - INR Depreciation (in USD terms)
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import seaborn as sns
from pathlib import Path

# ── Paths ──────────────────────────────────────────────────────────────────────
BASE_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = BASE_DIR / "data"
OUTPUT_DIR = BASE_DIR / "output"
OUTPUT_DIR.mkdir(exist_ok=True)

# ── Style ──────────────────────────────────────────────────────────────────────
sns.set_theme(style="whitegrid", font_scale=1.1)
COLORS = {
    "carry": "#2196F3",
    "depreciation": "#F44336",
    "net_return": "#4CAF50",
    "negative": "#F44336",
    "positive": "#4CAF50",
    "india": "#FF9800",
    "us": "#2196F3",
    "cumulative": "#9C27B0",
    "differential": "#FF5722",
}


def load_data():
    """Load and prepare the annual dataset."""
    df = pd.read_csv(DATA_DIR / "usdinr_annual_data.csv")
    return df


def compute_investor_pnl(df):
    """
    Compute carry trade P&L from a US investor's perspective.

    For each year Y:
      - Carry earned = RBI Repo Rate(Y-1 end) - US Fed Rate(Y-1 end)
        (the differential at the START of the year, which is the rate locked in)
      - FX return = (USDINR_Y-1 / USDINR_Y) - 1
        (if INR depreciates, USDINR rises, investor loses on conversion back)
      - INR depreciation = (USDINR_Y - USDINR_Y-1) / USDINR_Y-1
      - Net USD return = Carry earned - INR depreciation
      - Total USD return = (1 + INR_rate) * (1 / (1 + depreciation_pct)) - 1 - USD_borrowing_cost
        Simplified: ≈ carry_differential - depreciation
    """
    results = []

    for i in range(1, len(df)):
        year = df.loc[i, "Year"]
        prev = df.loc[i - 1]
        curr = df.loc[i]

        # Rates at start of year (= previous year-end)
        inr_rate = prev["RBI_Repo_Rate_YearEnd"] / 100
        usd_rate = prev["US_Fed_Rate_YearEnd"] / 100
        carry_differential = (prev["RBI_Repo_Rate_YearEnd"] - prev["US_Fed_Rate_YearEnd"])

        # FX movement during the year
        usdinr_start = prev["USDINR_Avg"]
        usdinr_end = curr["USDINR_Avg"]
        inr_depreciation = ((usdinr_end - usdinr_start) / usdinr_start) * 100

        # Exact P&L calculation (in USD terms)
        # Investor puts $1 -> gets INR at start rate -> earns INR interest -> converts back
        inr_received = usdinr_start  # INR per $1
        inr_at_yearend = inr_received * (1 + inr_rate)  # INR after interest
        usd_at_yearend = inr_at_yearend / usdinr_end  # convert back to USD
        gross_usd_return = (usd_at_yearend - 1) * 100  # % return before borrowing cost
        borrowing_cost = usd_rate * 100  # cost of funding in USD
        net_usd_return = gross_usd_return - borrowing_cost  # net P&L

        # Simplified approximation for comparison
        approx_return = carry_differential - inr_depreciation

        results.append({
            "Year": int(year),
            "USDINR_Start": round(usdinr_start, 2),
            "USDINR_End": round(usdinr_end, 2),
            "INR_Rate_pct": round(prev["RBI_Repo_Rate_YearEnd"], 2),
            "USD_Rate_pct": round(prev["US_Fed_Rate_YearEnd"], 2),
            "Carry_Differential_pct": round(carry_differential, 2),
            "INR_Depreciation_pct": round(inr_depreciation, 2),
            "Gross_USD_Return_pct": round(gross_usd_return, 2),
            "USD_Borrowing_Cost_pct": round(borrowing_cost, 2),
            "Net_USD_Return_pct": round(net_usd_return, 2),
            "Approx_Return_pct": round(approx_return, 2),
        })

    return pd.DataFrame(results)


def compute_cumulative_returns(pnl_df):
    """Compute cumulative $1 investment growth."""
    cumulative = [1.0]
    for ret in pnl_df["Net_USD_Return_pct"]:
        cumulative.append(cumulative[-1] * (1 + ret / 100))
    pnl_df = pnl_df.copy()
    pnl_df["Cumulative_Value"] = cumulative[1:]
    return pnl_df


def compute_risk_metrics(pnl_df):
    """Compute key risk/return metrics."""
    returns = pnl_df["Net_USD_Return_pct"].values
    n = len(returns)

    avg_return = np.mean(returns)
    std_return = np.std(returns, ddof=1)
    sharpe = avg_return / std_return if std_return != 0 else 0

    # Max drawdown from cumulative
    cum = pnl_df["Cumulative_Value"].values
    peak = np.maximum.accumulate(cum)
    drawdown = (cum - peak) / peak * 100
    max_dd = np.min(drawdown)

    # Win rate
    wins = np.sum(returns > 0)
    win_rate = wins / n * 100

    # Best / worst
    best = np.max(returns)
    worst = np.min(returns)
    best_year = pnl_df.loc[pnl_df["Net_USD_Return_pct"].idxmax(), "Year"]
    worst_year = pnl_df.loc[pnl_df["Net_USD_Return_pct"].idxmin(), "Year"]

    # Total return
    total_return = (cum[-1] - 1) * 100

    # Average carry and depreciation
    avg_carry = pnl_df["Carry_Differential_pct"].mean()
    avg_depreciation = pnl_df["INR_Depreciation_pct"].mean()

    metrics = {
        "Period": f"{int(pnl_df['Year'].iloc[0])}-{int(pnl_df['Year'].iloc[-1])}",
        "Number of Years": n,
        "Average Annual Return (%)": round(avg_return, 2),
        "Std Dev of Returns (%)": round(std_return, 2),
        "Sharpe Ratio (no risk-free adj)": round(sharpe, 2),
        "Total Cumulative Return (%)": round(total_return, 2),
        "Max Drawdown (%)": round(max_dd, 2),
        "Win Rate (%)": round(win_rate, 1),
        "Best Year Return (%)": f"{round(best, 2)} ({int(best_year)})",
        "Worst Year Return (%)": f"{round(worst, 2)} ({int(worst_year)})",
        "Avg Carry Differential (%)": round(avg_carry, 2),
        "Avg INR Depreciation (%)": round(avg_depreciation, 2),
        "Carry vs Depreciation": "Carry WINS" if avg_carry > avg_depreciation else "Depreciation WINS",
    }
    return metrics


# ═══════════════════════════════════════════════════════════════════════════════
# PLOTTING FUNCTIONS
# ═══════════════════════════════════════════════════════════════════════════════

def plot_carry_vs_depreciation(pnl_df):
    """Plot 1: Carry earned vs INR depreciation side by side."""
    fig, ax = plt.subplots(figsize=(14, 7))

    x = np.arange(len(pnl_df))
    width = 0.35

    bars1 = ax.bar(x - width/2, pnl_df["Carry_Differential_pct"], width,
                   label="Carry Earned (Rate Differential)", color=COLORS["carry"], edgecolor="white")
    bars2 = ax.bar(x + width/2, pnl_df["INR_Depreciation_pct"], width,
                   label="INR Depreciation (Cost)", color=COLORS["depreciation"], edgecolor="white")

    ax.set_xlabel("Year", fontsize=12)
    ax.set_ylabel("Percentage (%)", fontsize=12)
    ax.set_title("Carry Trade: Interest Earned vs Currency Cost\n(US Investor Perspective)",
                 fontsize=14, fontweight="bold")
    ax.set_xticks(x)
    ax.set_xticklabels(pnl_df["Year"].astype(int))
    ax.legend(fontsize=11, loc="upper right")
    ax.axhline(y=0, color="black", linewidth=0.8)

    # Add value labels
    for bar in bars1:
        h = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2, h + 0.1, f"{h:.1f}",
                ha="center", va="bottom", fontsize=9)
    for bar in bars2:
        h = bar.get_height()
        offset = 0.1 if h >= 0 else -0.4
        ax.text(bar.get_x() + bar.get_width()/2, h + offset, f"{h:.1f}",
                ha="center", va="bottom", fontsize=9)

    plt.tight_layout()
    fig.savefig(OUTPUT_DIR / "01_carry_vs_depreciation.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    print("  Saved: 01_carry_vs_depreciation.png")


def plot_net_pnl(pnl_df):
    """Plot 2: Net carry trade P&L waterfall."""
    fig, ax = plt.subplots(figsize=(14, 7))

    colors = [COLORS["positive"] if r > 0 else COLORS["negative"]
              for r in pnl_df["Net_USD_Return_pct"]]

    bars = ax.bar(pnl_df["Year"].astype(str), pnl_df["Net_USD_Return_pct"],
                  color=colors, edgecolor="white", linewidth=0.8)

    ax.axhline(y=0, color="black", linewidth=1)
    ax.set_xlabel("Year", fontsize=12)
    ax.set_ylabel("Net USD Return (%)", fontsize=12)
    ax.set_title("Carry Trade Net P&L by Year\n(US Investor: Borrow USD, Invest INR, Convert Back)",
                 fontsize=14, fontweight="bold")

    # Value labels
    for bar, val in zip(bars, pnl_df["Net_USD_Return_pct"]):
        h = bar.get_height()
        offset = 0.15 if h >= 0 else -0.45
        ax.text(bar.get_x() + bar.get_width()/2, h + offset, f"{val:.2f}%",
                ha="center", va="bottom" if h >= 0 else "top", fontsize=10, fontweight="bold")

    # Add average line
    avg = pnl_df["Net_USD_Return_pct"].mean()
    ax.axhline(y=avg, color=COLORS["cumulative"], linewidth=1.5, linestyle="--",
               label=f"Average: {avg:.2f}%")
    ax.legend(fontsize=11)

    plt.tight_layout()
    fig.savefig(OUTPUT_DIR / "02_net_pnl_by_year.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    print("  Saved: 02_net_pnl_by_year.png")


def plot_cumulative_return(pnl_df):
    """Plot 3: Cumulative $1 investment growth."""
    fig, ax = plt.subplots(figsize=(14, 7))

    years = [pnl_df["Year"].iloc[0] - 1] + list(pnl_df["Year"])
    values = [1.0] + list(pnl_df["Cumulative_Value"])

    ax.plot(years, values, marker="o", linewidth=2.5, color=COLORS["cumulative"],
            markersize=8, markerfacecolor="white", markeredgewidth=2)
    ax.fill_between(years, 1.0, values, alpha=0.15, color=COLORS["cumulative"])
    ax.axhline(y=1.0, color="gray", linewidth=1, linestyle="--", label="Break-even ($1.00)")

    # Annotate final value
    final = values[-1]
    ax.annotate(f"${final:.3f}", xy=(years[-1], final),
                xytext=(15, 15), textcoords="offset points",
                fontsize=12, fontweight="bold", color=COLORS["cumulative"],
                arrowprops=dict(arrowstyle="->", color=COLORS["cumulative"]))

    ax.set_xlabel("Year", fontsize=12)
    ax.set_ylabel("Portfolio Value ($)", fontsize=12)
    ax.set_title("Cumulative Value of $1 Invested in USD/INR Carry Trade\n(Unhedged, US Investor Perspective)",
                 fontsize=14, fontweight="bold")
    ax.legend(fontsize=11)
    ax.set_xticks(years)

    plt.tight_layout()
    fig.savefig(OUTPUT_DIR / "03_cumulative_return.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    print("  Saved: 03_cumulative_return.png")


def plot_interest_rates(df):
    """Plot 4: Policy rates comparison."""
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 10), sharex=True)

    # Top: Policy rates
    ax1.plot(df["Year"], df["RBI_Repo_Rate_YearEnd"], marker="s", linewidth=2.5,
             color=COLORS["india"], label="RBI Repo Rate", markersize=8)
    ax1.plot(df["Year"], df["US_Fed_Rate_YearEnd"], marker="o", linewidth=2.5,
             color=COLORS["us"], label="US Fed Funds Rate", markersize=8)
    ax1.fill_between(df["Year"], df["US_Fed_Rate_YearEnd"], df["RBI_Repo_Rate_YearEnd"],
                     alpha=0.2, color=COLORS["differential"], label="Differential (carry)")
    ax1.set_ylabel("Interest Rate (%)", fontsize=12)
    ax1.set_title("Policy Interest Rates: India vs United States (2016-2026)",
                  fontsize=14, fontweight="bold")
    ax1.legend(fontsize=11)

    # Bottom: Differential
    differential = df["RBI_Repo_Rate_YearEnd"] - df["US_Fed_Rate_YearEnd"]
    colors_diff = [COLORS["positive"] if d > 2 else COLORS["india"] if d > 0 else COLORS["negative"]
                   for d in differential]
    ax2.bar(df["Year"], differential, color=colors_diff, edgecolor="white")
    ax2.axhline(y=0, color="black", linewidth=0.8)
    ax2.axhline(y=2.0, color="gray", linewidth=1, linestyle="--",
                label="2% threshold (min viable carry)")
    ax2.set_xlabel("Year", fontsize=12)
    ax2.set_ylabel("Rate Differential (%)", fontsize=12)
    ax2.set_title("Interest Rate Differential (RBI Repo - US Fed Funds)", fontsize=13)
    ax2.legend(fontsize=11)
    ax2.set_xticks(df["Year"])

    for i, d in enumerate(differential):
        ax2.text(df["Year"].iloc[i], d + 0.1, f"{d:.1f}%", ha="center", fontsize=9)

    plt.tight_layout()
    fig.savefig(OUTPUT_DIR / "04_interest_rates.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    print("  Saved: 04_interest_rates.png")


def plot_usdinr_trend(df):
    """Plot 5: USDINR exchange rate trend with annotations."""
    fig, ax = plt.subplots(figsize=(14, 7))

    ax.plot(df["Year"], df["USDINR_Avg"], marker="o", linewidth=2.5,
            color="#E91E63", markersize=10, markerfacecolor="white", markeredgewidth=2.5)
    ax.fill_between(df["Year"], df["USDINR_Avg"].min() - 2, df["USDINR_Avg"],
                    alpha=0.1, color="#E91E63")

    # Annotate key events
    annotations = {
        2016: "Demonetisation",
        2018: "Oil & EM Crisis",
        2020: "COVID-19",
        2022: "Fed Hikes",
        2026: "Tariff Shock\n(Hit 99.82)",
    }
    for year, text in annotations.items():
        row = df[df["Year"] == year]
        if not row.empty:
            val = row["USDINR_Avg"].values[0]
            ax.annotate(text, xy=(year, val), xytext=(0, 20),
                        textcoords="offset points", ha="center", fontsize=9,
                        fontweight="bold", color="#880E4F",
                        arrowprops=dict(arrowstyle="->", color="#880E4F"))

    for _, row in df.iterrows():
        ax.text(row["Year"], row["USDINR_Avg"] - 1.5, f"{row['USDINR_Avg']:.1f}",
                ha="center", fontsize=9)

    ax.set_xlabel("Year", fontsize=12)
    ax.set_ylabel("USD/INR Exchange Rate", fontsize=12)
    ax.set_title("USD/INR Exchange Rate Trend (2016-2026)\nINR Depreciated ~39% Over the Decade",
                 fontsize=14, fontweight="bold")
    ax.set_xticks(df["Year"])

    plt.tight_layout()
    fig.savefig(OUTPUT_DIR / "05_usdinr_trend.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    print("  Saved: 05_usdinr_trend.png")


def plot_pnl_decomposition(pnl_df):
    """Plot 6: Stacked decomposition of P&L components."""
    fig, ax = plt.subplots(figsize=(14, 7))

    x = np.arange(len(pnl_df))
    width = 0.6

    # Gross return from INR investment (converted back to USD)
    ax.bar(x, pnl_df["Gross_USD_Return_pct"], width, label="Gross USD Return (INR invest + FX)",
           color=COLORS["carry"], edgecolor="white")
    ax.bar(x, -pnl_df["USD_Borrowing_Cost_pct"], width, bottom=pnl_df["Gross_USD_Return_pct"],
           label="USD Borrowing Cost (-)", color=COLORS["depreciation"], edgecolor="white", alpha=0.7)

    # Net return line
    ax.plot(x, pnl_df["Net_USD_Return_pct"], marker="D", linewidth=2, color="black",
            markersize=7, label="Net P&L", zorder=5)

    ax.axhline(y=0, color="black", linewidth=1)
    ax.set_xlabel("Year", fontsize=12)
    ax.set_ylabel("Return (%)", fontsize=12)
    ax.set_title("P&L Decomposition: Gross Return vs Borrowing Cost\n(US Investor Perspective)",
                 fontsize=14, fontweight="bold")
    ax.set_xticks(x)
    ax.set_xticklabels(pnl_df["Year"].astype(int))
    ax.legend(fontsize=11, loc="lower left")

    plt.tight_layout()
    fig.savefig(OUTPUT_DIR / "06_pnl_decomposition.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    print("  Saved: 06_pnl_decomposition.png")


def plot_inflation_differential(df):
    """Plot 7: Inflation differential vs actual depreciation (PPP check)."""
    fig, ax = plt.subplots(figsize=(14, 7))

    # Compute expected depreciation from inflation differential (PPP)
    inflation_diff = df["India_CPI_Inflation"] - df["US_CPI_Inflation"]

    # Actual depreciation (shift by 1)
    actual_depr = []
    for i in range(len(df)):
        if i == 0:
            actual_depr.append(np.nan)
        else:
            actual_depr.append(
                ((df.loc[i, "USDINR_Avg"] - df.loc[i-1, "USDINR_Avg"]) / df.loc[i-1, "USDINR_Avg"]) * 100
            )
    df_plot = df.copy()
    df_plot["Inflation_Diff"] = inflation_diff
    df_plot["Actual_Depreciation"] = actual_depr

    # Plot from 2017 onward (need prior year for depreciation calc)
    dp = df_plot.iloc[1:].copy()

    ax.bar(dp["Year"].astype(str), dp["Actual_Depreciation"], width=0.5,
           color=COLORS["depreciation"], alpha=0.7, label="Actual INR Depreciation")
    ax.plot(dp["Year"].astype(str), dp["Inflation_Diff"], marker="o", linewidth=2.5,
            color=COLORS["india"], markersize=8, label="Inflation Differential (India - US)")

    ax.axhline(y=0, color="black", linewidth=0.8)
    ax.set_xlabel("Year", fontsize=12)
    ax.set_ylabel("Percentage (%)", fontsize=12)
    ax.set_title("PPP Check: Inflation Differential vs Actual INR Depreciation\n"
                 "(Depreciation consistently overshoots inflation differential)",
                 fontsize=14, fontweight="bold")
    ax.legend(fontsize=11)

    plt.tight_layout()
    fig.savefig(OUTPUT_DIR / "07_inflation_vs_depreciation.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    print("  Saved: 07_inflation_vs_depreciation.png")


def plot_rolling_carry_attractiveness(pnl_df):
    """Plot 8: Carry-to-risk ratio over time."""
    fig, ax = plt.subplots(figsize=(14, 7))

    # Simple carry attractiveness = differential / abs(depreciation)
    # Higher = more attractive
    pnl_df = pnl_df.copy()
    pnl_df["Carry_Risk_Ratio"] = pnl_df["Carry_Differential_pct"] / pnl_df["INR_Depreciation_pct"].abs().clip(lower=0.5)

    colors = [COLORS["positive"] if r > 1 else COLORS["india"] if r > 0.5 else COLORS["negative"]
              for r in pnl_df["Carry_Risk_Ratio"]]

    ax.bar(pnl_df["Year"].astype(str), pnl_df["Carry_Risk_Ratio"], color=colors, edgecolor="white")
    ax.axhline(y=1.0, color="black", linewidth=1.5, linestyle="--", label="Break-even (ratio = 1.0)")
    ax.axhline(y=0.5, color=COLORS["negative"], linewidth=1, linestyle=":",
               label="Danger zone (ratio < 0.5)")

    ax.set_xlabel("Year", fontsize=12)
    ax.set_ylabel("Carry / Depreciation Ratio", fontsize=12)
    ax.set_title("Carry Attractiveness Ratio\n(Carry Earned / Currency Cost — above 1.0 = profitable)",
                 fontsize=14, fontweight="bold")
    ax.legend(fontsize=11)

    for i, (yr, val) in enumerate(zip(pnl_df["Year"], pnl_df["Carry_Risk_Ratio"])):
        ax.text(i, val + 0.05, f"{val:.2f}", ha="center", fontsize=9, fontweight="bold")

    plt.tight_layout()
    fig.savefig(OUTPUT_DIR / "08_carry_attractiveness.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    print("  Saved: 08_carry_attractiveness.png")


# ═══════════════════════════════════════════════════════════════════════════════
# EXPORT FUNCTIONS
# ═══════════════════════════════════════════════════════════════════════════════

def export_to_excel(df, pnl_df, metrics):
    """Export all data to a single Excel workbook."""
    filepath = OUTPUT_DIR / "carry_trade_data.xlsx"

    with pd.ExcelWriter(filepath, engine="openpyxl") as writer:
        df.to_excel(writer, sheet_name="Raw_Data", index=False)
        pnl_df.to_excel(writer, sheet_name="Investor_PnL", index=False)

        metrics_df = pd.DataFrame(list(metrics.items()), columns=["Metric", "Value"])
        metrics_df.to_excel(writer, sheet_name="Risk_Metrics", index=False)

    print(f"  Saved: {filepath.name}")


# ═══════════════════════════════════════════════════════════════════════════════
# MAIN
# ═══════════════════════════════════════════════════════════════════════════════

def main():
    print("=" * 70)
    print("USD/INR CARRY TRADE ANALYSIS — US INVESTOR PERSPECTIVE")
    print("=" * 70)

    # Load data
    print("\n[1] Loading data...")
    df = load_data()
    print(f"    Loaded {len(df)} years of data ({df['Year'].min()}-{df['Year'].max()})")

    # Compute P&L
    print("\n[2] Computing investor P&L...")
    pnl_df = compute_investor_pnl(df)
    pnl_df = compute_cumulative_returns(pnl_df)
    print(pnl_df[["Year", "Carry_Differential_pct", "INR_Depreciation_pct",
                   "Net_USD_Return_pct", "Cumulative_Value"]].to_string(index=False))

    # Risk metrics
    print("\n[3] Computing risk metrics...")
    metrics = compute_risk_metrics(pnl_df)
    for k, v in metrics.items():
        print(f"    {k}: {v}")

    # Generate plots
    print("\n[4] Generating plots...")
    plot_carry_vs_depreciation(pnl_df)
    plot_net_pnl(pnl_df)
    plot_cumulative_return(pnl_df)
    plot_interest_rates(df)
    plot_usdinr_trend(df)
    plot_pnl_decomposition(pnl_df)
    plot_inflation_differential(df)
    plot_rolling_carry_attractiveness(pnl_df)

    # Export data
    print("\n[5] Exporting data to Excel...")
    export_to_excel(df, pnl_df, metrics)

    print("\n" + "=" * 70)
    print("ANALYSIS COMPLETE. All outputs saved to:", OUTPUT_DIR)
    print("=" * 70)

    return df, pnl_df, metrics


if __name__ == "__main__":
    df, pnl_df, metrics = main()
