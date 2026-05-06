#!/usr/bin/env python3
"""
╔══════════════════════════════════════════════════════════════╗
║    MULTI-ASSET MOMENTUM PORTFOLIO — SECTOR ROTATION ENGINE    ║
║                   creator: Prince Maurya                     ║
╚══════════════════════════════════════════════════════════════╝
Auto-installs all dependencies. Just run: python momentum_portfolio.py
"""

import subprocess, sys

REQUIRED = ["yfinance", "pandas", "numpy", "matplotlib", "scipy", "ta"]

def install(pkg):
    subprocess.check_call([sys.executable, "-m", "pip", "install", pkg, "-q"])

print("Checking & installing dependencies...")
for pkg in REQUIRED:
    try:
        __import__("scipy" if pkg == "scipy" else pkg if pkg != "ta" else "ta")
    except ImportError:
        print(f"   Installing {pkg}...")
        install(pkg)
print(" All dependencies ready!\n")

import yfinance as yf
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.colors import LinearSegmentedColormap
import matplotlib.ticker as mtick
import ta
import warnings
warnings.filterwarnings("ignore")

# ─── DARK THEME ───────────────────────────────────────────────
plt.style.use("dark_background")
BG      = "#07090e"
PANEL   = "#0c0f18"
CARD    = "#111520"
CYAN    = "#00e5ff"
GREEN   = "#00ff9d"
RED     = "#ff2d55"
AMBER   = "#ffaa00"
PINK    = "#ff61d8"
LAVEND  = "#9b8fff"
GRID    = "#16192a"
TEXT    = "#b8c4d8"

# ─── UNIVERSE ─────────────────────────────────────────────────
# Sector ETFs — one per sector for clean rotation signals
TICKERS = {
    "XLK":  "Technology",
    "XLF":  "Financials",
    "XLE":  "Energy",
    "XLV":  "Healthcare",
    "XLI":  "Industrials",
    "XLC":  "Comm. Svcs",
    "XLY":  "Cons. Disc.",
    "XLP":  "Cons. Staples",
    "XLRE": "Real Estate",
    "XLB":  "Materials",
    "XLU":  "Utilities",
    "GLD":  "Gold",
}
BENCH       = "SPY"     # benchmark
PERIOD      = "2y"
MOM_SHORT   = 21        # 1-month momentum (trading days)
MOM_LONG    = 63        # 3-month momentum
MOM_VOL_WIN = 21        # volatility window for risk-adj momentum
TOP_N       = 4         # hold top-N momentum assets
REBAL_DAYS  = 21        # rebalance every ~1 month
CAPITAL     = 10_000

# ─── DATA ─────────────────────────────────────────────────────
syms = list(TICKERS.keys()) + [BENCH]
print(f"📡 Fetching {len(syms)} tickers ({PERIOD})...")

raw = yf.download(syms, period=PERIOD, auto_adjust=True, progress=False)["Close"]
raw.columns = [c[0] if isinstance(c, tuple) else c for c in raw.columns]
raw.dropna(axis=1, how="all", inplace=True)
raw.ffill(inplace=True)
raw.dropna(inplace=True)

# Separate benchmark
bench_prices = raw[BENCH]
asset_prices = raw[[t for t in TICKERS if t in raw.columns]]
names        = {k: v for k, v in TICKERS.items() if k in raw.columns}

print(f"   {len(asset_prices.columns)} assets loaded, {len(asset_prices)} trading days\n")

# ─── MOMENTUM SCORE ───────────────────────────────────────────
# Risk-adjusted momentum: (return / volatility) — penalises noisy outperformers
def mom_score(prices):
    r_short = prices.pct_change(MOM_SHORT)
    r_long  = prices.pct_change(MOM_LONG)
    vol     = prices.pct_change().rolling(MOM_VOL_WIN).std() * np.sqrt(252)
    score   = (0.6 * r_short + 0.4 * r_long) / (vol + 1e-9)
    return score

scores = mom_score(asset_prices)

# ─── BACKTEST LOOP ────────────────────────────────────────────
equity  = CAPITAL
records = []
weights_history = []
dates   = asset_prices.index[MOM_LONG:]

prev_portfolio = []
trade_count    = 0

for i, date in enumerate(dates):
    if i % REBAL_DAYS == 0:
        day_scores  = scores.loc[date].dropna()
        day_scores  = day_scores[day_scores > 0]          # only positive momentum
        top_assets  = day_scores.nlargest(TOP_N).index.tolist()

        if top_assets:
            prev_portfolio = top_assets
            trade_count   += 1

    portfolio = prev_portfolio if prev_portfolio else []

    if portfolio:
        # Equal weight
        w = {t: 1.0 / len(portfolio) for t in portfolio}
    else:
        w = {}

    weights_history.append({"date": date, "weights": dict(w)})

    if i > 0:
        prev_date   = dates[i - 1]
        port_return = sum(
            w.get(t, 0) * ((asset_prices.loc[date, t] / asset_prices.loc[prev_date, t]) - 1)
            for t in asset_prices.columns
            if t in w
        )
        equity *= (1 + port_return)

    records.append({"date": date, "equity": equity})

equity_df = pd.DataFrame(records).set_index("date")
equity_df["BH"] = CAPITAL * (bench_prices.loc[dates] / bench_prices.loc[dates[0]])

# ─── METRICS ──────────────────────────────────────────────────
final_val  = equity_df["equity"].iloc[-1]
strat_ret  = (final_val - CAPITAL) / CAPITAL * 100
bh_ret     = (equity_df["BH"].iloc[-1] - CAPITAL) / CAPITAL * 100
dr         = equity_df["equity"].pct_change().dropna()
sharpe     = (dr.mean() / dr.std()) * np.sqrt(252) if dr.std() > 0 else 0
max_dd     = ((equity_df["equity"] - equity_df["equity"].cummax()) / equity_df["equity"].cummax() * 100).min()
calmar     = strat_ret / abs(max_dd) if max_dd != 0 else 0

# Latest allocation
latest_w = weights_history[-1]["weights"]

# ─── CORRELATION MATRIX ───────────────────────────────────────
ret_mat = asset_prices.pct_change().dropna()
corr    = ret_mat.corr()
labels  = [names.get(c, c) for c in corr.columns]

# ─── MOMENTUM RANKINGS (latest) ───────────────────────────────
latest_scores = scores.iloc[-1].dropna().sort_values(ascending=False)
bar_colors    = [GREEN if t in latest_w else (AMBER if s > 0 else RED)
                 for t, s in latest_scores.items()]
bar_labels    = [names.get(t, t) for t in latest_scores.index]

# ─── 6-MONTH RETURN HEATMAP DATA ──────────────────────────────
hm_data  = asset_prices.pct_change().dropna()
hm_data.index = pd.to_datetime(hm_data.index)
monthly  = hm_data.resample("ME").apply(lambda x: (1 + x).prod() - 1)
monthly  = monthly.iloc[-12:] if len(monthly) >= 12 else monthly
hm_cols  = [names.get(c, c) for c in monthly.columns]

print(f"\n{'='*58}")
print(f"  BACKTEST RESULTS — Multi-Asset Momentum Portfolio")
print(f"{'='*58}")
print(f"  Starting Capital   : ${CAPITAL:,.0f}")
print(f"  Final Value        : ${final_val:,.2f}")
print(f"  Strategy Return    : {strat_ret:+.2f}%")
print(f"  SPY Buy & Hold     : {bh_ret:+.2f}%")
print(f"  Sharpe Ratio       : {sharpe:.3f}")
print(f"  Max Drawdown       : {max_dd:.2f}%")
print(f"  Calmar Ratio       : {calmar:.2f}")
print(f"  Rebalances         : {trade_count}")
print(f"\n  Current Portfolio  : {', '.join(latest_w.keys())}")
print(f"{'='*58}\n")

# ─── PLOT ─────────────────────────────────────────────────────
fig = plt.figure(figsize=(20, 16), facecolor=BG)
fig.suptitle(
    "  SECTOR ROTATION MOMENTUM PORTFOLIO  ·  12 Assets  ·  Risk-Adjusted Momentum",
    fontsize=15, fontweight="bold", color=CYAN,
    fontfamily="monospace", y=0.97
)

gs = gridspec.GridSpec(3, 3, figure=fig,
                        hspace=0.38, wspace=0.32,
                        top=0.93, bottom=0.05, left=0.05, right=0.97,
                        height_ratios=[1.8, 1.4, 1.6])

# ── Row 0: Equity (full) ──
ax_eq  = fig.add_subplot(gs[0, :])
# ── Row 1: Momentum bar | Pie allocation ──
ax_mom = fig.add_subplot(gs[1, :2])
ax_pie = fig.add_subplot(gs[1, 2])
# ── Row 2: Corr heatmap | Monthly heatmap ──
ax_cor = fig.add_subplot(gs[2, :2])
ax_mth = fig.add_subplot(gs[2, 2])

all_axes = [ax_eq, ax_mom, ax_pie, ax_cor, ax_mth]
for ax in all_axes:
    ax.set_facecolor(PANEL)
    ax.tick_params(colors=TEXT, labelsize=8)
    ax.spines[:].set_color(GRID)
    ax.grid(color=GRID, linewidth=0.35, alpha=0.6)

# ── Equity Curve ──
ax_eq.plot(equity_df.index, equity_df["equity"], color=CYAN,   lw=1.8, label=f"Momentum Portfolio  {strat_ret:+.1f}%", zorder=3)
ax_eq.plot(equity_df.index, equity_df["BH"],     color=AMBER,  lw=1.1, ls="--", label=f"SPY Buy & Hold  {bh_ret:+.1f}%", alpha=0.85)
ax_eq.fill_between(equity_df.index, equity_df["equity"], equity_df["BH"],
                   where=equity_df["equity"] >= equity_df["BH"], alpha=0.12, color=GREEN, zorder=2)
ax_eq.fill_between(equity_df.index, equity_df["equity"], equity_df["BH"],
                   where=equity_df["equity"] < equity_df["BH"],  alpha=0.12, color=RED, zorder=2)
ax_eq.axhline(CAPITAL, color=TEXT, lw=0.5, ls=":", alpha=0.4)
ax_eq.set_ylabel("Portfolio Value ($)", color=TEXT, fontsize=9)
ax_eq.legend(loc="upper left", fontsize=8.5, facecolor=PANEL, edgecolor=GRID, labelcolor=TEXT)
ax_eq.set_title("Equity Curve vs Benchmark", color=TEXT, fontsize=9, pad=4)
ax_eq.yaxis.set_major_formatter(mtick.FuncFormatter(lambda x, _: f"${x:,.0f}"))

# ── Momentum Bar Chart ──
y_pos = range(len(latest_scores))
ax_mom.barh(y_pos, latest_scores.values, color=bar_colors, alpha=0.85, height=0.65)
ax_mom.set_yticks(list(y_pos))
ax_mom.set_yticklabels(bar_labels, fontsize=7.5, color=TEXT)
ax_mom.axvline(0, color=TEXT, lw=0.5, alpha=0.4)
ax_mom.set_xlabel("Risk-Adj Momentum Score", color=TEXT, fontsize=8)
ax_mom.set_title("Current Momentum Rankings  (■ selected  ■ positive  ■ negative)",
                  color=TEXT, fontsize=8.5, pad=4)
for spine in ax_mom.spines.values():
    spine.set_visible(False)
ax_mom.grid(axis="x", color=GRID, linewidth=0.4)
ax_mom.grid(axis="y", visible=False)

# Add legend patches
from matplotlib.patches import Patch
legend_elements = [Patch(facecolor=GREEN, label="In Portfolio"),
                   Patch(facecolor=AMBER, label="Positive Mom."),
                   Patch(facecolor=RED,   label="Negative Mom.")]
ax_mom.legend(handles=legend_elements, fontsize=7, facecolor=PANEL,
              edgecolor=GRID, labelcolor=TEXT, loc="lower right")

# ── Pie — Current Allocation ──
if latest_w:
    pie_labels = [names.get(t, t) for t in latest_w.keys()]
    pie_vals   = list(latest_w.values())
    pie_colors = [CYAN, GREEN, PINK, LAVEND, AMBER, RED][:len(pie_vals)]
    wedges, texts, autotexts = ax_pie.pie(
        pie_vals, labels=pie_labels, colors=pie_colors,
        autopct="%1.0f%%", startangle=90, pctdistance=0.75,
        wedgeprops=dict(edgecolor=PANEL, linewidth=2),
        textprops=dict(color=TEXT, fontsize=7.5)
    )
    for at in autotexts:
        at.set_color(BG)
        at.set_fontsize(8)
        at.set_fontweight("bold")
ax_pie.set_title("Current Allocation", color=TEXT, fontsize=9, pad=4)
ax_pie.grid(visible=False)

# ── Correlation Heatmap ──
ax_cor.set_facecolor(PANEL)
cmap_corr = LinearSegmentedColormap.from_list(
    "dark_diverge", [RED, PANEL, CYAN], N=256
)
corr_vals = corr.values
im = ax_cor.imshow(corr_vals, cmap=cmap_corr, vmin=-1, vmax=1, aspect="auto")
ax_cor.set_xticks(range(len(labels)))
ax_cor.set_yticks(range(len(labels)))
ax_cor.set_xticklabels(labels, rotation=55, ha="right", fontsize=6.5, color=TEXT)
ax_cor.set_yticklabels(labels, fontsize=6.5, color=TEXT)
for i in range(len(labels)):
    for j in range(len(labels)):
        c = corr_vals[i, j]
        ax_cor.text(j, i, f"{c:.2f}", ha="center", va="center",
                    fontsize=5.5,
                    color=BG if abs(c) > 0.5 else TEXT)
ax_cor.set_title("Asset Correlation Matrix", color=TEXT, fontsize=9, pad=4)
ax_cor.grid(visible=False)
cb = plt.colorbar(im, ax=ax_cor, shrink=0.8, pad=0.02)
cb.ax.tick_params(labelsize=7, colors=TEXT)
cb.outline.set_edgecolor(GRID)

# ── Monthly Return Heatmap ──
ax_mth.set_facecolor(PANEL)
cmap_ret = LinearSegmentedColormap.from_list("ret_map", [RED, PANEL, GREEN], N=256)
max_abs = np.nanmax(np.abs(monthly.values))
im2 = ax_mth.imshow(monthly.values.T * 100, cmap=cmap_ret,
                     vmin=-max_abs*100, vmax=max_abs*100, aspect="auto")
ax_mth.set_yticks(range(len(hm_cols)))
ax_mth.set_yticklabels(hm_cols, fontsize=6, color=TEXT)
month_labels = [d.strftime("%b %y") for d in monthly.index]
ax_mth.set_xticks(range(len(monthly)))
ax_mth.set_xticklabels(month_labels, rotation=55, ha="right", fontsize=5.5, color=TEXT)
for i in range(len(hm_cols)):
    for j in range(len(monthly)):
        v = monthly.values[j, i] * 100
        if not np.isnan(v):
            ax_mth.text(j, i, f"{v:+.1f}", ha="center", va="center",
                        fontsize=5, color=BG if abs(v) > max_abs*50 else TEXT)
ax_mth.set_title("Monthly Returns (%)", color=TEXT, fontsize=9, pad=4)
ax_mth.grid(visible=False)
cb2 = plt.colorbar(im2, ax=ax_mth, shrink=0.8, pad=0.02, format="%+.0f%%")
cb2.ax.tick_params(labelsize=6, colors=TEXT)
cb2.outline.set_edgecolor(GRID)

# Stats footer
stats_txt = (
    f"Sharpe: {sharpe:.2f}   Calmar: {calmar:.2f}   "
    f"MaxDD: {max_dd:.1f}%   Rebalances: {trade_count}   "
    f"Final: ${final_val:,.0f}   Alpha vs SPY: {strat_ret - bh_ret:+.1f}%"
)
fig.text(0.5, 0.01, stats_txt, ha="center", fontsize=9, color=AMBER,
         fontfamily="monospace",
         bbox=dict(facecolor=PANEL, edgecolor=GRID, boxstyle="round,pad=0.4"))

plt.savefig("momentum_portfolio_result.png", dpi=150, bbox_inches="tight", facecolor=BG)
print("Chart saved → momentum_portfolio_result.png")
plt.show()
