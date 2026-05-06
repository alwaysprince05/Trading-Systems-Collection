#!/usr/bin/env python3
"""
╔══════════════════════════════════════════════════════════════╗
║     MEAN REVERSION TRADING SYSTEM — RSI + BOLLINGER BANDS    ║
║                   creator: Prince Maurya                     ║
╚══════════════════════════════════════════════════════════════╝
Auto-installs all dependencies. Just run: python mean_reversion_trader.py
"""

import subprocess, sys

REQUIRED = ["yfinance", "pandas", "numpy", "matplotlib", "ta"]

def install(pkg):
    subprocess.check_call([sys.executable, "-m", "pip", "install", pkg, "-q"])

print("Checking & installing dependencies...")
for pkg in REQUIRED:
    try:
        __import__(pkg if pkg != "ta" else "ta")
    except ImportError:
        print(f"   Installing {pkg}...")
        install(pkg)
print(" All dependencies ready!\n")

import yfinance as yf
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import matplotlib.patches as mpatches
from matplotlib.lines import Line2D
import ta
import warnings
warnings.filterwarnings("ignore")

# ─── DARK THEME ───────────────────────────────────────────────
plt.style.use("dark_background")
DARK_BG    = "#0d0f14"
PANEL_BG   = "#13161f"
ACCENT1    = "#00ffcc"   # cyan-mint
ACCENT2    = "#ff4d6d"   # red-pink
ACCENT3    = "#ffe066"   # gold
ACCENT4    = "#7b61ff"   # purple
GRID_COL   = "#1e2230"
TEXT_COL   = "#c8d0e0"

# ─── CONFIG ───────────────────────────────────────────────────
TICKER      = "AAPL"
PERIOD      = "6mo"
BB_WINDOW   = 20
BB_STD      = 2.0
RSI_WINDOW  = 14
RSI_OB      = 70   # overbought
RSI_OS      = 30   # oversold
CAPITAL     = 10_000

# ─── DATA ─────────────────────────────────────────────────────
print(f"📡 Fetching {TICKER} data ({PERIOD})...")
df = yf.download(TICKER, period=PERIOD, auto_adjust=True, progress=False)
df.columns = [c[0] if isinstance(c, tuple) else c for c in df.columns]
df = df[["Open", "High", "Low", "Close", "Volume"]].dropna()

# ─── INDICATORS ───────────────────────────────────────────────
bb = ta.volatility.BollingerBands(df["Close"], window=BB_WINDOW, window_dev=BB_STD)
df["BB_upper"] = bb.bollinger_hband()
df["BB_lower"] = bb.bollinger_lband()
df["BB_mid"]   = bb.bollinger_mavg()
df["BB_pct"]   = bb.bollinger_pband()

df["RSI"] = ta.momentum.RSIIndicator(df["Close"], window=RSI_WINDOW).rsi()
df["Vol_MA"] = df["Volume"].rolling(20).mean()
df.dropna(inplace=True)

# ─── SIGNALS ──────────────────────────────────────────────────
df["Signal"] = 0
df.loc[(df["Close"] < df["BB_lower"]) & (df["RSI"] < RSI_OS), "Signal"] = 1   # BUY
df.loc[(df["Close"] > df["BB_upper"]) & (df["RSI"] > RSI_OB), "Signal"] = -1  # SELL

# ─── BACKTEST ─────────────────────────────────────────────────
position = 0
cash     = CAPITAL
equity   = []
trades   = []
buy_signals, sell_signals = [], []

for i, (idx, row) in enumerate(df.iterrows()):
    price = row["Close"]
    sig   = row["Signal"]

    if sig == 1 and position == 0:
        shares = cash // price
        if shares > 0:
            position = shares
            cash -= shares * price
            buy_signals.append((idx, price))
            trades.append({"date": idx, "type": "BUY", "price": price, "shares": shares})

    elif sig == -1 and position > 0:
        cash += position * price
        sell_signals.append((idx, price))
        trades.append({"date": idx, "type": "SELL", "price": price, "shares": position})
        position = 0

    equity.append(cash + position * price)

df["Equity"] = equity

# Buy & hold baseline
bh_shares  = CAPITAL / df["Close"].iloc[0]
df["BH_Equity"] = bh_shares * df["Close"]

# ─── METRICS ──────────────────────────────────────────────────
final_val   = df["Equity"].iloc[-1]
total_ret   = (final_val - CAPITAL) / CAPITAL * 100
bh_ret      = (df["BH_Equity"].iloc[-1] - CAPITAL) / CAPITAL * 100
daily_ret   = df["Equity"].pct_change().dropna()
sharpe      = (daily_ret.mean() / daily_ret.std()) * np.sqrt(252) if daily_ret.std() > 0 else 0
roll_max    = df["Equity"].cummax()
drawdown    = (df["Equity"] - roll_max) / roll_max * 100
max_dd      = drawdown.min()
n_trades    = len([t for t in trades if t["type"] == "BUY"])

print(f"\n{'='*52}")
print(f"  BACKTEST RESULTS — {TICKER} Mean Reversion")
print(f"{'='*52}")
print(f"  Starting Capital : ${CAPITAL:,.0f}")
print(f"  Final Value      : ${final_val:,.2f}")
print(f"  Strategy Return  : {total_ret:+.2f}%")
print(f"  Buy & Hold       : {bh_ret:+.2f}%")
print(f"  Sharpe Ratio     : {sharpe:.3f}")
print(f"  Max Drawdown     : {max_dd:.2f}%")
print(f"  Total Trades     : {n_trades}")
print(f"{'='*52}\n")

# ─── PLOT ─────────────────────────────────────────────────────
fig = plt.figure(figsize=(18, 13), facecolor=DARK_BG)
fig.suptitle(
    f"  MEAN REVERSION SYSTEM  ·  {TICKER}  ·  RSI + Bollinger Bands",
    fontsize=16, fontweight="bold", color=ACCENT1,
    fontfamily="monospace", y=0.97
)

gs = gridspec.GridSpec(4, 1, figure=fig, hspace=0.06,
                       top=0.93, bottom=0.05, left=0.06, right=0.97,
                       height_ratios=[3, 1.2, 1.2, 1.5])

axes = [fig.add_subplot(gs[i]) for i in range(4)]
for ax in axes:
    ax.set_facecolor(PANEL_BG)
    ax.tick_params(colors=TEXT_COL, labelsize=8)
    ax.spines[:].set_color(GRID_COL)
    ax.grid(color=GRID_COL, linewidth=0.4, alpha=0.7)

# Panel 0 — Price + Bollinger Bands
ax0 = axes[0]
ax0.plot(df.index, df["Close"],   color=TEXT_COL,  lw=1.2, label="Close")
ax0.plot(df.index, df["BB_upper"],color=ACCENT4,   lw=0.8, ls="--", alpha=0.8, label="BB Upper")
ax0.plot(df.index, df["BB_lower"],color=ACCENT4,   lw=0.8, ls="--", alpha=0.8, label="BB Lower")
ax0.plot(df.index, df["BB_mid"],  color=ACCENT3,   lw=0.6, ls=":",  alpha=0.6, label="BB Mid")
ax0.fill_between(df.index, df["BB_upper"], df["BB_lower"], alpha=0.06, color=ACCENT4)

if buy_signals:
    bx, by = zip(*buy_signals)
    ax0.scatter(bx, by, marker="^", color=ACCENT1, s=80, zorder=5, label="BUY")
if sell_signals:
    sx, sy = zip(*sell_signals)
    ax0.scatter(sx, sy, marker="v", color=ACCENT2, s=80, zorder=5, label="SELL")

ax0.set_ylabel("Price (USD)", color=TEXT_COL, fontsize=9)
ax0.legend(loc="upper left", fontsize=7.5, facecolor=PANEL_BG, edgecolor=GRID_COL, labelcolor=TEXT_COL)
ax0.tick_params(labelbottom=False)

# Panel 1 — RSI
ax1 = axes[1]
ax1.plot(df.index, df["RSI"], color=ACCENT3, lw=1.1, label="RSI")
ax1.axhline(RSI_OB, color=ACCENT2, lw=0.7, ls="--", alpha=0.8)
ax1.axhline(RSI_OS, color=ACCENT1, lw=0.7, ls="--", alpha=0.8)
ax1.fill_between(df.index, df["RSI"], RSI_OB, where=df["RSI"] > RSI_OB, alpha=0.25, color=ACCENT2)
ax1.fill_between(df.index, df["RSI"], RSI_OS, where=df["RSI"] < RSI_OS, alpha=0.25, color=ACCENT1)
ax1.set_ylim(0, 100)
ax1.set_ylabel("RSI", color=TEXT_COL, fontsize=9)
ax1.tick_params(labelbottom=False)

# Panel 2 — Volume
ax2 = axes[2]
colors = [ACCENT1 if c >= o else ACCENT2 for c, o in zip(df["Close"], df["Open"])]
ax2.bar(df.index, df["Volume"], color=colors, alpha=0.7, width=1)
ax2.plot(df.index, df["Vol_MA"], color=ACCENT3, lw=0.9, label="Vol MA20")
ax2.set_ylabel("Volume", color=TEXT_COL, fontsize=9)
ax2.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: f"{x/1e6:.1f}M"))
ax2.tick_params(labelbottom=False)

# Panel 3 — Equity Curve
ax3 = axes[3]
ax3.plot(df.index, df["Equity"],    color=ACCENT1, lw=1.5, label=f"Strategy  {total_ret:+.1f}%")
ax3.plot(df.index, df["BH_Equity"], color=ACCENT3, lw=1.0, ls="--", label=f"Buy & Hold {bh_ret:+.1f}%", alpha=0.8)
ax3.fill_between(df.index, df["Equity"], CAPITAL, where=df["Equity"] >= CAPITAL, alpha=0.15, color=ACCENT1)
ax3.fill_between(df.index, df["Equity"], CAPITAL, where=df["Equity"] < CAPITAL,  alpha=0.15, color=ACCENT2)
ax3.axhline(CAPITAL, color=TEXT_COL, lw=0.5, ls=":", alpha=0.5)
ax3.set_ylabel("Portfolio ($)", color=TEXT_COL, fontsize=9)
ax3.legend(loc="upper left", fontsize=7.5, facecolor=PANEL_BG, edgecolor=GRID_COL, labelcolor=TEXT_COL)

# Stats box
stats_text = (
    f"Sharpe: {sharpe:.2f}   MaxDD: {max_dd:.1f}%   "
    f"Trades: {n_trades}   Final: ${final_val:,.0f}"
)
fig.text(0.5, 0.01, stats_text, ha="center", fontsize=9,
         color=ACCENT3, fontfamily="monospace",
         bbox=dict(facecolor=PANEL_BG, edgecolor=GRID_COL, boxstyle="round,pad=0.4"))

plt.savefig("mean_reversion_result.png", dpi=150, bbox_inches="tight", facecolor=DARK_BG)
print("Chart saved → mean_reversion_result.png")
plt.show()
