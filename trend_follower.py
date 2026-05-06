#!/usr/bin/env python3
"""
╔══════════════════════════════════════════════════════════════╗
║        TREND FOLLOWING SYSTEM — MACD + EMA CROSSOVER         ║
║                   creator: Prince Maurya                     ║
╚══════════════════════════════════════════════════════════════╝
Auto-installs all dependencies. Just run: python trend_follower.py
"""

import subprocess, sys

REQUIRED = ["yfinance", "pandas", "numpy", "matplotlib", "ta"]

def install(pkg):
    subprocess.check_call([sys.executable, "-m", "pip", "install", pkg, "-q"])

print(" Checking & installing dependencies...")
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
import ta
import warnings
warnings.filterwarnings("ignore")

# ─── DARK THEME ───────────────────────────────────────────────
plt.style.use("dark_background")
BG       = "#090b10"
PANEL    = "#0f1117"
GREEN    = "#39ff14"   # neon green
RED      = "#ff3355"   # neon red
BLUE     = "#00b4ff"   # electric blue
GOLD     = "#ffd700"
PURPLE   = "#bf5fff"
GRID     = "#1a1d28"
TEXT     = "#aab4c8"

# ─── CONFIG ───────────────────────────────────────────────────
TICKER     = "SPY"
PERIOD     = "1y"
EMA_FAST   = 12
EMA_SLOW   = 26
EMA_SIGNAL = 9
EMA_TREND  = 200     # long-term trend filter
ATR_PERIOD = 14
CAPITAL    = 10_000

# ─── DATA ─────────────────────────────────────────────────────
print(f"📡 Fetching {TICKER} data ({PERIOD})...")
df = yf.download(TICKER, period=PERIOD, auto_adjust=True, progress=False)
df.columns = [c[0] if isinstance(c, tuple) else c for c in df.columns]
df = df[["Open","High","Low","Close","Volume"]].dropna()

# ─── INDICATORS ───────────────────────────────────────────────
macd_ind       = ta.trend.MACD(df["Close"], EMA_FAST, EMA_SLOW, EMA_SIGNAL)
df["MACD"]     = macd_ind.macd()
df["MACD_sig"] = macd_ind.macd_signal()
df["MACD_hist"]= macd_ind.macd_diff()

df["EMA_fast"] = ta.trend.EMAIndicator(df["Close"], EMA_FAST).ema_indicator()
df["EMA_slow"] = ta.trend.EMAIndicator(df["Close"], EMA_SLOW).ema_indicator()
df["EMA_trend"]= ta.trend.EMAIndicator(df["Close"], EMA_TREND).ema_indicator()

df["ATR"] = ta.volatility.AverageTrueRange(df["High"], df["Low"], df["Close"], ATR_PERIOD).average_true_range()
df["Vol_MA"] = df["Volume"].rolling(20).mean()

df.dropna(inplace=True)

# ─── SIGNALS ──────────────────────────────────────────────────
# MACD crosses + price above/below EMA200 trend filter
df["Signal"] = 0
prev_macd     = df["MACD"].shift(1)
prev_sig      = df["MACD_sig"].shift(1)

bull_cross = (df["MACD"] > df["MACD_sig"]) & (prev_macd <= prev_sig)
bear_cross = (df["MACD"] < df["MACD_sig"]) & (prev_macd >= prev_sig)

df.loc[bull_cross & (df["Close"] > df["EMA_trend"]), "Signal"] = 1
df.loc[bear_cross & (df["Close"] < df["EMA_trend"]), "Signal"] = -1

# ─── BACKTEST ─────────────────────────────────────────────────
position      = 0
cash          = CAPITAL
equity        = []
trades        = []
buy_pts, sell_pts = [], []
entry_price   = 0

for idx, row in df.iterrows():
    price = row["Close"]
    sig   = row["Signal"]

    if sig == 1 and position == 0:
        shares    = cash // price
        if shares > 0:
            position    = shares
            entry_price = price
            cash       -= shares * price
            buy_pts.append((idx, price))
            trades.append({"date": idx, "action": "BUY", "price": price,
                           "shares": shares, "pnl": None})

    elif (sig == -1 or (position > 0 and price < entry_price - 2 * row["ATR"])) and position > 0:
        pnl  = (price - entry_price) * position
        cash += position * price
        sell_pts.append((idx, price))
        trades.append({"date": idx, "action": "SELL", "price": price,
                       "shares": position, "pnl": pnl})
        position    = 0
        entry_price = 0

    equity.append(cash + position * price)

df["Equity"] = equity
df["BH"]     = (CAPITAL / df["Close"].iloc[0]) * df["Close"]

# ─── METRICS ──────────────────────────────────────────────────
final_val  = df["Equity"].iloc[-1]
strat_ret  = (final_val - CAPITAL) / CAPITAL * 100
bh_ret     = (df["BH"].iloc[-1] - CAPITAL) / CAPITAL * 100
dr         = df["Equity"].pct_change().dropna()
sharpe     = (dr.mean() / dr.std()) * np.sqrt(252) if dr.std() > 0 else 0
max_dd     = ((df["Equity"] - df["Equity"].cummax()) / df["Equity"].cummax() * 100).min()
win_trades = [t for t in trades if t["action"] == "SELL" and t["pnl"] and t["pnl"] > 0]
tot_sells  = [t for t in trades if t["action"] == "SELL" and t["pnl"] is not None]
win_rate   = len(win_trades) / len(tot_sells) * 100 if tot_sells else 0

print(f"\n{'='*52}")
print(f"  BACKTEST RESULTS — {TICKER} Trend Following")
print(f"{'='*52}")
print(f"  Starting Capital : ${CAPITAL:,.0f}")
print(f"  Final Value      : ${final_val:,.2f}")
print(f"  Strategy Return  : {strat_ret:+.2f}%")
print(f"  Buy & Hold       : {bh_ret:+.2f}%")
print(f"  Sharpe Ratio     : {sharpe:.3f}")
print(f"  Max Drawdown     : {max_dd:.2f}%")
print(f"  Win Rate         : {win_rate:.1f}%")
print(f"  Total Trades     : {len(buy_pts)}")
print(f"{'='*52}\n")

# ─── PLOT ─────────────────────────────────────────────────────
fig = plt.figure(figsize=(18, 14), facecolor=BG)
fig.suptitle(
    f"  TREND FOLLOWING SYSTEM  ·  {TICKER}  ·  MACD + EMA Crossover  ·  ATR Stop-Loss",
    fontsize=15, fontweight="bold", color=GREEN,
    fontfamily="monospace", y=0.97
)

gs   = gridspec.GridSpec(4, 2, figure=fig, hspace=0.06, wspace=0.25,
                          top=0.93, bottom=0.06, left=0.06, right=0.97,
                          height_ratios=[3, 1.3, 1.3, 1.6])

ax0  = fig.add_subplot(gs[0, :])    # price full width
ax1  = fig.add_subplot(gs[1, :])    # MACD full width
ax2  = fig.add_subplot(gs[2, :])    # volume full width
ax3  = fig.add_subplot(gs[3, 0])    # equity curve
ax4  = fig.add_subplot(gs[3, 1])    # drawdown

for ax in [ax0, ax1, ax2, ax3, ax4]:
    ax.set_facecolor(PANEL)
    ax.tick_params(colors=TEXT, labelsize=8)
    ax.spines[:].set_color(GRID)
    ax.grid(color=GRID, linewidth=0.4, alpha=0.7)

# Panel 0 — Price + EMAs
ax0.plot(df.index, df["Close"],     color=TEXT,   lw=1.0, alpha=0.9, label="Close")
ax0.plot(df.index, df["EMA_fast"],  color=BLUE,   lw=0.9, ls="--",   label=f"EMA {EMA_FAST}")
ax0.plot(df.index, df["EMA_slow"],  color=PURPLE, lw=0.9, ls="--",   label=f"EMA {EMA_SLOW}")
ax0.plot(df.index, df["EMA_trend"], color=GOLD,   lw=1.1,             label=f"EMA {EMA_TREND}")
ax0.fill_between(df.index, df["EMA_fast"], df["EMA_slow"],
                 where=df["EMA_fast"] > df["EMA_slow"], alpha=0.12, color=GREEN)
ax0.fill_between(df.index, df["EMA_fast"], df["EMA_slow"],
                 where=df["EMA_fast"] <= df["EMA_slow"], alpha=0.12, color=RED)

if buy_pts:
    bx, by = zip(*buy_pts)
    ax0.scatter(bx, by, marker="^", color=GREEN, s=90, zorder=6, label="BUY", edgecolors="white", lw=0.4)
if sell_pts:
    sx, sy = zip(*sell_pts)
    ax0.scatter(sx, sy, marker="v", color=RED, s=90, zorder=6, label="SELL", edgecolors="white", lw=0.4)

ax0.set_ylabel("Price (USD)", color=TEXT, fontsize=9)
ax0.legend(loc="upper left", fontsize=7.5, facecolor=PANEL, edgecolor=GRID, labelcolor=TEXT, ncol=3)
ax0.tick_params(labelbottom=False)

# Panel 1 — MACD
ax1.plot(df.index, df["MACD"],     color=BLUE,   lw=1.1, label="MACD")
ax1.plot(df.index, df["MACD_sig"], color=RED,    lw=1.1, label="Signal")
hist_colors = [GREEN if v >= 0 else RED for v in df["MACD_hist"]]
ax1.bar(df.index, df["MACD_hist"], color=hist_colors, alpha=0.6, width=1, label="Histogram")
ax1.axhline(0, color=TEXT, lw=0.5, alpha=0.4)
ax1.set_ylabel("MACD", color=TEXT, fontsize=9)
ax1.legend(loc="upper left", fontsize=7.5, facecolor=PANEL, edgecolor=GRID, labelcolor=TEXT)
ax1.tick_params(labelbottom=False)

# Panel 2 — Volume
vol_cols = [GREEN if c >= o else RED for c, o in zip(df["Close"], df["Open"])]
ax2.bar(df.index, df["Volume"], color=vol_cols, alpha=0.6, width=1)
ax2.plot(df.index, df["Vol_MA"], color=GOLD, lw=0.9, label="Vol MA20")
ax2.set_ylabel("Volume", color=TEXT, fontsize=9)
ax2.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: f"{x/1e6:.0f}M"))
ax2.tick_params(labelbottom=False)

# Panel 3 — Equity
ax3.plot(df.index, df["Equity"], color=GREEN, lw=1.5, label=f"Strategy {strat_ret:+.1f}%")
ax3.plot(df.index, df["BH"],     color=GOLD,  lw=1.0, ls="--", label=f"B&H {bh_ret:+.1f}%")
ax3.fill_between(df.index, df["Equity"], df["BH"],
                 where=df["Equity"] >= df["BH"], alpha=0.12, color=GREEN)
ax3.fill_between(df.index, df["Equity"], df["BH"],
                 where=df["Equity"] < df["BH"],  alpha=0.12, color=RED)
ax3.axhline(CAPITAL, color=TEXT, lw=0.5, ls=":", alpha=0.4)
ax3.set_ylabel("Portfolio ($)", color=TEXT, fontsize=9)
ax3.legend(fontsize=7.5, facecolor=PANEL, edgecolor=GRID, labelcolor=TEXT)

# Panel 4 — Drawdown
dd = (df["Equity"] - df["Equity"].cummax()) / df["Equity"].cummax() * 100
ax4.fill_between(df.index, dd, 0, alpha=0.5, color=RED)
ax4.plot(df.index, dd, color=RED, lw=0.8)
ax4.set_ylabel("Drawdown (%)", color=TEXT, fontsize=9)
ax4.axhline(max_dd, color=GOLD, lw=0.7, ls="--", alpha=0.8,
            label=f"Max DD: {max_dd:.1f}%")
ax4.legend(fontsize=7.5, facecolor=PANEL, edgecolor=GRID, labelcolor=TEXT)

stats = (f"Sharpe: {sharpe:.2f}   MaxDD: {max_dd:.1f}%   "
         f"Win Rate: {win_rate:.0f}%   Trades: {len(buy_pts)}   Final: ${final_val:,.0f}")
fig.text(0.5, 0.015, stats, ha="center", fontsize=9, color=GOLD,
         fontfamily="monospace",
         bbox=dict(facecolor=PANEL, edgecolor=GRID, boxstyle="round,pad=0.4"))

plt.savefig("trend_follower_result.png", dpi=150, bbox_inches="tight", facecolor=BG)
print("Chart saved → trend_follower_result.png")
plt.show()
