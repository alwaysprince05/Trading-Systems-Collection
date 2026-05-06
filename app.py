import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import ta
import warnings
import os
from datetime import datetime

warnings.filterwarnings("ignore")

# --- CONFIG ---
st.set_page_config(
    page_title="Prince Maurya | Quant Terminal",
    page_icon="⚡",
    layout="wide",
)

# --- THEME & CSS ---
# Using a more robust CSS injection method
CYBER_CSS = """
<link href="https://fonts.googleapis.com/css2?family=Outfit:wght@300;400;600;700&display=swap" rel="stylesheet">
<style>
    * { font-family: 'Outfit', sans-serif; }
    .main {
        background: #0d1117;
        color: #e0e0e0;
    }
    .stApp { background: transparent; }
    .glass-card {
        background: rgba(255, 255, 255, 0.03);
        backdrop-filter: blur(10px);
        border-radius: 15px;
        border: 1px solid rgba(255, 255, 255, 0.1);
        padding: 25px;
        margin-bottom: 25px;
    }
    .metric-container {
        display: flex;
        justify-content: space-between;
        gap: 15px;
        margin-bottom: 25px;
    }
    .metric-card {
        flex: 1;
        background: rgba(255, 255, 255, 0.05);
        padding: 20px;
        border-radius: 12px;
        border-left: 5px solid #00ffcc;
        box-shadow: 0 4px 15px rgba(0,0,0,0.3);
    }
    .metric-label { font-size: 0.8rem; color: #888; text-transform: uppercase; letter-spacing: 1px; }
    .metric-value { font-size: 1.7rem; font-weight: 700; color: #fff; margin-top: 5px; }
    .metric-delta { font-size: 0.9rem; margin-top: 5px; }
    .pos { color: #00ff9d; }
    .neg { color: #ff3355; }
    
    .stButton>button {
        width: 100%;
        background: linear-gradient(135deg, #00ffcc 0%, #00b4ff 100%);
        color: #0d0f14;
        font-weight: 700;
        border: none;
        border-radius: 10px;
        padding: 15px;
        transition: 0.3s;
    }
    .stButton>button:hover {
        transform: translateY(-2px);
        box-shadow: 0 5px 20px rgba(0, 255, 204, 0.4);
    }
</style>
"""
st.markdown(CYBER_CSS, unsafe_allow_html=True)

# --- UTILS ---
def get_data(ticker, period):
    try:
        df = yf.download(ticker, period=period, auto_adjust=True, progress=False)
        if df.empty: return None
        if isinstance(df.columns, pd.MultiIndex): df.columns = [c[0] for c in df.columns]
        return df[["Open", "High", "Low", "Close", "Volume"]].dropna()
    except: return None

def custom_metric(label, value, delta=None):
    d_class = "pos" if delta and "+" in str(delta) else "neg"
    st.markdown(f"""
        <div class="metric-card">
            <div class="metric-label">{label}</div>
            <div class="metric-value">{value}</div>
            <div class="metric-delta {d_class}">{delta if delta else ""}</div>
        </div>
    """, unsafe_allow_html=True)

# --- CORE LOGIC ---
def run_backtest(df_input, system, params=None):
    capital = 10000; pos = 0; cash = capital; equity = []
    buy_signals, sell_signals = [], []
    
    if system == "Momentum Portfolio":
        TICKERS = ["XLK", "XLF", "XLE", "XLV", "XLI", "XLC", "XLY", "XLP", "XLRE", "XLB", "XLU", "GLD"]
        raw = yf.download(TICKERS + ["SPY"], period=params.get('period', '2y'), auto_adjust=True, progress=False)["Close"]
        if isinstance(raw.columns, pd.MultiIndex): raw.columns = [c[0] for c in raw.columns]
        raw.ffill(inplace=True); raw.dropna(inplace=True)
        bench = raw["SPY"]; assets = raw[TICKERS]
        ret_s = assets.pct_change(21); ret_l = assets.pct_change(63); vol = assets.pct_change().rolling(21).std() * np.sqrt(252)
        scores = (0.6 * ret_s + 0.4 * ret_l) / (vol + 1e-9)
        e = capital; records = []; prev_p = []; dates = assets.index[63:]
        for i, date in enumerate(dates):
            if i % 21 == 0:
                s = scores.loc[date].dropna(); s = s[s > 0]
                top = s.nlargest(4).index.tolist(); prev_p = top if top else prev_p
            p = prev_p
            if i > 0:
                p_date = dates[i-1]
                r = sum((1.0/len(p)) * ((assets.loc[date, t] / assets.loc[p_date, t]) - 1) for t in p) if p else 0
                e *= (1 + r)
            records.append({"date": date, "equity": e})
        df = pd.DataFrame(records).set_index("date")
        df["BH"] = capital * (bench.loc[dates] / bench.loc[dates[0]])
        return df, [], []

    if df_input is None or df_input.empty:
        return None, [], []
        
    df = df_input.copy()

    if system == "Trend Following":
        macd = ta.trend.MACD(df['Close'])
        df['MACD'] = macd.macd()
        df['MACD_sig'] = macd.macd_signal()
        df['MACD_hist'] = macd.macd_diff()
        df['EMA_trend'] = ta.trend.EMAIndicator(df['Close'], params.get('trend', 200)).ema_indicator()
        df.dropna(inplace=True)
        
        if df.empty: return None, [], []
        
        # Vectorized signals for performance
        df['Cross'] = 0
        df.loc[(df['MACD'] > df['MACD_sig']) & (df['MACD'].shift(1) <= df['MACD_sig'].shift(1)), 'Cross'] = 1
        df.loc[(df['MACD'] < df['MACD_sig']) & (df['MACD'].shift(1) >= df['MACD_sig'].shift(1)), 'Cross'] = -1
        
        for idx, row in df.iterrows():
            if row['Cross'] == 1 and row['Close'] > row['EMA_trend'] and pos == 0:
                pos = cash // row['Close']; cash -= pos * row['Close']; buy_signals.append((idx, row['Close']))
            elif row['Cross'] == -1 and pos > 0:
                cash += pos * row['Close']; sell_signals.append((idx, row['Close'])); pos = 0
            equity.append(cash + pos * row['Close'])
            
    elif system == "Mean Reversion":
        bb = ta.volatility.BollingerBands(df['Close'], window=params.get('bb_window', 20), window_dev=params.get('bb_std', 2.0))
        df['BB_L'] = bb.bollinger_lband(); df['BB_U'] = bb.bollinger_hband()
        df['RSI'] = ta.momentum.RSIIndicator(df['Close']).rsi()
        df.dropna(inplace=True)
        
        if df.empty: return None, [], []
        
        for idx, row in df.iterrows():
            if row['Close'] < row['BB_L'] and row['RSI'] < 30 and pos == 0:
                pos = cash // row['Close']; cash -= pos * row['Close']; buy_signals.append((idx, row['Close']))
            elif row['Close'] > row['BB_U'] and row['RSI'] > 70 and pos > 0:
                cash += pos * row['Close']; sell_signals.append((idx, row['Close'])); pos = 0
            equity.append(cash + pos * row['Close'])
            
    df['Equity'] = equity
    df['BH'] = (capital / df['Close'].iloc[0]) * df['Close']
    return df, buy_signals, sell_signals

def plot_interactive(df, ticker, system, buy_pts, sell_pts):
    fig = make_subplots(rows=3, cols=1, shared_xaxes=True, vertical_spacing=0.05, 
                        row_heights=[0.5, 0.2, 0.3], subplot_titles=("Price Action", "Indicators", "Capital Growth"))
    
    # Trace 1: Price
    fig.add_trace(go.Scatter(x=df.index, y=df['Close'], name='Price', line=dict(color='#ffffff', width=1.5)), row=1, col=1)
    if buy_pts:
        bx, by = zip(*buy_pts)
        fig.add_trace(go.Scatter(x=bx, y=by, mode='markers', name='BUY', marker=dict(symbol='triangle-up', size=12, color='#00ffcc')), row=1, col=1)
    if sell_pts:
        sx, sy = zip(*sell_pts)
        fig.add_trace(go.Scatter(x=sx, y=sy, mode='markers', name='SELL', marker=dict(symbol='triangle-down', size=12, color='#ff3355')), row=1, col=1)
    
    # Trace 2: Indicators
    if 'MACD' in df.columns:
        fig.add_trace(go.Scatter(x=df.index, y=df['MACD'], name='MACD', line=dict(color='#00ffcc')), row=2, col=1)
        fig.add_trace(go.Scatter(x=df.index, y=df['MACD_sig'], name='Signal', line=dict(color='#ff3355')), row=2, col=1)
    elif 'RSI' in df.columns:
        fig.add_trace(go.Scatter(x=df.index, y=df['RSI'], name='RSI', line=dict(color='#00ffcc')), row=2, col=1)
        fig.add_hline(y=70, line_dash="dash", line_color="#ff3355", row=2, col=1)
        fig.add_hline(y=30, line_dash="dash", line_color="#00ffcc", row=2, col=1)
        
    # Trace 3: Capital
    fig.add_trace(go.Scatter(x=df.index, y=df['Equity'], name='Strategy', fill='tozeroy', line=dict(color='#00ffcc', width=2)), row=3, col=1)
    fig.add_trace(go.Scatter(x=df.index, y=df['BH'], name='Benchmark', line=dict(color='rgba(255,255,255,0.3)', dash='dot')), row=3, col=1)
    
    fig.update_layout(height=800, template='plotly_dark', paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)', 
                      margin=dict(l=20, r=20, t=60, b=20), hovermode='x unified')
    return fig

# --- SIDEBAR ---
with st.sidebar:
    st.markdown('<h1 style="color:#00ffcc; text-align:center;">QUANT LAB</h1>', unsafe_allow_html=True)
    st.markdown('<p style="text-align:center; color:#888;">Architect: Prince Maurya</p>', unsafe_allow_html=True)
    st.markdown("---")
    engine = st.selectbox("🎯 Core Engine", ["Trend Following", "Mean Reversion", "Momentum Portfolio"])
    ticker = st.text_input("📈 Asset Ticker", value="SPY").upper()
    period = st.selectbox("📅 Lookback", ["1y", "2y", "5y", "max"], index=1)
    st.markdown("---")
    execute = st.button("🚀 EXECUTE SYSTEM")

# --- MAIN ---
st.title("PROPRIETARY QUANT TERMINAL")
st.markdown('<p style="color:#00ffcc; font-size:1.2rem;">Advanced Algorithmic Research & Execution</p>', unsafe_allow_html=True)

if execute:
    with st.spinner("⚡ DECODING MARKET DATA..."):
        df_raw = get_data(ticker, period)
        if df_raw is not None or engine == "Momentum Portfolio":
            df, b_pts, s_pts = run_backtest(df_raw, engine, {"period": period})
            
            if df is not None and not df.empty:
                # Metrics
                final_val = df['Equity'].iloc[-1] if 'Equity' in df.columns else df['equity'].iloc[-1]
                ret = (final_val - 10000)/100
                bh_v = df['BH'].iloc[-1]
                bh_ret = (bh_v - 10000)/100
                
                st.markdown('<div class="metric-container">', unsafe_allow_html=True)
                col1, col2, col3, col4 = st.columns(4)
                with col1: custom_metric("Portfolio Return", f"{ret:+.2f}%", "Strategic Yield")
                with col2: custom_metric("Alpha Generation", f"{(ret-bh_ret):+.2f}%", "vs Benchmark")
                with col3: 
                    mdd = (((df['Equity'] if 'Equity' in df.columns else df['equity']) - (df['Equity'] if 'Equity' in df.columns else df['equity']).cummax()) / (df['Equity'] if 'Equity' in df.columns else df['equity']).cummax() * 100).min()
                    custom_metric("Peak Risk", f"{mdd:.2f}%", "Max Drawdown")
                with col4: custom_metric("Final Equity", f"${final_val:,.0f}", "Net Value")
                st.markdown('</div>', unsafe_allow_html=True)
                
                # Plot
                if engine != "Momentum Portfolio":
                    st.markdown('<div class="glass-card">', unsafe_allow_html=True)
                    st.plotly_chart(plot_interactive(df, ticker, engine, b_pts, s_pts), use_container_width=True)
                    st.markdown('</div>', unsafe_allow_html=True)
                else:
                    # Specialized Plotly for Portfolio
                    fig = go.Figure()
                    fig.add_trace(go.Scatter(x=df.index, y=df['equity'], name='Portfolio', fill='tozeroy', line=dict(color='#00ffcc')))
                    fig.add_trace(go.Scatter(x=df.index, y=df['BH'], name='SPY Benchmark', line=dict(color='rgba(255,255,255,0.3)', dash='dot')))
                    fig.update_layout(height=500, template='plotly_dark', paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)')
                    st.markdown('<div class="glass-card">', unsafe_allow_html=True)
                    st.plotly_chart(fig, use_container_width=True)
                    st.markdown('</div>', unsafe_allow_html=True)
                
                with st.expander("📝 ANALYTICS LOGS"):
                    st.dataframe(df.tail(20), use_container_width=True)
            else:
                st.error("NOT ENOUGH DATA: Lookback period is too short for indicator calculation (e.g., 200 EMA needs >200 days).")
        else:
            st.error("FAILED TO FETCH MARKET DATA. TICKER MAY BE INVALID.")
else:
    st.markdown("""
        <div class="glass-card" style="text-align:center; padding: 80px 40px;">
            <h2 style="color:#00ffcc;">SYSTEM READY</h2>
            <p style="color:#888;">Select an algorithmic core and click <b>Execute System</b> to begin simulation.</p>
        </div>
    """, unsafe_allow_html=True)

st.divider()
st.markdown('<p style="text-align:center; color:#444; font-size:0.8rem;">© 2026 PRINCE MAURYA QUANT LABS. ALL RIGHTS RESERVED.</p>', unsafe_allow_html=True)
