"""
Nifty50 Intraday Backtesting Dashboard
=======================================
Streamlit UI for backtesting ORB and EMA Pullback strategies
on Nifty50 5-minute candle data (no volume required).

Supports uploading 1-minute CSV data (auto-resampled to 5-min).

Run:  streamlit run app.py
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from strategies import backtest, compute_metrics
from data_generator import generate_intraday_data, resample_1min_to_5min

# ── Page config ──────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Nifty50 Intraday Backtester",
    page_icon="📊",
    layout="wide",
)

st.title("Nifty50 Intraday Strategy Backtester")


# ── Sidebar: Data & Strategy Selection ───────────────────────────────────────
st.sidebar.header("Data Source")

data_source = st.sidebar.radio(
    "Choose data source",
    ["Upload CSV File", "Generate Sample Data"],
    index=0,
)

df = None

if data_source == "Upload CSV File":
    st.sidebar.markdown(
        """
        **Expected CSV columns:**
        `datetime, open, high, low, close`

        - Supports **1-minute** or **5-minute** candles
        - 1-min data is auto-resampled to 5-min
        - Volume column is optional (not used)
        """
    )
    uploaded = st.sidebar.file_uploader("Upload OHLC CSV", type=["csv"])
    if uploaded is not None:
        df = pd.read_csv(uploaded)
        # Normalize column names to lowercase
        df.columns = [c.strip().lower() for c in df.columns]

        # Auto-detect datetime column
        dt_col = None
        for candidate in ["datetime", "date_time", "timestamp", "time", "date"]:
            if candidate in df.columns:
                dt_col = candidate
                break

        if dt_col is None:
            st.error("Could not find a datetime column. Please ensure your CSV has a 'datetime' or 'date' column.")
            st.stop()

        df["datetime"] = pd.to_datetime(df[dt_col], dayfirst=True, errors="coerce")
        df = df.dropna(subset=["datetime"])
        df = df.sort_values("datetime").reset_index(drop=True)

        # Ensure OHLC columns exist
        required = ["open", "high", "low", "close"]
        missing = [c for c in required if c not in df.columns]
        if missing:
            st.error(f"Missing columns: {missing}. Please check your CSV.")
            st.stop()

        for c in required:
            df[c] = pd.to_numeric(df[c], errors="coerce")
        df = df.dropna(subset=required)

        # Auto-detect interval: check median gap between rows on first day
        df["date"] = df["datetime"].dt.strftime("%Y-%m-%d")
        df["time"] = df["datetime"].dt.strftime("%H:%M")
        first_day = df[df["date"] == df["date"].iloc[0]]
        if len(first_day) > 2:
            gaps = first_day["datetime"].diff().dt.total_seconds().dropna()
            median_gap = gaps.median()
        else:
            median_gap = 300  # assume 5-min

        if median_gap < 120:
            # 1-minute data → resample to 5-min
            st.sidebar.info("Detected 1-min data — resampling to 5-min candles...")
            df = resample_1min_to_5min(df)
            st.sidebar.success(
                f"Resampled to {len(df):,} candles  |  {df['date'].nunique()} trading days"
            )
        else:
            st.sidebar.success(f"Loaded {len(df):,} rows  |  {df['date'].nunique()} trading days")
    else:
        st.info("Upload a CSV file with 1-min or 5-min Nifty50 OHLC data to begin backtesting, or switch to **Generate Sample Data**.")
        st.stop()

else:
    st.sidebar.markdown("Generates realistic synthetic 5-min Nifty50 OHLC data for testing.")
    sample_start = st.sidebar.date_input("Sample start", value=pd.to_datetime("2024-01-01"))
    sample_end = st.sidebar.date_input("Sample end", value=pd.to_datetime("2024-12-31"))
    if st.sidebar.button("Generate Data"):
        df = generate_intraday_data(
            start_date=str(sample_start),
            end_date=str(sample_end),
        )
        st.session_state["generated_df"] = df
    if "generated_df" in st.session_state:
        df = st.session_state["generated_df"]
    else:
        st.info("Click **Generate Data** to create sample data, or upload a real CSV.")
        st.stop()


# ── Sidebar: Backtest Parameters ─────────────────────────────────────────────
st.sidebar.markdown("---")
st.sidebar.header("Strategy Settings")

strategy = st.sidebar.selectbox("Strategy", ["ORB", "EMA Pullback"])

available_dates = sorted(df["date"].unique())
col_d1, col_d2 = st.sidebar.columns(2)
start_date = col_d1.selectbox("Start Date", available_dates, index=0)
end_date = col_d2.selectbox("End Date", available_dates, index=len(available_dates) - 1)

if strategy == "ORB":
    st.sidebar.subheader("ORB Parameters")
    risk_reward = st.sidebar.slider("Risk:Reward Ratio", 1.0, 3.0, 1.5, 0.1)
    exit_time = st.sidebar.selectbox(
        "Max Exit Time",
        ["11:00", "11:30", "12:00", "12:30", "13:00"],
        index=2,
    )
    strategy_kwargs = dict(risk_reward=risk_reward, exit_time=exit_time)
else:
    st.sidebar.subheader("EMA Pullback Parameters")
    ema_period = st.sidebar.slider("EMA Period", 5, 50, 20, 1)
    risk_reward = st.sidebar.slider("Risk:Reward Ratio", 1.0, 3.0, 1.5, 0.1)
    exit_time = st.sidebar.selectbox(
        "Max Exit Time",
        ["13:00", "13:30", "14:00", "14:30", "15:00"],
        index=3,
    )
    strategy_kwargs = dict(ema_period=ema_period, risk_reward=risk_reward, exit_time=exit_time)


# ── Run Backtest ──────────────────────────────────────────────────────────────
run_btn = st.sidebar.button("Run Backtest", type="primary", use_container_width=True)

if run_btn:
    trades_df = backtest(
        df,
        strategy=strategy,
        start_date=start_date,
        end_date=end_date,
        **strategy_kwargs,
    )
    st.session_state["trades_df"] = trades_df
    st.session_state["strategy_name"] = strategy

if "trades_df" not in st.session_state:
    st.info("Configure parameters in the sidebar and click **Run Backtest**.")
    st.stop()

trades_df = st.session_state["trades_df"]
strategy_name = st.session_state.get("strategy_name", strategy)


# ── Results ───────────────────────────────────────────────────────────────────
st.header(f"{strategy_name} Backtest Results")

if trades_df.empty:
    st.warning("No trades generated for the selected period and parameters. Try adjusting filters.")
    st.stop()

metrics = compute_metrics(trades_df)

# ── KPI cards ─────────────────────────────────────────────────────────────────
k1, k2, k3, k4, k5, k6 = st.columns(6)
k1.metric("Total Trades", metrics["total_trades"])
k2.metric("Win Rate", f"{metrics['win_rate']}%")
k3.metric("Total P&L (pts)", f"{metrics['total_pnl']:+.1f}")
k4.metric("Profit Factor", f"{metrics['profit_factor']:.2f}")
k5.metric("Max Drawdown", f"{metrics['max_drawdown']:.1f}")
k6.metric("Avg R:R", f"{metrics['avg_risk_reward']:.2f}")

# ── Detailed metrics ──────────────────────────────────────────────────────────
st.subheader("Detailed Metrics")
m1, m2, m3 = st.columns(3)

with m1:
    st.markdown("**Win/Loss Breakdown**")
    st.write(f"- Winners: {metrics['winners']}")
    st.write(f"- Losers: {metrics['losers']}")
    st.write(f"- Avg Winner: {metrics['avg_winner']:+.1f} pts")
    st.write(f"- Avg Loser: {metrics['avg_loser']:+.1f} pts")

with m2:
    st.markdown("**P&L Stats**")
    st.write(f"- Max Win: {metrics['max_win']:+.1f} pts")
    st.write(f"- Max Loss: {metrics['max_loss']:+.1f} pts")
    st.write(f"- Avg P&L: {metrics['avg_pnl']:+.1f} pts")
    st.write(f"- Total P&L: {metrics['total_pnl']:+.1f} pts")

with m3:
    st.markdown("**Exit Analysis**")
    st.write(f"- Target Hits: {metrics['target_hits']}")
    st.write(f"- SL Hits: {metrics['sl_hits']}")
    st.write(f"- Time/EOD Exits: {metrics['time_exits']}")


# ── Equity Curve ──────────────────────────────────────────────────────────────
st.subheader("Equity Curve")
equity = trades_df["pnl_points"].cumsum()
fig_eq = go.Figure()
fig_eq.add_trace(go.Scatter(
    x=trades_df["date"],
    y=equity,
    mode="lines+markers",
    name="Cumulative P&L",
    line=dict(color="#1f77b4", width=2),
    marker=dict(size=4),
))
fig_eq.update_layout(
    xaxis_title="Date",
    yaxis_title="Cumulative P&L (points)",
    height=400,
    margin=dict(l=40, r=20, t=30, b=40),
)
st.plotly_chart(fig_eq, use_container_width=True)


# ── P&L Distribution ─────────────────────────────────────────────────────────
st.subheader("P&L Distribution")
col_hist, col_pie = st.columns(2)

with col_hist:
    fig_hist = go.Figure()
    colors = ["#2ecc71" if x > 0 else "#e74c3c" for x in trades_df["pnl_points"]]
    fig_hist.add_trace(go.Bar(
        x=trades_df["date"],
        y=trades_df["pnl_points"],
        marker_color=colors,
        name="Daily P&L",
    ))
    fig_hist.update_layout(
        xaxis_title="Date",
        yaxis_title="P&L (points)",
        height=350,
        margin=dict(l=40, r=20, t=30, b=40),
    )
    st.plotly_chart(fig_hist, use_container_width=True)

with col_pie:
    fig_pie = go.Figure()
    exit_counts = trades_df["exit_reason"].value_counts()
    fig_pie.add_trace(go.Pie(
        labels=exit_counts.index,
        values=exit_counts.values,
        marker=dict(colors=["#2ecc71", "#e74c3c", "#f39c12", "#3498db"]),
    ))
    fig_pie.update_layout(
        title="Exit Reasons",
        height=350,
        margin=dict(l=20, r=20, t=50, b=20),
    )
    st.plotly_chart(fig_pie, use_container_width=True)


# ── Drawdown Chart ────────────────────────────────────────────────────────────
st.subheader("Drawdown")
cumulative = trades_df["pnl_points"].cumsum()
running_max = cumulative.cummax()
dd = cumulative - running_max

fig_dd = go.Figure()
fig_dd.add_trace(go.Scatter(
    x=trades_df["date"],
    y=dd,
    fill="tozeroy",
    mode="lines",
    name="Drawdown",
    line=dict(color="#e74c3c", width=1),
    fillcolor="rgba(231,76,60,0.2)",
))
fig_dd.update_layout(
    xaxis_title="Date",
    yaxis_title="Drawdown (points)",
    height=300,
    margin=dict(l=40, r=20, t=30, b=40),
)
st.plotly_chart(fig_dd, use_container_width=True)


# ── Trade Log ─────────────────────────────────────────────────────────────────
st.subheader("Trade Log")

def highlight_pnl(val):
    if val > 0:
        return "color: #2ecc71; font-weight: bold"
    elif val < 0:
        return "color: #e74c3c; font-weight: bold"
    return ""

display_cols = [c for c in trades_df.columns if c != "reward_ratio"]
styled = trades_df[display_cols].style.map(highlight_pnl, subset=["pnl_points"])
st.dataframe(styled, use_container_width=True, height=400)

# ── Download ──────────────────────────────────────────────────────────────────
csv_data = trades_df.to_csv(index=False)
st.download_button(
    label="Download Trade Log CSV",
    data=csv_data,
    file_name=f"{strategy_name}_backtest_results.csv",
    mime="text/csv",
)
