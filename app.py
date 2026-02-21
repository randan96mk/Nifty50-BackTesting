"""
Nifty50 Intraday Backtesting Dashboard
=======================================
Streamlit UI for backtesting 6 intraday strategies on Nifty50 5-minute
candle data (no volume required).

Strategies: ORB, EMA Pullback, PDH/PDL Breakout, MACD Crossover,
            Supertrend, Inside Bar Breakout.

Supports uploading 1-minute CSV data (auto-resampled to 5-min).

Run:  streamlit run app.py
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from strategies import backtest, compute_metrics, STRATEGY_LIST
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

strategy = st.sidebar.selectbox("Strategy", STRATEGY_LIST)

available_dates = sorted(df["date"].unique())
min_dt = pd.to_datetime(available_dates[0]).date()
max_dt = pd.to_datetime(available_dates[-1]).date()
col_d1, col_d2 = st.sidebar.columns(2)
start_date = str(col_d1.date_input("Start Date", value=min_dt, min_value=min_dt, max_value=max_dt))
end_date = str(col_d2.date_input("End Date", value=max_dt, min_value=min_dt, max_value=max_dt))

# Strategy-specific parameter panels
if strategy == "ORB":
    st.sidebar.subheader("ORB Parameters")
    risk_reward = st.sidebar.slider("Risk:Reward Ratio", 1.0, 3.0, 1.5, 0.1)
    exit_time = st.sidebar.selectbox(
        "Max Exit Time",
        ["11:00", "11:30", "12:00", "12:30", "13:00"],
        index=2,
    )
    strategy_kwargs = dict(risk_reward=risk_reward, exit_time=exit_time)

elif strategy == "EMA Pullback":
    st.sidebar.subheader("EMA Pullback Parameters")
    ema_period = st.sidebar.slider("EMA Period", 5, 50, 20, 1)
    risk_reward = st.sidebar.slider("Risk:Reward Ratio", 1.0, 3.0, 1.5, 0.1)
    exit_time = st.sidebar.selectbox(
        "Max Exit Time",
        ["13:00", "13:30", "14:00", "14:30", "15:00"],
        index=3,
    )
    strategy_kwargs = dict(ema_period=ema_period, risk_reward=risk_reward, exit_time=exit_time)

elif strategy == "PDH/PDL Breakout":
    st.sidebar.subheader("PDH/PDL Parameters")
    buffer_points = st.sidebar.slider("Buffer Points", 0.0, 20.0, 5.0, 1.0)
    risk_reward = st.sidebar.slider("Risk:Reward Ratio", 1.0, 3.0, 1.5, 0.1)
    exit_time = st.sidebar.selectbox(
        "Max Exit Time",
        ["12:00", "12:30", "13:00", "13:30", "14:00", "14:30"],
        index=4,
    )
    strategy_kwargs = dict(buffer_points=buffer_points, risk_reward=risk_reward, exit_time=exit_time)

elif strategy == "MACD Crossover":
    st.sidebar.subheader("MACD Parameters")
    fast_period = st.sidebar.slider("Fast EMA Period", 5, 20, 12, 1)
    slow_period = st.sidebar.slider("Slow EMA Period", 15, 40, 26, 1)
    signal_period = st.sidebar.slider("Signal Period", 3, 15, 9, 1)
    risk_reward = st.sidebar.slider("Risk:Reward Ratio", 1.0, 3.0, 1.5, 0.1)
    exit_time = st.sidebar.selectbox(
        "Max Exit Time",
        ["13:00", "13:30", "14:00", "14:30", "15:00"],
        index=3,
    )
    strategy_kwargs = dict(
        fast_period=fast_period, slow_period=slow_period,
        signal_period=signal_period, risk_reward=risk_reward, exit_time=exit_time,
    )

elif strategy == "Supertrend":
    st.sidebar.subheader("Supertrend Parameters")
    atr_period = st.sidebar.slider("ATR Period", 5, 20, 10, 1)
    multiplier = st.sidebar.slider("Multiplier", 1.0, 5.0, 3.0, 0.5)
    risk_reward = st.sidebar.slider("Risk:Reward Ratio", 1.0, 3.0, 1.5, 0.1)
    exit_time = st.sidebar.selectbox(
        "Max Exit Time",
        ["13:00", "13:30", "14:00", "14:30", "15:00"],
        index=3,
    )
    strategy_kwargs = dict(
        atr_period=atr_period, multiplier=multiplier,
        risk_reward=risk_reward, exit_time=exit_time,
    )

elif strategy == "Inside Bar":
    st.sidebar.subheader("Inside Bar Parameters")
    min_mother_range = st.sidebar.slider("Min Mother Bar Range (pts)", 5.0, 50.0, 10.0, 1.0)
    risk_reward = st.sidebar.slider("Risk:Reward Ratio", 1.0, 3.0, 1.5, 0.1)
    exit_time = st.sidebar.selectbox(
        "Max Exit Time",
        ["12:00", "12:30", "13:00", "13:30", "14:00", "14:30"],
        index=4,
    )
    strategy_kwargs = dict(min_mother_range=min_mother_range, risk_reward=risk_reward, exit_time=exit_time)


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
    # Store full filtered data for charting
    filtered = df[(df["date"] >= start_date) & (df["date"] <= end_date)].copy()
    st.session_state["chart_df"] = filtered

if "trades_df" not in st.session_state:
    st.info("Configure parameters in the sidebar and click **Run Backtest**.")
    st.stop()

trades_df = st.session_state["trades_df"]
strategy_name = st.session_state.get("strategy_name", strategy)
chart_df = st.session_state.get("chart_df", df)


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


# ── Candlestick Chart with Buy/Sell Signals ──────────────────────────────────
st.subheader("Candlestick Chart with Trade Signals")

trade_dates = sorted(trades_df["date"].unique().tolist())

if len(trade_dates) == 0:
    st.info("No trades to display on chart.")
else:
    # Date picker for candlestick chart
    trade_dates_dt = [pd.to_datetime(d).date() for d in trade_dates]
    selected_chart_dt = st.date_input(
        "Select trading day to view",
        value=trade_dates_dt[0],
        min_value=trade_dates_dt[0],
        max_value=trade_dates_dt[-1],
        key="chart_date_input",
    )
    # Snap to nearest trade date if user picks a non-trade day
    selected_chart_date = str(selected_chart_dt)
    if selected_chart_date not in trade_dates:
        # Find closest trade date
        closest = min(trade_dates, key=lambda d: abs(pd.to_datetime(d).date() - selected_chart_dt))
        selected_chart_date = closest

    # Get candle data for the selected date
    day_candles = chart_df[chart_df["date"] == selected_chart_date].copy()
    day_trade = trades_df[trades_df["date"] == selected_chart_date]

    if not day_candles.empty:
        # Build datetime for x-axis
        day_candles["dt"] = pd.to_datetime(
            day_candles["date"] + " " + day_candles["time"]
        )

        fig_candle = make_subplots(
            rows=1, cols=1,
            shared_xaxes=True,
        )

        # Candlestick trace
        fig_candle.add_trace(
            go.Candlestick(
                x=day_candles["dt"],
                open=day_candles["open"],
                high=day_candles["high"],
                low=day_candles["low"],
                close=day_candles["close"],
                increasing_line_color="#2ecc71",
                decreasing_line_color="#e74c3c",
                increasing_fillcolor="#2ecc71",
                decreasing_fillcolor="#e74c3c",
                name="Price",
            ),
            row=1, col=1,
        )

        # Plot trade signals if trade exists for this day
        if not day_trade.empty:
            trade = day_trade.iloc[0]
            entry_dt = pd.to_datetime(f"{selected_chart_date} {trade['entry_time']}")
            exit_dt = pd.to_datetime(f"{selected_chart_date} {trade['exit_time']}")
            is_long = trade["direction"] in ("CE", "LONG")

            # BUY signal marker
            buy_color = "#2ecc71" if is_long else "#e74c3c"
            buy_symbol = "triangle-up" if is_long else "triangle-down"
            buy_label = "BUY (Long)" if is_long else "SELL (Short)"
            fig_candle.add_trace(
                go.Scatter(
                    x=[entry_dt],
                    y=[trade["entry_price"]],
                    mode="markers+text",
                    marker=dict(symbol=buy_symbol, size=16, color=buy_color, line=dict(width=2, color="white")),
                    text=[buy_label],
                    textposition="top center" if is_long else "bottom center",
                    textfont=dict(size=11, color=buy_color, family="Arial Black"),
                    name="Entry",
                    showlegend=True,
                )
            )

            # EXIT signal marker
            exit_color = "#e74c3c" if is_long else "#2ecc71"
            exit_symbol = "triangle-down" if is_long else "triangle-up"
            exit_label = f"EXIT ({trade['exit_reason']})"
            fig_candle.add_trace(
                go.Scatter(
                    x=[exit_dt],
                    y=[trade["exit_price"]],
                    mode="markers+text",
                    marker=dict(symbol=exit_symbol, size=16, color=exit_color, line=dict(width=2, color="white")),
                    text=[exit_label],
                    textposition="bottom center" if is_long else "top center",
                    textfont=dict(size=11, color=exit_color, family="Arial Black"),
                    name="Exit",
                    showlegend=True,
                )
            )

            # SL line
            fig_candle.add_hline(
                y=trade["sl"],
                line_dash="dash",
                line_color="#e74c3c",
                line_width=1,
                annotation_text=f"SL: {trade['sl']:.1f}",
                annotation_position="right",
                annotation_font_color="#e74c3c",
            )

            # Target line
            fig_candle.add_hline(
                y=trade["target"],
                line_dash="dash",
                line_color="#2ecc71",
                line_width=1,
                annotation_text=f"Target: {trade['target']:.1f}",
                annotation_position="right",
                annotation_font_color="#2ecc71",
            )

            # Entry price line
            fig_candle.add_hline(
                y=trade["entry_price"],
                line_dash="dot",
                line_color="#3498db",
                line_width=1,
                annotation_text=f"Entry: {trade['entry_price']:.1f}",
                annotation_position="left",
                annotation_font_color="#3498db",
            )

            # P&L annotation
            pnl_color = "#2ecc71" if trade["pnl_points"] > 0 else "#e74c3c"
            pnl_text = f"P&L: {trade['pnl_points']:+.1f} pts  |  {trade['exit_reason']}"

        # Strategy-specific overlays
        if strategy_name == "ORB" and not day_trade.empty:
            trade = day_trade.iloc[0]
            if "or_high" in trade and "or_low" in trade:
                or_start = pd.to_datetime(f"{selected_chart_date} 09:15")
                or_end = pd.to_datetime(f"{selected_chart_date} 09:30")
                # OR High shading
                fig_candle.add_shape(
                    type="rect",
                    x0=or_start, x1=day_candles["dt"].iloc[-1],
                    y0=trade["or_low"], y1=trade["or_high"],
                    fillcolor="rgba(52, 152, 219, 0.1)",
                    line=dict(width=0),
                )
                fig_candle.add_hline(
                    y=trade["or_high"], line_dash="dot", line_color="#f39c12", line_width=1,
                    annotation_text=f"OR High: {trade['or_high']:.1f}",
                    annotation_position="left", annotation_font_color="#f39c12",
                )
                fig_candle.add_hline(
                    y=trade["or_low"], line_dash="dot", line_color="#f39c12", line_width=1,
                    annotation_text=f"OR Low: {trade['or_low']:.1f}",
                    annotation_position="left", annotation_font_color="#f39c12",
                )

        elif strategy_name == "PDH/PDL Breakout" and not day_trade.empty:
            trade = day_trade.iloc[0]
            if "pdh" in trade and "pdl" in trade:
                fig_candle.add_hline(
                    y=trade["pdh"], line_dash="dashdot", line_color="#9b59b6", line_width=1,
                    annotation_text=f"PDH: {trade['pdh']:.1f}",
                    annotation_position="left", annotation_font_color="#9b59b6",
                )
                fig_candle.add_hline(
                    y=trade["pdl"], line_dash="dashdot", line_color="#9b59b6", line_width=1,
                    annotation_text=f"PDL: {trade['pdl']:.1f}",
                    annotation_position="left", annotation_font_color="#9b59b6",
                )

        elif strategy_name == "Inside Bar" and not day_trade.empty:
            trade = day_trade.iloc[0]
            if "mother_high" in trade and "mother_low" in trade:
                fig_candle.add_hline(
                    y=trade["mother_high"], line_dash="dashdot", line_color="#f39c12", line_width=1,
                    annotation_text=f"Mother High: {trade['mother_high']:.1f}",
                    annotation_position="left", annotation_font_color="#f39c12",
                )
                fig_candle.add_hline(
                    y=trade["mother_low"], line_dash="dashdot", line_color="#f39c12", line_width=1,
                    annotation_text=f"Mother Low: {trade['mother_low']:.1f}",
                    annotation_position="left", annotation_font_color="#f39c12",
                )

        # Trade info box
        if not day_trade.empty:
            trade = day_trade.iloc[0]
            pnl_color = "#2ecc71" if trade["pnl_points"] > 0 else "#e74c3c"
            fig_candle.add_annotation(
                x=0.5, y=1.12,
                xref="paper", yref="paper",
                text=(
                    f"<b>{trade['direction']}</b> | "
                    f"Entry: {trade['entry_price']:.1f} @ {trade['entry_time']} | "
                    f"Exit: {trade['exit_price']:.1f} @ {trade['exit_time']} | "
                    f"<span style='color:{pnl_color}'><b>P&L: {trade['pnl_points']:+.1f} pts</b></span> | "
                    f"{trade['exit_reason']}"
                ),
                showarrow=False,
                font=dict(size=13),
                align="center",
            )

        fig_candle.update_layout(
            title=f"{strategy_name} — {selected_chart_date}",
            yaxis_title="Price",
            xaxis_title="Time",
            height=550,
            margin=dict(l=50, r=50, t=80, b=40),
            xaxis_rangeslider_visible=False,
            legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
            plot_bgcolor="#1a1a2e",
            paper_bgcolor="#16213e",
            font=dict(color="#e0e0e0"),
            xaxis=dict(gridcolor="#2a2a4a"),
            yaxis=dict(gridcolor="#2a2a4a"),
        )

        st.plotly_chart(fig_candle, use_container_width=True)

        # Navigation buttons for previous/next trade
        nav1, nav2, nav3 = st.columns([1, 2, 1])
        current_idx = trade_dates.index(selected_chart_date) if selected_chart_date in trade_dates else 0
        with nav1:
            if current_idx > 0:
                if st.button("< Previous Trade"):
                    st.session_state["chart_date_input"] = pd.to_datetime(trade_dates[current_idx - 1]).date()
                    st.rerun()
        with nav3:
            if current_idx < len(trade_dates) - 1:
                if st.button("Next Trade >"):
                    st.session_state["chart_date_input"] = pd.to_datetime(trade_dates[current_idx + 1]).date()
                    st.rerun()


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
