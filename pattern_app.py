"""
Nifty 50 Chart Pattern Backtesting Dashboard
=============================================
Streamlit UI for detecting and backtesting breakout patterns on intraday
(3-min / 5-min) Nifty 50 OHLC data.  No volume data required.

Patterns: Triple Top, Triple Bottom, Ascending / Descending / Symmetrical
          Triangle, Bull Flag, Bear Flag, Rectangle.

Run:  streamlit run pattern_app.py
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from pattern_detection import (
    detect_triple_tops,
    detect_triple_bottoms,
    detect_triangles,
    detect_flags_rectangles,
    backtest_patterns,
    compute_pattern_metrics,
    run_all_patterns,
)
from data_generator import generate_intraday_data, resample_1min_to_5min

# ── Page config ──────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Nifty 50 Intraday Pattern Backtester",
    page_icon="📈",
    layout="wide",
)

st.title("Nifty 50 — Intraday Pattern Breakout Backtester")
st.caption(
    "Detects Triple Tops/Bottoms, Triangles, and Flags/Rectangles "
    "on 3-min / 5-min OHLC candles for intraday entry & exit"
)


# ── Sidebar: Data Source ─────────────────────────────────────────────────────
st.sidebar.header("Data Source")
data_source = st.sidebar.radio(
    "Choose data source",
    ["Upload CSV File", "Generate Sample Data"],
    index=1,
)

df = None
timeframe_label = "5 min"  # default

if data_source == "Upload CSV File":
    st.sidebar.markdown(
        "**Expected CSV columns:** `datetime, open, high, low, close`\n\n"
        "Minute-level or intraday candles.\n"
        "Volume column is optional (not used)."
    )

    tf_choice = st.sidebar.selectbox(
        "Candle timeframe",
        ["1 min (resample to 3 min)", "1 min (resample to 5 min)",
         "3 min (use as-is)", "5 min (use as-is)"],
        index=1,
    )

    uploaded = st.sidebar.file_uploader("Upload Intraday OHLC CSV", type=["csv"])
    if uploaded is not None:
        df = pd.read_csv(uploaded)
        df.columns = [c.strip().lower() for c in df.columns]

        dt_col = None
        for candidate in ["datetime", "date_time", "date", "timestamp", "time"]:
            if candidate in df.columns:
                dt_col = candidate
                break
        if dt_col is None:
            st.error("CSV must have a 'datetime' or 'date' column.")
            st.stop()

        # Robust datetime parsing
        raw_dt = df[dt_col].astype(str)
        try:
            parsed = pd.to_datetime(raw_dt, format="ISO8601")
        except Exception:
            try:
                parsed = pd.to_datetime(raw_dt, format="mixed", dayfirst=False)
            except Exception:
                parsed = pd.to_datetime(raw_dt, dayfirst=True, errors="coerce")

        df["datetime"] = parsed
        df = df.dropna(subset=["datetime"]).sort_values("datetime").reset_index(drop=True)

        required = ["open", "high", "low", "close"]
        missing = [c for c in required if c not in df.columns]
        if missing:
            st.error(f"Missing columns: {missing}")
            st.stop()
        for c in required:
            df[c] = pd.to_numeric(df[c], errors="coerce")
        df = df.dropna(subset=required)

        # Resample if needed
        if tf_choice.startswith("1 min (resample to 3"):
            timeframe_label = "3 min"
            df = df.set_index("datetime")
            df = (
                df.resample("3min")
                .agg({"open": "first", "high": "max", "low": "min", "close": "last"})
                .dropna(subset=["open"])
                .reset_index()
            )
        elif tf_choice.startswith("1 min (resample to 5"):
            timeframe_label = "5 min"
            df = df.set_index("datetime")
            df = (
                df.resample("5min")
                .agg({"open": "first", "high": "max", "low": "min", "close": "last"})
                .dropna(subset=["open"])
                .reset_index()
            )
        elif "3 min" in tf_choice:
            timeframe_label = "3 min"
        else:
            timeframe_label = "5 min"

        # Add date/time columns for display
        df["date"] = df["datetime"].dt.strftime("%Y-%m-%d")
        df["time"] = df["datetime"].dt.strftime("%H:%M")
        df["dt_label"] = df["datetime"].dt.strftime("%Y-%m-%d %H:%M")

        # Filter market hours (09:15 – 15:30)
        df = df[(df["time"] >= "09:15") & (df["time"] <= "15:30")].reset_index(drop=True)

        st.sidebar.success(
            f"Loaded {len(df):,} candles ({timeframe_label})  \n"
            f"{df['date'].iloc[0]} → {df['date'].iloc[-1]}  \n"
            f"Trading days: {df['date'].nunique()}"
        )
    else:
        st.info("Upload an intraday OHLC CSV or switch to **Generate Sample Data**.")
        st.stop()
else:
    st.sidebar.markdown("Generates synthetic 5-min Nifty 50 intraday data.")
    col_s, col_e = st.sidebar.columns(2)
    sample_start = col_s.date_input("Start", value=pd.to_datetime("2024-01-01"))
    sample_end = col_e.date_input("End", value=pd.to_datetime("2024-12-31"))
    if st.sidebar.button("Generate Data", type="primary"):
        df = generate_intraday_data(
            start_date=str(sample_start), end_date=str(sample_end))
        df["dt_label"] = pd.to_datetime(df["datetime"]).dt.strftime("%Y-%m-%d %H:%M")
        st.session_state["intraday_df"] = df
        timeframe_label = "5 min"
    if "intraday_df" in st.session_state:
        df = st.session_state["intraday_df"]
        timeframe_label = "5 min"
    else:
        st.info("Click **Generate Data** to create sample Nifty 50 intraday data.")
        st.stop()


# ── Sidebar: Backtest Date Range ─────────────────────────────────────────────
st.sidebar.markdown("---")
st.sidebar.header("Backtest Date Range")

# Parse dates available in the loaded data
_all_dates = sorted(df["date"].unique())
_data_start = pd.to_datetime(_all_dates[0]).date()
_data_end = pd.to_datetime(_all_dates[-1]).date()

from datetime import timedelta
from dateutil.relativedelta import relativedelta

_range_preset = st.sidebar.selectbox(
    "Quick range",
    ["Last 1 Year", "Last 2 Years", "Last 3 Years", "YTD", "Full Data", "Custom"],
    index=4,
    key="range_preset",
)

if _range_preset == "Last 1 Year":
    _preset_start = _data_end - relativedelta(years=1)
    _preset_end = _data_end
elif _range_preset == "Last 2 Years":
    _preset_start = _data_end - relativedelta(years=2)
    _preset_end = _data_end
elif _range_preset == "Last 3 Years":
    _preset_start = _data_end - relativedelta(years=3)
    _preset_end = _data_end
elif _range_preset == "YTD":
    _preset_start = pd.to_datetime(f"{_data_end.year}-01-01").date()
    _preset_end = _data_end
elif _range_preset == "Full Data":
    _preset_start = _data_start
    _preset_end = _data_end
else:  # Custom
    _preset_start = _data_start
    _preset_end = _data_end

_col_ds, _col_de = st.sidebar.columns(2)
bt_start = _col_ds.date_input(
    "From", value=max(_preset_start, _data_start),
    min_value=_data_start, max_value=_data_end, key="bt_start",
)
bt_end = _col_de.date_input(
    "To", value=min(_preset_end, _data_end),
    min_value=_data_start, max_value=_data_end, key="bt_end",
)

# Apply the date filter
_bt_start_str = str(bt_start)
_bt_end_str = str(bt_end)
df = df[(df["date"] >= _bt_start_str) & (df["date"] <= _bt_end_str)].reset_index(drop=True)

if df.empty:
    st.warning("No data in the selected date range. Adjust the range.")
    st.stop()

_filtered_days = df["date"].nunique()
st.sidebar.caption(
    f"Backtesting: {df['date'].iloc[0]} to {df['date'].iloc[-1]}  \n"
    f"{len(df):,} candles across {_filtered_days} trading days"
)


# ── Sidebar: Pattern Selection ────────────────────────────────────────────────
st.sidebar.markdown("---")
st.sidebar.header("Pattern Selection")
st.sidebar.caption("Choose which patterns to detect and backtest")

_sel_col1, _sel_col2 = st.sidebar.columns(2)
with _sel_col1:
    pat_triple_top = st.checkbox("Triple Top", value=True, key="pat_tt")
    pat_triple_bot = st.checkbox("Triple Bottom", value=True, key="pat_tb")
    pat_asc_tri = st.checkbox("Ascending Triangle", value=True, key="pat_at")
with _sel_col2:
    pat_desc_tri = st.checkbox("Descending Triangle", value=True, key="pat_dt")
    pat_sym_tri = st.checkbox("Symmetrical Triangle", value=True, key="pat_st")
    pat_rectangle = st.checkbox("Rectangle", value=True, key="pat_rect")

_sel_col3, _sel_col4 = st.sidebar.columns(2)
with _sel_col3:
    pat_bull_flag = st.checkbox("Bull Flag", value=True, key="pat_bf")
with _sel_col4:
    pat_bear_flag = st.checkbox("Bear Flag", value=True, key="pat_brf")

enabled_patterns = set()
if pat_triple_top:
    enabled_patterns.add("Triple Top")
if pat_triple_bot:
    enabled_patterns.add("Triple Bottom")
if pat_asc_tri:
    enabled_patterns.add("Ascending Triangle")
if pat_desc_tri:
    enabled_patterns.add("Descending Triangle")
if pat_sym_tri:
    enabled_patterns.add("Symmetrical Triangle")
if pat_rectangle:
    enabled_patterns.add("Rectangle")
if pat_bull_flag:
    enabled_patterns.add("Bull Flag")
if pat_bear_flag:
    enabled_patterns.add("Bear Flag")

if not enabled_patterns:
    st.sidebar.warning("Select at least one pattern.")

# ── Sidebar: Pattern Parameters ─────────────────────────────────────────────
st.sidebar.markdown("---")
st.sidebar.header("Pattern Parameters")
st.sidebar.caption(f"Tuned for {timeframe_label} intraday candles")

with st.sidebar.expander("ATR & Noise Filter", expanded=True):
    use_atr = st.checkbox("Use ATR-adaptive thresholds",
                          value=True, key="use_atr",
                          help="Adapts tolerance, SL, and min height to "
                               "current volatility via ATR")
    atr_period = st.slider("ATR period (bars)", 7, 30, 14, 1, key="atr_period",
                           help="Lookback for Average True Range calculation")
    sl_atr_mult = st.slider("SL distance (ATR multiples)", 0.5, 3.0, 1.5, 0.5,
                            key="sl_atr",
                            help="Stop loss = resistance/support +/- N x ATR")
    zigzag_thresh = st.slider("ZigZag noise threshold (ATR mult)", 0.5, 3.0, 1.0, 0.5,
                              key="zz_thresh",
                              help="Min swing size to count as a pivot. "
                                   "Higher = fewer but cleaner peaks/troughs")

with st.sidebar.expander("Triple Top / Bottom", expanded=False):
    tt_peak_order = st.slider("Peak/Trough sensitivity (bars)", 3, 15, 5, 1,
                              key="tt_order",
                              help="Fallback when ATR mode is off. "
                                   "Higher = fewer, more significant peaks")
    tt_tolerance = st.slider("Price tolerance (%)", 0.1, 2.0, 0.3, 0.1,
                             key="tt_tol",
                             help="Fallback when ATR mode is off")
    tt_min_spacing = st.slider("Min spacing (bars)", 3, 20, 8, 1,
                               key="tt_sp")
    tt_max_pattern = st.slider("Max pattern span (bars)", 20, 120, 60, 5,
                               key="tt_max")

with st.sidebar.expander("Triangles", expanded=False):
    tri_peak_order = st.slider("Peak/Trough sensitivity (bars)", 2, 10, 3, 1,
                               key="tri_order")
    tri_min_window = st.slider("Min window (bars)", 10, 30, 15, 5,
                               key="tri_min")
    tri_max_window = st.slider("Max window (bars)", 30, 80, 50, 5,
                               key="tri_max")

with st.sidebar.expander("Flags / Rectangles", expanded=False):
    fl_pole_lookback = st.slider("Pole lookback (bars)", 3, 20, 8, 1,
                                 key="fl_pole")
    fl_pole_min = st.slider("Min pole move (%)", 0.3, 4.0, 0.8, 0.1,
                            key="fl_pmin")
    fl_consol_min = st.slider("Min consolidation (bars)", 3, 12, 5, 1,
                              key="fl_cmin")
    fl_consol_max = st.slider("Max consolidation (bars)", 8, 40, 20, 2,
                              key="fl_cmax")

max_hold = st.sidebar.slider("Max hold period (bars)", 5, 60, 20, 5,
                              help="Max bars to hold a position before time exit")

with st.sidebar.expander("Entry Time Window", expanded=False):
    st.caption("Filter out entries outside the best intraday window")
    earliest_entry = st.text_input("Earliest entry (HH:MM)", value="09:30",
                                    key="earliest_entry",
                                    help="No entries before this time (IST)")
    latest_entry = st.text_input("Latest entry (HH:MM)", value="14:00",
                                  key="latest_entry",
                                  help="No entries after this time — "
                                       "not enough runway for target")


# ── Run Backtest ─────────────────────────────────────────────────────────────
run_btn = st.sidebar.button("Run Pattern Backtest", type="primary",
                            use_container_width=True)

if run_btn:
    if not enabled_patterns:
        st.error("Select at least one pattern to backtest.")
        st.stop()
    trades_df, per_pattern = run_all_patterns(
        df,
        triple_top_params=dict(
            peak_order=tt_peak_order, tolerance_pct=tt_tolerance,
            min_spacing=tt_min_spacing, max_pattern_bars=tt_max_pattern),
        triple_bottom_params=dict(
            trough_order=tt_peak_order, tolerance_pct=tt_tolerance,
            min_spacing=tt_min_spacing, max_pattern_bars=tt_max_pattern),
        triangle_params=dict(
            peak_order=tri_peak_order,
            min_window=tri_min_window, max_window=tri_max_window),
        flag_rect_params=dict(
            pole_lookback=fl_pole_lookback, pole_min_pct=fl_pole_min,
            consol_min_bars=fl_consol_min, consol_max_bars=fl_consol_max),
        max_hold_bars=max_hold,
        use_atr=use_atr,
        atr_period=atr_period,
        sl_atr_mult=sl_atr_mult,
        zigzag_threshold=zigzag_thresh,
        earliest_entry=earliest_entry,
        latest_entry=latest_entry,
        enabled_patterns=enabled_patterns,
    )
    st.session_state["pat_trades"] = trades_df
    st.session_state["pat_metrics"] = per_pattern
    st.session_state["pat_df"] = df.copy()

if "pat_trades" not in st.session_state:
    st.info("Configure parameters and click **Run Pattern Backtest**.")
    st.stop()

trades_df = st.session_state["pat_trades"]
per_pattern = st.session_state["pat_metrics"]
chart_df = st.session_state["pat_df"]

if trades_df.empty:
    st.warning("No patterns detected. Try relaxing the parameters (lower peak "
               "sensitivity, wider windows, lower pole move threshold).")
    st.stop()


# ═══════════════════════════════════════════════════════════════════════════════
# Results
# ═══════════════════════════════════════════════════════════════════════════════

overall = compute_pattern_metrics(trades_df)

st.header("Overall Results")
k1, k2, k3, k4, k5, k6 = st.columns(6)
k1.metric("Total Trades", overall["total_trades"])
k2.metric("Win Rate", f"{overall['win_rate']}%")
k3.metric("Total P&L (pts)", f"{overall['total_pnl']:+.1f}")
k4.metric("Profit Factor", f"{overall['profit_factor']:.2f}")
k5.metric("Max Drawdown", f"{overall['max_drawdown']:.1f}")
k6.metric("Avg R:R", f"{overall['avg_risk_reward']:.2f}")


# ── Pattern Comparison Table ─────────────────────────────────────────────────
st.header("Pattern Comparison")

comparison_rows = []
for ptype, m in per_pattern.items():
    comparison_rows.append({
        "Pattern": ptype,
        "Trades": m["total_trades"],
        "Win Rate (%)": m["win_rate"],
        "Avg Return (%)": m["avg_return_pct"],
        "Total P&L (pts)": m["total_pnl"],
        "Profit Factor": m["profit_factor"],
        "Max Drawdown": m["max_drawdown"],
        "Avg R:R": m["avg_risk_reward"],
        "Avg Hold (bars)": m["avg_hold_bars"],
        "Target Hits": m["target_hits"],
        "SL Hits": m["sl_hits"],
    })

comp_df = pd.DataFrame(comparison_rows)
if not comp_df.empty:
    # Highlight best pattern
    best_idx = comp_df["Win Rate (%)"].idxmax()
    best_pattern = comp_df.loc[best_idx, "Pattern"]

    st.dataframe(
        comp_df.style.highlight_max(
            subset=["Win Rate (%)", "Avg Return (%)", "Total P&L (pts)",
                    "Profit Factor", "Avg R:R"],
            color="#1a472a",
        ).highlight_min(
            subset=["Max Drawdown"],
            color="#4a1a1a",
        ),
        use_container_width=True,
    )

    st.success(
        f"**Highest statistical edge: {best_pattern}** — "
        f"{comp_df.loc[best_idx, 'Win Rate (%)']}% win rate, "
        f"{comp_df.loc[best_idx, 'Avg Return (%)']}% avg return, "
        f"profit factor {comp_df.loc[best_idx, 'Profit Factor']}"
    )


# ── Price Chart with Detected Patterns ───────────────────────────────────────
st.header("Price Chart with Detected Patterns")

# Let user pick a date range to zoom into
all_dates = sorted(chart_df["date"].unique())
if len(all_dates) > 1:
    date_col1, date_col2 = st.columns(2)
    chart_start = date_col1.selectbox("Chart from", all_dates,
                                       index=max(0, len(all_dates) - 5))
    chart_end = date_col2.selectbox("Chart to", all_dates,
                                     index=len(all_dates) - 1)
    chart_mask = (chart_df["date"] >= chart_start) & (chart_df["date"] <= chart_end)
    chart_window = chart_df[chart_mask]
else:
    chart_window = chart_df

fig_main = go.Figure()

# Candlestick
fig_main.add_trace(go.Candlestick(
    x=chart_window["dt_label"],
    open=chart_window["open"],
    high=chart_window["high"],
    low=chart_window["low"],
    close=chart_window["close"],
    increasing_line_color="#2ecc71",
    decreasing_line_color="#e74c3c",
    increasing_fillcolor="#2ecc71",
    decreasing_fillcolor="#e74c3c",
    name="Price",
    opacity=0.7,
))

# Colour map for pattern types
color_map = {
    "Triple Top": "#e74c3c",
    "Triple Bottom": "#2ecc71",
    "Ascending Triangle": "#3498db",
    "Descending Triangle": "#e67e22",
    "Symmetrical Triangle": "#9b59b6",
    "Bull Flag": "#1abc9c",
    "Bear Flag": "#c0392b",
    "Rectangle": "#f39c12",
}

# Plot entry markers
for _, trade in trades_df.iterrows():
    color = color_map.get(trade["pattern"], "#ffffff")
    marker_sym = "triangle-up" if trade["direction"] == "LONG" else "triangle-down"
    fig_main.add_trace(go.Scatter(
        x=[trade["entry_dt"]],
        y=[trade["entry_price"]],
        mode="markers",
        marker=dict(symbol=marker_sym, size=10, color=color,
                    line=dict(width=1, color="white")),
        name=trade["pattern"],
        showlegend=False,
        hovertext=(
            f"{trade['pattern']}<br>"
            f"Entry: {trade['entry_price']:.1f} @ {trade['entry_dt']}<br>"
            f"Exit: {trade['exit_price']:.1f} ({trade['exit_reason']})<br>"
            f"P&L: {trade['pnl_points']:+.1f} pts"
        ),
        hoverinfo="text",
    ))

# Legend entries (one per pattern type)
for ptype, color in color_map.items():
    if ptype in trades_df["pattern"].values:
        fig_main.add_trace(go.Scatter(
            x=[None], y=[None], mode="markers",
            marker=dict(size=10, color=color),
            name=ptype,
        ))

fig_main.update_layout(
    height=600,
    xaxis_rangeslider_visible=False,
    legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
    plot_bgcolor="#1a1a2e",
    paper_bgcolor="#16213e",
    font=dict(color="#e0e0e0"),
    xaxis=dict(gridcolor="#2a2a4a"),
    yaxis=dict(gridcolor="#2a2a4a", title="Price"),
    margin=dict(l=50, r=50, t=50, b=40),
)
st.plotly_chart(fig_main, use_container_width=True)


# ── Individual Pattern Detail View ───────────────────────────────────────────
st.header("Pattern Detail View")

trade_list = trades_df.reset_index(drop=True)
options = [
    f"{row['pattern']} — {row['entry_dt']} — "
    f"{'▲' if row['direction'] == 'LONG' else '▼'} "
    f"{row['pnl_points']:+.1f} pts"
    for _, row in trade_list.iterrows()
]

selected_idx = st.selectbox("Select a pattern to inspect", range(len(options)),
                            format_func=lambda i: options[i])

trade = trade_list.iloc[selected_idx]

# Determine chart window
b_idx = int(trade["breakout_idx"])
pat_start_key = None
for key in ["peak1_idx", "trough1_idx", "window_start_idx", "pole_start_idx"]:
    if key in trade and pd.notna(trade.get(key)):
        pat_start_key = key
        break

if pat_start_key:
    start_idx = max(0, int(trade[pat_start_key]) - 10)
else:
    start_idx = max(0, b_idx - 40)

end_idx = min(len(chart_df) - 1, b_idx + int(trade["hold_bars"]) + 15)
window = chart_df.iloc[start_idx:end_idx + 1]

fig_detail = go.Figure()

fig_detail.add_trace(go.Candlestick(
    x=window["dt_label"],
    open=window["open"],
    high=window["high"],
    low=window["low"],
    close=window["close"],
    increasing_line_color="#2ecc71",
    decreasing_line_color="#e74c3c",
    increasing_fillcolor="#2ecc71",
    decreasing_fillcolor="#e74c3c",
    name="Price",
))

# Pattern-specific overlays
ptype = trade["pattern"]
pcolor = color_map.get(ptype, "#ffffff")

if ptype in ("Triple Top", "Triple Bottom"):
    if ptype == "Triple Top":
        res_level = trade.get("resistance", None)
        neck_level = trade.get("neckline", None)
        if res_level:
            fig_detail.add_hline(
                y=res_level, line_dash="dash", line_color="#e74c3c",
                annotation_text=f"Resistance: {res_level:.0f}",
                annotation_position="right", annotation_font_color="#e74c3c")
        if neck_level:
            fig_detail.add_hline(
                y=neck_level, line_dash="dash", line_color="#f39c12",
                annotation_text=f"Neckline: {neck_level:.0f}",
                annotation_position="right", annotation_font_color="#f39c12")
        for pi_key in ["peak1_idx", "peak2_idx", "peak3_idx"]:
            if pi_key in trade and pd.notna(trade[pi_key]):
                pi = int(trade[pi_key])
                if 0 <= pi < len(chart_df):
                    fig_detail.add_trace(go.Scatter(
                        x=[chart_df.iloc[pi]["dt_label"]],
                        y=[chart_df.iloc[pi]["high"]],
                        mode="markers",
                        marker=dict(symbol="diamond", size=12, color="#e74c3c"),
                        showlegend=False))
    else:
        sup_level = trade.get("support", None)
        neck_level = trade.get("neckline", None)
        if sup_level:
            fig_detail.add_hline(
                y=sup_level, line_dash="dash", line_color="#2ecc71",
                annotation_text=f"Support: {sup_level:.0f}",
                annotation_position="right", annotation_font_color="#2ecc71")
        if neck_level:
            fig_detail.add_hline(
                y=neck_level, line_dash="dash", line_color="#f39c12",
                annotation_text=f"Neckline: {neck_level:.0f}",
                annotation_position="right", annotation_font_color="#f39c12")
        for ti_key in ["trough1_idx", "trough2_idx", "trough3_idx"]:
            if ti_key in trade and pd.notna(trade[ti_key]):
                ti = int(trade[ti_key])
                if 0 <= ti < len(chart_df):
                    fig_detail.add_trace(go.Scatter(
                        x=[chart_df.iloc[ti]["dt_label"]],
                        y=[chart_df.iloc[ti]["low"]],
                        mode="markers",
                        marker=dict(symbol="diamond", size=12, color="#2ecc71"),
                        showlegend=False))

elif "Triangle" in ptype:
    u_slope = trade.get("upper_slope", 0)
    u_int = trade.get("upper_intercept", 0)
    l_slope = trade.get("lower_slope", 0)
    l_int = trade.get("lower_intercept", 0)
    ws = trade.get("window_start_idx", start_idx)
    we = trade.get("window_end_idx", b_idx)
    if u_slope and u_int and l_slope and l_int:
        ws, we = int(ws), int(we)
        line_x = [chart_df.iloc[ws]["dt_label"],
                  chart_df.iloc[min(we, len(chart_df) - 1)]["dt_label"]]
        upper_y = [u_int + u_slope * ws, u_int + u_slope * we]
        lower_y = [l_int + l_slope * ws, l_int + l_slope * we]
        fig_detail.add_trace(go.Scatter(
            x=line_x, y=upper_y, mode="lines",
            line=dict(color="#e74c3c", dash="dash", width=2),
            name="Upper trendline"))
        fig_detail.add_trace(go.Scatter(
            x=line_x, y=lower_y, mode="lines",
            line=dict(color="#2ecc71", dash="dash", width=2),
            name="Lower trendline"))

elif ptype in ("Bull Flag", "Bear Flag", "Rectangle"):
    c_high = trade.get("consol_high", None)
    c_low = trade.get("consol_low", None)
    cs_idx = trade.get("consol_start_idx", None)
    ce_idx = trade.get("consol_end_idx", None)
    if c_high and c_low and cs_idx is not None and ce_idx is not None:
        cs_idx, ce_idx = int(cs_idx), int(ce_idx)
        cs_dt = chart_df.iloc[min(cs_idx, len(chart_df) - 1)]["dt_label"]
        ce_dt = chart_df.iloc[min(ce_idx, len(chart_df) - 1)]["dt_label"]
        fig_detail.add_shape(
            type="rect", x0=cs_dt, x1=ce_dt,
            y0=c_low, y1=c_high,
            fillcolor="rgba(243, 156, 18, 0.15)",
            line=dict(color="#f39c12", dash="dash", width=1))
        fig_detail.add_annotation(
            x=cs_dt, y=c_high, text=f"Range: {c_high:.0f}–{c_low:.0f}",
            showarrow=False, font=dict(color="#f39c12", size=11))

# Entry / Exit markers
is_long = trade["direction"] == "LONG"
entry_sym = "triangle-up" if is_long else "triangle-down"
exit_sym = "triangle-down" if is_long else "triangle-up"

fig_detail.add_trace(go.Scatter(
    x=[trade["entry_dt"]], y=[trade["entry_price"]],
    mode="markers+text",
    marker=dict(symbol=entry_sym, size=16, color=pcolor,
                line=dict(width=2, color="white")),
    text=["ENTRY"], textposition="top center" if is_long else "bottom center",
    textfont=dict(size=11, color=pcolor), name="Entry"))

fig_detail.add_trace(go.Scatter(
    x=[trade["exit_dt"]], y=[trade["exit_price"]],
    mode="markers+text",
    marker=dict(symbol=exit_sym, size=16,
                color="#2ecc71" if trade["pnl_points"] > 0 else "#e74c3c",
                line=dict(width=2, color="white")),
    text=[f"EXIT ({trade['exit_reason']})"],
    textposition="bottom center" if is_long else "top center",
    textfont=dict(size=11), name="Exit"))

# SL / Target lines
fig_detail.add_hline(y=trade["sl"], line_dash="dot", line_color="#e74c3c",
                     line_width=1,
                     annotation_text=f"SL: {trade['sl']:.0f}",
                     annotation_position="left",
                     annotation_font_color="#e74c3c")
fig_detail.add_hline(y=trade["target"], line_dash="dot", line_color="#2ecc71",
                     line_width=1,
                     annotation_text=f"Target: {trade['target']:.0f}",
                     annotation_position="left",
                     annotation_font_color="#2ecc71")

pnl_col = "#2ecc71" if trade["pnl_points"] > 0 else "#e74c3c"
fig_detail.add_annotation(
    x=0.5, y=1.10, xref="paper", yref="paper",
    text=(
        f"<b>{trade['pattern']}</b> | {trade['direction']} | "
        f"Entry: {trade['entry_price']:.1f} → Exit: {trade['exit_price']:.1f} | "
        f"<span style='color:{pnl_col}'><b>P&L: {trade['pnl_points']:+.1f} pts "
        f"({trade['pnl_pct']:+.2f}%)</b></span> | "
        f"{trade['exit_reason']} | Hold: {trade['hold_bars']} bars"
    ),
    showarrow=False, font=dict(size=13), align="center",
)

fig_detail.update_layout(
    height=550,
    xaxis_rangeslider_visible=False,
    plot_bgcolor="#1a1a2e",
    paper_bgcolor="#16213e",
    font=dict(color="#e0e0e0"),
    xaxis=dict(gridcolor="#2a2a4a", title="DateTime"),
    yaxis=dict(gridcolor="#2a2a4a", title="Price"),
    margin=dict(l=50, r=50, t=80, b=40),
    legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
)
st.plotly_chart(fig_detail, use_container_width=True)


# ── Equity Curve ─────────────────────────────────────────────────────────────
st.header("Equity Curve")

sorted_trades = trades_df.sort_values("entry_dt").reset_index(drop=True)
equity = sorted_trades["pnl_points"].cumsum()

fig_eq = go.Figure()
fig_eq.add_trace(go.Scatter(
    x=sorted_trades["entry_dt"], y=equity,
    mode="lines+markers",
    line=dict(color="#1f77b4", width=2),
    marker=dict(size=5, color=[
        "#2ecc71" if p > 0 else "#e74c3c" for p in sorted_trades["pnl_points"]
    ]),
    name="Cumulative P&L",
))
fig_eq.update_layout(
    xaxis_title="DateTime", yaxis_title="Cumulative P&L (points)",
    height=400, margin=dict(l=50, r=30, t=30, b=40),
    plot_bgcolor="#1a1a2e", paper_bgcolor="#16213e",
    font=dict(color="#e0e0e0"),
    xaxis=dict(gridcolor="#2a2a4a"), yaxis=dict(gridcolor="#2a2a4a"),
)
st.plotly_chart(fig_eq, use_container_width=True)


# ── P&L Distribution by Pattern ─────────────────────────────────────────────
st.header("P&L Distribution")

col_bar, col_pie = st.columns(2)

with col_bar:
    fig_bar = go.Figure()
    for ptype in sorted_trades["pattern"].unique():
        sub = sorted_trades[sorted_trades["pattern"] == ptype]
        fig_bar.add_trace(go.Bar(
            x=sub["entry_dt"], y=sub["pnl_points"],
            name=ptype,
            marker_color=color_map.get(ptype, "#888"),
        ))
    fig_bar.update_layout(
        barmode="group", height=380,
        xaxis_title="Entry DateTime", yaxis_title="P&L (points)",
        margin=dict(l=40, r=20, t=30, b=40),
        plot_bgcolor="#1a1a2e", paper_bgcolor="#16213e",
        font=dict(color="#e0e0e0"),
        xaxis=dict(gridcolor="#2a2a4a"), yaxis=dict(gridcolor="#2a2a4a"),
    )
    st.plotly_chart(fig_bar, use_container_width=True)

with col_pie:
    exit_counts = trades_df["exit_reason"].value_counts()
    fig_pie = go.Figure(go.Pie(
        labels=exit_counts.index, values=exit_counts.values,
        marker=dict(colors=["#2ecc71", "#e74c3c", "#f39c12"]),
    ))
    fig_pie.update_layout(
        title="Exit Reasons", height=380,
        margin=dict(l=20, r=20, t=50, b=20),
        paper_bgcolor="#16213e", font=dict(color="#e0e0e0"),
    )
    st.plotly_chart(fig_pie, use_container_width=True)


# ── Drawdown ─────────────────────────────────────────────────────────────────
st.header("Drawdown")
cum_pnl = sorted_trades["pnl_points"].cumsum()
running_max = cum_pnl.cummax()
dd = cum_pnl - running_max

fig_dd = go.Figure()
fig_dd.add_trace(go.Scatter(
    x=sorted_trades["entry_dt"], y=dd,
    fill="tozeroy", mode="lines",
    line=dict(color="#e74c3c", width=1),
    fillcolor="rgba(231,76,60,0.2)", name="Drawdown",
))
fig_dd.update_layout(
    xaxis_title="DateTime", yaxis_title="Drawdown (points)",
    height=300, margin=dict(l=40, r=20, t=30, b=40),
    plot_bgcolor="#1a1a2e", paper_bgcolor="#16213e",
    font=dict(color="#e0e0e0"),
    xaxis=dict(gridcolor="#2a2a4a"), yaxis=dict(gridcolor="#2a2a4a"),
)
st.plotly_chart(fig_dd, use_container_width=True)


# ── Win Rate by Pattern Type (bar chart) ─────────────────────────────────────
st.header("Win Rate by Pattern Type")

wr_data = []
for ptype, m in per_pattern.items():
    wr_data.append({"Pattern": ptype, "Win Rate (%)": m["win_rate"],
                    "Trades": m["total_trades"]})
wr_df = pd.DataFrame(wr_data)

if not wr_df.empty:
    fig_wr = go.Figure()
    fig_wr.add_trace(go.Bar(
        x=wr_df["Pattern"], y=wr_df["Win Rate (%)"],
        marker_color=[color_map.get(p, "#888") for p in wr_df["Pattern"]],
        text=[f"{wr:.0f}% (n={n})" for wr, n in
              zip(wr_df["Win Rate (%)"], wr_df["Trades"])],
        textposition="outside",
    ))
    fig_wr.update_layout(
        yaxis_title="Win Rate (%)", height=400,
        margin=dict(l=40, r=20, t=30, b=40),
        plot_bgcolor="#1a1a2e", paper_bgcolor="#16213e",
        font=dict(color="#e0e0e0"),
        xaxis=dict(gridcolor="#2a2a4a"), yaxis=dict(gridcolor="#2a2a4a",
                                                     range=[0, 100]),
    )
    st.plotly_chart(fig_wr, use_container_width=True)


# ── Trade Log ────────────────────────────────────────────────────────────────
st.header("Trade Log")

display_cols = ["pattern", "direction", "entry_dt", "entry_price",
                "sl", "target", "exit_price", "exit_dt", "exit_reason",
                "hold_bars", "pnl_points", "pnl_pct", "actual_rr"]
avail_cols = [c for c in display_cols if c in trades_df.columns]

def _highlight(val):
    if isinstance(val, (int, float)):
        if val > 0:
            return "color: #2ecc71; font-weight: bold"
        if val < 0:
            return "color: #e74c3c; font-weight: bold"
    return ""

styled = trades_df[avail_cols].style.map(
    _highlight, subset=["pnl_points", "pnl_pct"])
st.dataframe(styled, use_container_width=True, height=450)

csv_out = trades_df.to_csv(index=False)
st.download_button("Download Trade Log CSV", csv_out,
                   file_name="pattern_backtest_results.csv", mime="text/csv")
