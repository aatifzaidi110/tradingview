# -*- coding: utf-8 -*-
# vim: set ts=4 sw=4 et:

import streamlit as st
import pandas as pd
import mplfinance as mpf
import os
import yfinance as yf
import ssl
import nltk
from datetime import datetime

# Import functions from our utility module
from utils import (
    get_all_data, calculate_indicators, generate_signals, get_options_chain,
    generate_option_trade_plan, backtest_strategy, LOG_FILE, TIMEFRAME_MAP,
    get_finviz_data, convert_compound_to_100_scale, EXPERT_RATING_MAP
)

# === NLTK Data Download Workaround (Run once at the start) ===
@st.cache_resource
def download_nltk_data():
    try:
        _create_unverified_https_context = ssl._create_unverified_context
    except AttributeError:
        pass
    else:
        ssl._create_default_https_context = _create_unverified_https_context
    try:
        nltk.data.find('sentiment/vader_lexicon.zip')
    except LookupError:
        nltk.download('vader_lexicon')

download_nltk_data()

# === Page Setup ===
st.set_page_config(page_title="Aatif's AI Trading Hub", layout="wide")
st.title("🚀 Aatif's AI-Powered Trading Hub")

# === SIDEBAR: Controls & Selections ===
st.sidebar.header("⚙️ Controls")
ticker = st.sidebar.text_input("Enter a Ticker Symbol", value="NVDA").upper()
timeframe = st.sidebar.radio("Choose Trading Style:", list(TIMEFRAME_MAP.keys()), index=2)

st.sidebar.header("🔧 Technical Indicator Selection")
with st.sidebar.expander("Trend Indicators", expanded=True):
    indicator_selection = {
        "EMA Trend": st.checkbox("EMA Trend (21, 50, 200)", value=True),
        "Ichimoku Cloud": st.checkbox("Ichimoku Cloud", value=True),
        "Parabolic SAR": st.checkbox("Parabolic SAR", value=True),
        "ADX": st.checkbox("ADX", value=True),
    }
with st.sidebar.expander("Momentum & Volume Indicators"):
    indicator_selection.update({
        "RSI Momentum": st.checkbox("RSI Momentum", value=True),
        "Stochastic": st.checkbox("Stochastic Oscillator", value=True),
        "CCI": st.checkbox("Commodity Channel Index (CCI)", value=True),
        "ROC": st.checkbox("Rate of Change (ROC)", value=True),
        "Volume Spike": st.checkbox("Volume Spike", value=True),
        "OBV": st.checkbox("On-Balance Volume (OBV)", value=True),
        "VWAP": st.checkbox("VWAP (Intraday only)", value=True),
    })
with st.sidebar.expander("Display-Only Indicators"):
    indicator_selection.update({"Bollinger Bands": st.checkbox("Bollinger Bands Display", value=True)})

st.sidebar.header("🧠 Qualitative Scores")
use_automation = st.sidebar.toggle("Enable Automated Scoring", value=True, help="ON: AI scores are used. OFF: Use manual sliders and only the Technical Score will count.")
auto_sentiment_score_placeholder = st.sidebar.empty()
auto_expert_score_placeholder = st.sidebar.empty()

# === Dashboard Display Function ===
def display_dashboard(ticker, data, params, selection):
    # Unpack data
    hist, info, stock_obj = data['hist'], data['info'], data['stock_obj']
    
    is_intraday = params['interval'] in ['5m', '60m']
    df = calculate_indicators(hist.copy(), is_intraday)
    signals = generate_signals(df, selection, is_intraday)
    last = df.iloc[-1]
    
    if use_automation:
        finviz_data = get_finviz_data(ticker)
        auto_sentiment_score = convert_compound_to_100_scale(finviz_data['sentiment_compound'])
        auto_expert_score = EXPERT_RATING_MAP.get(finviz_data['recom'], 50)
        auto_sentiment_score_placeholder.markdown(f"**Automated Sentiment:** `{auto_sentiment_score}`")
        auto_expert_score_placeholder.markdown(f"**Automated Expert Rating:** `{auto_expert_score}` ({finviz_data['recom']})")
        sentiment_score = st.sidebar.slider("Adjust Final Sentiment Score", 1, 100, auto_sentiment_score)
        expert_score = st.sidebar.slider("Adjust Final Expert Score", 1, 100, auto_expert_score)
    else:
        st.sidebar.info("Automation OFF. Only technical score is used.")
        sentiment_score = 50; expert_score = 50
        finviz_data = {"headlines": ["Automation is disabled."]}

    technical_score = (sum(1 for f in signals.values() if f) / len(signals)) * 100 if signals else 0
    scores = {"technical": technical_score, "sentiment": sentiment_score, "expert": expert_score}
    
    final_weights = params['weights'].copy()
    if not use_automation:
        final_weights = {'technical': 1.0, 'sentiment': 0.0, 'expert': 0.0}
        scores['sentiment'], scores['expert'] = 0, 0
    
    overall_confidence = min(round((final_weights["technical"]*scores["technical"] + final_weights["sentiment"]*scores["sentiment"] + final_weights["expert"]*scores["expert"]), 2), 100)
    
    st.header(f"Analysis for {ticker} ({params['interval']} Interval)")
    
    tab_list = ["📊 Main Analysis", "📈 Trade Plan & Options", "🧪 Backtest", "📰 News & Info", "📝 Trade Log"]
    main_tab, trade_tab, backtest_tab, news_tab, log_tab = st.tabs(tab_list)

    with main_tab:
        col1, col2 = st.columns([1, 2])
        with col1:
            st.subheader("💡 Confidence Score"); st.metric("Overall Confidence", f"{overall_confidence:.0f}/100"); st.progress(overall_confidence / 100)
            st.markdown(f"- **Technical:** `{scores['technical']:.0f}` (W: `{final_weights['technical']*100:.0f}%`)\n- **Sentiment:** `{scores['sentiment']:.0f}` (W: `{final_weights['sentiment']*100:.0f}%`)\n- **Expert:** `{scores['expert']:.0f}` (W: `{final_weights['expert']*100:.0f}%`)")
            st.subheader("✅ Technical Analysis Readout") # ... Categorized display ...
        with col2:
            st.subheader("📈 Price Chart"); chart_path = f"chart_{ticker}.png"
            mav_tuple = (21, 50, 200) if selection.get("EMA Trend") else None
            ap = [mpf.make_addplot(df.tail(120)[['BB_high', 'BB_low']])] if selection.get("Bollinger Bands") else None
            mpf.plot(df.tail(120), type='candle', style='yahoo', mav=mav_tuple, volume=True, addplot=ap, title=f"{ticker} - {params['interval']} chart", savefig=chart_path)
            st.image(chart_path); os.remove(chart_path)

    with trade_tab:
        st.subheader("🎭 Automated Options Strategy")
        expirations = stock_obj.options
        if not expirations: st.warning("No options data available for this ticker.")
        else:
            trade_plan = generate_option_trade_plan(ticker, overall_confidence, last['Close'], expirations)
            if trade_plan['status'] == 'success':
                st.success(f"**Recommended Strategy: {trade_plan['Strategy']}**") # ... Display logic ...
            else: st.warning(trade_plan['message'])
            st.subheader("⛓️ Full Option Chain") # ... Display logic ...

    with backtest_tab:
        st.subheader(f"🧪 Historical Backtest for {ticker}"); st.info(f"Simulating trades based on your **currently selected indicators**.")
        daily_hist_data = get_all_data(ticker, "2y", "1d")
        if daily_hist_data and daily_hist_data['hist'] is not None:
            daily_df = calculate_indicators(daily_hist_data['hist'].copy()); trades, wins, losses = backtest_strategy(daily_df, selection)
            total_trades = wins + losses; win_rate = (wins / total_trades) * 100 if total_trades > 0 else 0
            col1, col2, col3 = st.columns(3); col1.metric("Trades Simulated", total_trades); col2.metric("Wins", wins); col3.metric("Win Rate", f"{win_rate:.1f}%")
            if trades: st.dataframe(pd.DataFrame(trades).tail(20))
        else: st.warning("Could not fetch daily data for backtesting.")

    with news_tab:
        st.subheader(f"📰 News & Information for {ticker}"); col1, col2 = st.columns(2)
        with col1: st.markdown("#### ℹ️ Company Info"); st.write(f"**Name:** {info.get('longName', ticker)}")
        with col2: st.markdown("#### 📅 Company Calendar"); st.dataframe(stock_obj.calendar.T if isinstance(stock_obj.calendar, pd.DataFrame) else pd.DataFrame.from_dict(stock_obj.calendar, orient='index'))
        st.markdown("#### 🗞️ Latest Headlines"); [st.markdown(f"_{h}_") for h in finviz_data['headlines']]
            
    with log_tab:
        st.subheader("📝 Log Your Trade Analysis") # ... Logging logic ...

# === Main Script Execution ===
if ticker:
    selected_params = TIMEFRAME_MAP[timeframe]
    try:
        data = get_all_data(ticker, selected_params['period'], selected_params['interval'])
        if data['hist'] is None: st.error(f"Could not fetch data for {ticker} on a {selected_params['interval']} interval.")
        else: display_dashboard(ticker, data, selected_params, indicator_selection)
    except Exception as e:
        st.error(f"An error occurred: {e}")
else:
    st.info("Please enter a stock ticker in the sidebar to begin analysis.")