# app.py
# -*- coding: utf-8 -*-
# vim: set ts=4 sw=4 et:

import streamlit as st
import pandas as pd
import yfinance as yf
import nltkTrue),
        # ... other checkboxes
    }
# ... other expanders

st.sidebar.header("🧠 Qualitative Scores")
use_automation = st.sidebar.toggle("Enable Automated Scoring", value=True, help="ON: AI scores are used. OFF: Only technical score counts.")
auto_sentiment_score_placeholder = st.sidebar.empty()
auto_expert_score_placeholder = st.sidebar.empty()

# === Main Logic ===
if
import ssl

# Import functions from our utility modules
from utils import (get_hist_and_info, calculate_indicators, generate_signals_for_row, 
                   EXPERT_RATING_MAP, convert_compound_to_100_scale, get_finviz_data, LOG_FILE)
from ticker:
    selected_params = TIMEFRAME_MAP[timeframe]
    hist_data, info_data = get_hist_and_info(ticker, selected_params['period'], selected_params[' display_components import display_main_analysis_tab, display_trade_log_tab

# === NLTK Data Download Workaround ===
@st.cache_resource
def download_nltk_data():
    try:
interval'])

    if hist_data is None:
        st.error(f"Could not fetch data for {ticker}. Please check the symbol or try again later.")
    else:
        # --- Qualitative Score Calculation ---
        if        _create_unverified_https_context = ssl._create_unverified_context
    except AttributeError: pass
    else: ssl._create_default_https_context = _create_unverified_https_context
    try: nltk.data.find('sentiment/vader_lexicon.zip')
    except Lookup use_automation:
            finviz_data = get_finviz_data(ticker)
            auto_sentiment_score = convert_compound_to_100_scale(finviz_data['sentiment_compoundError: nltk.download('vader_lexicon')

download_nltk_data()

# === Page Setup ===
st.set_page_config(page_title="Aatif's AI Trading Hub", layout="wide")'])
            auto_expert_score = EXPERT_RATING_MAP.get(finviz_data['recom'], 50)
            auto_sentiment_score_placeholder.markdown(f"**Automated Sentiment:** `{auto_sentiment_score}`")
            auto_expert_score_placeholder.markdown(f"
st.title("🚀 Aatif's AI-Powered Trading Hub")

# === SIDEBAR ===
st.sidebar.header("⚙️ Controls")
ticker = st.sidebar.text_input("Enter a Ticker**Automated Expert Rating:** `{auto_expert_score}` ({finviz_data['recom']})")
            sentiment_score = st.sidebar.slider("Adjust Final Sentiment Score", 1, 100, auto_sentiment_score)
            expert_score = st.sidebar.slider("Adjust Final Expert Score", 1, 100, auto_expert_score)
        else:
            st.sidebar. Symbol", value="NVDA").upper()
TIMEFRAME_MAP = {
    "Swing Trading": {"period": "1y", "interval": "1d", "weights": {"technical": 0.6, "sentiment": 0.2, "expert": 0.2}},
    "Position Trading": {"period": "5yinfo("Automation OFF. Using manual scores."); sentiment_score = 50; expert_score = 50; finviz_data = {"headlines": []}

        df_calculated = calculate_indicators(hist_data.copy(), selected_params['interval'] in ['5m', '60m'])
        signals = generate_signals_for", "interval": "1wk", "weights": {"technical": 0.4, "sentiment": 0.2, "expert": 0.4}}
}
timeframe = st.sidebar.radio("Choose Trading Style:", list(TIMEFRAME_MAP.keys()), index=0)

st.sidebar.header_row(df_calculated.iloc[-1], indicator_selection, df_calculated, selected_params['interval'] in ['5m', '60m'])
        
        technical_score = (sum(1 for("🔧 Technical Indicator Selection")
with st.sidebar.expander("Trend Indicators", expanded=True):
    indicator_selection = {
        "EMA Trend": st.checkbox("EMA Trend (21, 50 f in signals.values() if f) / len(signals)) * 100 if signals else 0, 200)", value=True),
    }
with st.sidebar.expander("Display-
        scores = {"technical": technical_score, "sentiment": sentiment_score, "expert": expert_score}
        
        final_weights = selected_params['weights'].copy()
        if not use_automation: finalOnly Indicators"):
    indicator_selection.update({"Bollinger Bands": st.checkbox("Bollinger Bands Display",_weights = {'technical': 1.0, 'sentiment': 0.0, 'expert': 0 value=True)})

st.sidebar.header("🧠 Qualitative Scores")
use_automation = st.sidebar.toggle("Enable Automated Scoring", value=True, help="ON: AI scores are used. OFF: Only technical score counts.").0}; scores['sentiment'], scores['expert'] = 0, 0
        
        overall_confidence = min(
auto_sentiment_score_placeholder = st.sidebar.empty()
auto_expert_score_placeholder =round((final_weights["technical"]*scores["technical"] + final_weights["sentiment"]*scores["sentiment"] + final_weights["expert"]*scores["expert"]), 2), 100)
        
        display_params = {'ticker': ticker, 'interval': selected_params['interval']}
        
        tab st.sidebar.empty()

# === Main Logic ===
if ticker:
    selected_params = TIMEFRAME_MAP[timeframe]
    hist_data, info_data = get_hist_and_info(ticker, selected_params['period'], selected_params['interval'])

    if hist_data is None:
        st.error(f"Could not fetch data for {ticker}. Please check the symbol or try again later.")
_list = ["📊 Main Analysis", "📈 Trade Plan & Options", "🧪 Backtest", "📰 News & Info", "📝 Trade Log"]
        main_tab, trade_tab, backtest_tab, news_    else:
        # Qualitative Score Calculation
        if use_automation:
            finviz_data = get_finviz_data(ticker)
            auto_sentiment_score = convert_compound_to_100_scale(finviz_data['sentiment_compound'])
            auto_expert_score = EXPERT_tab, log_tab = st.tabs(tab_list)

        with main_tab:
            display_main_analysis_tab(signals, df_calculated, indicator_selection, display_params, overall_confidence, scores, final_weights)
        with trade_tab:
            display_trade_plan_options_RATING_MAP.get(finviz_data['recom'], 50)
            auto_sentiment_score_placeholder.markdown(f"**Automated Sentiment:** `{auto_sentiment_score}`")
            auto_expert_score_placeholder.markdown(f"**Automated Expert Rating:** `{auto_expert_scoretab(ticker, df_calculated, overall_confidence)
        with backtest_tab:
            display_backtest_tab(ticker, indicator_selection)
        with news_tab:
            display_news_info_tab(ticker, info_data, finviz_data)
        with log_tab:
            }` ({finviz_data['recom']})")
            sentiment_score = st.sidebar.slider("Adjust Final Sentiment Score", 1, 100, auto_sentiment_score)
            expert_scoredisplay_trade_log_tab(LOG_FILE, ticker, timeframe, overall_confidence)
else:
    st.info("Please enter a stock ticker in the sidebar to begin analysis.")