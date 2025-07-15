# app.py
# -*- coding: utf-8 -*-
# vim: set ts=4 sw=4 et:

import streamlit as st
import pandas as pd
import yfinance as_trade:
            row = df_historical.iloc[i-1]
            if pd.isna(row.get('EMA200')): continue
            signals = generate_signals_for_row(row, yf
import nltk
import ssl

# Import functions from our utility modules
from utils import (get_hist_and_info, calculate_indicators, generate_signals_for_row, 
                   EXPERT_RATING selection, df_historical.iloc[:i])
            if signals and all(signals.values()):
                entry_MAP, convert_compound_to_100_scale, get_finviz_data, LOG__price = df_historical['Open'].iloc[i]
                stop_loss = entry_price - (FILE)
from display_components import display_main_analysis_tab, display_trade_log_tab

row['ATR'] * atr_multiplier); take_profit = entry_price + (row['ATR'] * atr_multiplier *# === NLTK Data Download Workaround ===
@st.cache_resource
def download_nltk_data():
 reward_risk_ratio)
                trades.append({"Date": df_historical.index[i].strftime('%Y-%m-%d'), "Type": "Entry", "Price": entry_price}); in_trade = True    try:
        _create_unverified_https_context = ssl._create_unverified_context

    wins = len([t for t in trades if t['Type'] == 'Exit (Win)']); losses = len    except AttributeError: pass
    else: ssl._create_default_https_context = _create_unverified_https_context
    try: nltk.data.find('sentiment/vader_lexicon.zip')([t for t in trades if t['Type'] == 'Exit (Loss)'])
    return trades, wins
    except LookupError: nltk.download('vader_lexicon')

download_nltk_data()

# === Page Setup ===
st.set_page_config(page_title="Aatif's AI Trading, losses

def generate_option_trade_plan(ticker, confidence, stock_price, expirations):
    if confidence < 60: return {"status": "warning", "message": "Confidence score is too low."}
    today Hub", layout="wide")
st.title("🚀 Aatif's AI-Powered Trading Hub")

# === SIDEBAR ===
st.sidebar.header("⚙️ Controls")
ticker = st.sidebar.text_input(" = datetime.now()
    target_exp_date = next((exp for exp in expirations if 45 <= (datetime.strptime(exp, '%Y-%m-%d') - today).days <= 90), NoneEnter a Ticker Symbol", value="NVDA").upper()
TIMEFRAME_MAP = {
    "Swing Trading": {"period": "1y", "interval": "1d", "weights": {"technical":)
    if not target_exp_date: return {"status": "warning", "message": "No suitable expiration found (45-90 days)."}
    
    calls, _ = get_options_chain(ticker 0.6, "sentiment": 0.2, "expert": 0.2}},
    ", target_exp_date)
    if calls.empty: return {"status": "error", "message": f"No call options for {target_exp_date}."}
    
    strategy = "Buy ATMPosition Trading": {"period": "5y", "interval": "1wk", "weights": {"technical": 0.4, "sentiment": 0.2, "expert": 0.4}}
}
time Call"; reason = "Moderate confidence favors an At-the-Money call to balance cost and potential."
    target_options = calls.iloc[[(calls['strike'] - stock_price).abs().idxmin()]]
frame = st.sidebar.radio("Choose Trading Style:", list(TIMEFRAME_MAP.keys()), index=0)

st.sidebar.header("🔧 Technical Indicator Selection")
with st.sidebar.expander("Trend Indicators", expanded=True):
    indicator_selection = {
        "EMA Trend": st.checkbox("    if confidence >= 75:
        strategy = "Buy ITM Call"; reason = "High confidence suggestsEMA Trend (21, 50, 200)", value=True),
    }
with st.sidebar.expander("Display-Only Indicators"):
    indicator_selection.update({"Bollinger Bands": st.checkbox("Bollinger Bands Display", value=True)})

st.sidebar.header("🧠 Qualitative Scores") a directional play. An ITM call (Delta > 0.60) offers good leverage."
        itm_options = calls[(calls['inTheMoney']) & (calls.get('delta', 0) > 0.60)]
use_automation = st.sidebar.toggle("Enable Automated Scoring", value=True, help="ON: AI scores are used. OFF: Only technical score counts.")
auto_sentiment_score_placeholder = st.sidebar.empty()
auto_expert_score_placeholder = st.sidebar.empty()

# === Main Logic ===
if ticker
        if not itm_options.empty: target_options = itm_options
    
    if target_options.empty: return {"status": "error", "message": "Could not find a suitable option contract."}
    
    rec_option = target_options.iloc[0]
    entry_price =:
    selected_params = TIMEFRAME_MAP[timeframe]
    hist_data, info_data = get_hist_and_info(ticker, selected_params['period'], selected_params['interval'])

     rec_option.get('ask', rec_option.get('lastPrice', 0))
    if entry_price == 0: entry_price = rec_option.get('lastPrice')
    if not isinstance(entry_price, (int, float)) or entry_price == 0: return {"status": "error",if hist_data is None:
        st.error(f"Could not fetch data for {ticker}. Please check the symbol or try again later.")
    else:
        # Qualitative Score Calculation
        if use_automation:
             "message": "Could not determine a valid entry price for the option."}
    
    risk_per_share = entry_price * 0.50; stop_loss = entry_price - risk_per_share; profit_target = entry_price + (risk_per_share * 2)
    return {"finviz_data = get_finviz_data(ticker)
            auto_sentiment_score = convert_compound_to_100_scale(finviz_data['sentiment_compound'])
            auto_expert_score = EXPERT_RATING_MAP.get(finviz_data['recom'], 50)
status": "success", "Strategy": strategy, "Reason": reason, "Expiration": target_exp_date,
            "Strike": f"${rec_option['strike']:.2f}", "Entry Price": f"~${entry_price:.2            auto_sentiment_score_placeholder.markdown(f"**Automated Sentiment:** `{auto_sentiment_score}`")
            auto_expert_score_placeholder.markdown(f"**Automated Expert Rating:** `{auto_expert_score}` ({finviz_data['recom']})")
            sentiment_score = st.f}",
            "Stop-Loss": f"~${stop_loss:.2f}", "Profit Target": f"~${profit_target:.2f}",
            "Max Risk / Share": f"${risk_persidebar.slider("Adjust Final Sentiment Score", 1, 100, auto_sentiment_score)
            expert_score = st.sidebar.slider("Adjust Final Expert Score", 1, 100, auto__share:.2f}", "Reward / Risk": "2 to 1", "Contract": rec_option}

def log_analysis(log_file, log_data):
    log_df = pd.DataFrame([log_dataexpert_score)
        else:
            st.sidebar.info("Automation OFF. Using manual scores."); sentiment_score = 50; expert_score = 50; finviz_data = {"headlines": []])
    file_exists = os.path.isfile(log_file)
    log_df.to_csv(log_file, mode='a', header=not file_exists, index=False)
    st.success(f"Analysis for {log_data['Ticker']} logged successfully!")