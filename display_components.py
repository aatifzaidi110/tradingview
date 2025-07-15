# display_components.py
import streamlit as st
import pandas as pd
import mplfinance as mpf
import os
import yfinance as yf
from utils import get_options_chain0m'])
        
        technical_score = (sum(1 for f in signals.values() if f) / len(signals)) * 100 if signals else 0
        scores = {"technical": technical_score, "sentiment": sentiment_score, "expert": expert_score}
        
        final_weights = selected_, generate_option_trade_plan, backtest_strategy, get_hist_and_info, calculate_indicators, log_analysis
from datetime import datetime

def format_value(signal_name, value, signals):
    is_params['weights'].copy()
        if not use_automation: final_weights = {'technical': 1.0, 'sentiment': 0.0, 'expert': 0.0}; scores['sentiment'], scores['fired = signals.get(signal_name, False); status_icon = '🟢' if is_fired else '🔴'
    name = signal_name.split('(')[0].strip(); value_str = f"`{value:.2f}`" if isinstance(value, (int, float)) else ""
    return f"{status_expert'] = 0, 0
        
        overall_confidence = min(round((final_weights["technical"]*scores["technical"] + final_weights["sentiment"]*scores["sentiment"] + final_weights["expert"]*scores["expert"]), 2), 100)
        
        display_params = {'ticker': ticker, 'interval': selected_params['interval']}
        
        tab_list = ["📊 Mainicon} **{name}:** {value_str}"

def display_main_analysis_tab(signals, df, selection, params, overall_confidence, scores, final_weights):
    last = df.iloc[-1]; is_intraday = params['interval'] in ['5m', '60m']
    col1, Analysis", "📝 Trade Log"]
        main_tab, log_tab = st.tabs(tab_list)

        with main_tab:
            display_main_analysis_tab(signals, df_calculated, indicator_selection, display_params, overall_confidence, scores, final_weights)
        with log_tab col2 = st.columns([1, 2])
    with col1:
        st.subheader("💡 Confidence Score"); st.metric("Overall Confidence", f"{overall_confidence:.0f}/100"); st.progress(overall_confidence / 100)
        st.markdown(f"- **Technical:**:
            display_trade_log_tab(LOG_FILE, ticker, timeframe, overall_confidence)
else:
    st.info("Please enter a stock ticker in the sidebar to begin analysis.")