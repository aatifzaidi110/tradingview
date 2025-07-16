# app.py - Version 1.3
import sys
import os
import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt # Needed for plt.close() in display_components
import yfinance as yf # Keep this import here for direct yf usage if any, though it's also in utils

print("Current working directory:", os.getcwd())
print("Directory contents:", os.listdir())
print("=== DEBUG INFO ===")
print("Current directory:", os.getcwd())
print("Directory contents:", os.listdir())
print("Python path:", sys.path)
print("=================")

# Add current directory to path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

# Import functions from modules
from utils import (
    get_finviz_data, get_data, get_options_chain,
    calculate_indicators, calculate_pivot_points,
    generate_signals_for_row, backtest_strategy,
    generate_option_trade_plan, convert_compound_to_100_scale, EXPERT_RATING_MAP
)
try:
    from display_components import (
        display_main_analysis_tab, display_trade_plan_options_tab,
        display_backtest_tab, display_news_info_tab, display_trade_log_tab,
        display_ticker_comparison_chart # Import the new comparison chart function
    )
except ImportError as e:
    print("Import error details:", str(e))
    raise

# === Page Setup ===
st.set_page_config(page_title="Aatif's AI Trading Hub", layout="wide")
st.title("🚀 Aatif's AI-Powered Trading Hub")

# === Refresh Button ===
if st.button("🔄 Refresh Data", help="Click to fetch the latest data for all tickers and re-run analysis."):
    st.cache_data.clear() # Clear all cached data
    st.rerun() # Rerun the app

# === Constants and Configuration ===
LOG_FILE = "trade_log.csv" # Define here or pass from a config module

# === SIDEBAR: Controls & Selections ===
st.sidebar.header("⚙️ Controls")
# Changed to text_area for multiple tickers
ticker_input = st.sidebar.text_area("Enter Ticker Symbol(s) (comma or newline separated)", value="NVDA").upper()
# Split the input into a list of tickers, remove empty strings
tickers = [t.strip() for t in ticker_input.replace(',', '\n').split('\n') if t.strip()]

timeframe = st.sidebar.radio("Choose Trading Style:", ["Scalp Trading", "Day Trading", "Swing Trading", "Position Trading"], index=2)

st.sidebar.header("🔧 Technical Indicator Selection")
with st.sidebar.expander("Trend Indicators", expanded=True):
    indicator_selection = {
        "EMA Trend": st.checkbox("EMA Trend (21, 50, 200)", value=True),
        "Ichimoku Cloud": st.checkbox("Ichimoku Cloud", value=False, disabled=True), # Disabled Ichimoku
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
        "VWAP": st.checkbox("VWAP (Intraday only)", value=True, disabled=(timeframe not in ["Scalp Trading", "Day Trading"])),
    })
with st.sidebar.expander("Display-Only Indicators"):
    indicator_selection.update({
        "Bollinger Bands": st.checkbox("Bollinger Bands Display", value=True),
        "Pivot Points": st.checkbox("Pivot Points Display (Daily only)", value=True, disabled=(timeframe not in ["Swing Trading", "Position Trading"])),
    })

st.sidebar.header("🧠 Qualitative Scores")
use_automation = st.sidebar.toggle("Enable Automated Scoring", value=True, help="ON: AI scores are used. OFF: Use manual sliders and only the Technical Score will count.")
auto_sentiment_score_placeholder = st.sidebar.empty()
auto_expert_score_placeholder = st.sidebar.empty()

# === Dynamic Qualitative Score Calculation ===
sentiment_score = 50
expert_score = 50
finviz_data = {"headlines": ["Automation is disabled."]} # Default for when automation is off

# Main Script Execution - Loop through each ticker
TIMEFRAME_MAP = {
    "Scalp Trading": {"period": "5d", "interval": "5m", "weights": {"technical": 0.9, "sentiment": 0.1, "expert": 0.0}},
    "Day Trading": {"period": "60d", "interval": "60m", "weights": {"technical": 0.7, "sentiment": 0.2, "expert": 0.1}},
    "Swing Trading": {"period": "1y", "interval": "1d", "weights": {"technical": 0.6, "sentiment": 0.2, "expert": 0.2}},
    "Position Trading": {"period": "5y", "interval": "1wk", "weights": {"technical": 0.4, "sentiment": 0.2, "expert": 0.4}}
}
selected_params_main = TIMEFRAME_MAP[timeframe]

comparison_data = [] # List to store data for the comparison chart

if not tickers:
    st.info("Please enter at least one stock ticker in the sidebar to begin analysis.")
else:
    for i, ticker in enumerate(tickers):
        # Re-calculate sentiment and expert scores for each ticker if automation is on
        if use_automation:
            finviz_data = get_finviz_data(ticker) # This calls utils.get_finviz_data
            auto_sentiment_score = convert_compound_to_100_scale(finviz_data['sentiment_compound'])
            auto_expert_score = EXPERT_RATING_MAP.get(finviz_data['recom'], 50)
            # Display automated scores per ticker, or keep it global if preferred
            # For simplicity, keeping the sliders global as they are currently, but the underlying scores will update.
            # If you want per-ticker sliders, this section needs more complex state management.
            sentiment_score_current = auto_sentiment_score
            expert_score_current = auto_expert_score
        else:
            sentiment_score_current = sentiment_score # Use global manual slider value
            expert_score_current = expert_score     # Use global manual slider value

        try:
            hist_data, info_data = get_data(ticker, selected_params_main['period'], selected_params_main['interval'])
            if hist_data is None or info_data is None:
                st.error(f"Could not fetch data for {ticker} on a {selected_params_main['interval']} interval. Please check the ticker symbol or try again later.")
            else:
                # Calculate indicators once for the main display
                is_intraday_data = selected_params_main['interval'] in ['5m', '60m']
                df_calculated = calculate_indicators(hist_data.copy(), is_intraday_data)
                
                # Calculate pivot points separately for display
                df_pivots = calculate_pivot_points(hist_data.copy()) # Use original hist_data for pivots

                if df_calculated.empty:
                    st.warning(f"No data available for {ticker} after indicator calculations and cleaning. Please check ticker or time period.", icon="⚠️")
                    continue # Skip to next ticker

                # Calculate scores for display
                last_row_for_signals = df_calculated.iloc[-1]
                signals_for_score = generate_signals_for_row(last_row_for_signals, indicator_selection, df_calculated, is_intraday_data)
                
                technical_score = (sum(1 for f in signals_for_score.values() if f) / len(signals_for_score)) * 100 if signals_for_score else 0
                
                scores = {"technical": technical_score, "sentiment": sentiment_score_current, "expert": expert_score_current}
                
                # Apply weights for overall confidence
                final_weights = selected_params_main['weights'].copy()
                if not use_automation:
                    final_weights = {'technical': 1.0, 'sentiment': 0.0, 'expert': 0.0} # Only technical counts if automation is off
                
                overall_confidence = min(round((final_weights["technical"]*scores["technical"] + final_weights["sentiment"]*scores["sentiment"] + final_weights["expert"]*scores["expert"]), 2), 100)

                # Collect data for comparison chart
                current_price = last_row_for_signals['Close']
                comparison_data.append({
                    "Ticker": ticker,
                    "Current Price": current_price,
                    "Confidence Score": overall_confidence
                })

                st.subheader(f"📈 Analysis for {ticker}") # Move subheader here to be above tabs for each ticker
                # Display tabs
                tab_list = ["📊 Main Analysis", "📈 Trade Plan & Options", "🧪 Backtest", "📰 News & Info", "📝 Trade Log"]
                main_tab, trade_tab, backtest_tab, news_tab, log_tab = st.tabs(tab_list)

                with main_tab:
                    display_main_analysis_tab(ticker, df_calculated, info_data, selected_params_main, indicator_selection, overall_confidence, scores, final_weights, sentiment_score_current, expert_score_current, df_pivots)
                
                with trade_tab:
                    display_trade_plan_options_tab(ticker, df_calculated, overall_confidence)
                
                with backtest_tab:
                    display_backtest_tab(ticker, indicator_selection)
                
                with news_tab:
                    display_news_info_tab(ticker, info_data, finviz_data)
                
                with log_tab:
                    display_trade_log_tab(LOG_FILE, ticker, timeframe, overall_confidence)
            
            if i < len(tickers) - 1: # Add a separator between tickers
                st.markdown("---")
                st.markdown("---")

        except Exception as e:
            st.error(f"An unexpected error occurred during data processing for {ticker}: {e}", icon="🚫")
            st.exception(e)
    
    # Display the comparison chart at the very top after all tickers are processed
    if comparison_data:
        st.header("📊 Ticker Comparison Overview")
        display_ticker_comparison_chart(comparison_data)
        st.markdown("---") # Add a separator after the comparison chart
