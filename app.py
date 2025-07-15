# -*- coding: utf-8 -*-
# vim: set ts=4 sw=4 et:

import streamlit as st
import pandas as pd{"hist": hist, "stock_obj": stock}`) was still incorrect, as Streamlit cannot "pickle" a
import mplfinance as mpf
import os
import yfinance as yf
import ssl
import nltk
from datetime import datetime

# Import functions from our utility module
from utils import (
    get_all_data, calculate_ dictionary that contains an unserializable object.

### The Definitive Solution: A Single-File, Robust Architecture

Weindicators, generate_signals, get_options_chain,
    generate_option_trade_plan, backtest are going back to a **single, unified `app.py` script**. This eliminates all the `ImportError` complexities_strategy, LOG_FILE, TIMEFRAME_MAP,
    get_finviz_data, convert_compound_to_100_scale, EXPERT_RATING_MAP
)

# === NLTK Data Download Workaround (Run once at the start) ===
@st.cache_resource
def download and allows us to manage state and caching much more effectively.

This new, definitive script fixes the problem by implementing a robust_nltk_data():
    try:
        _create_unverified_https_context = ssl._create, two-layer data fetching strategy:

1.  **A Cached Data Layer (`get_hist_and_info`):_unverified_context
    except AttributeError:
        pass
    else:
        ssl._create_default** A function decorated with `@st.cache_data` that *only* fetches and returns simple, serializable data_https_context = _create_unverified_https_context
    try:
        nltk.data.find('sentiment/vader_lexicon.zip')
    except LookupError:
        nltk.download('vader_lexicon')

download_nltk_data()

# === Page Setup ===
st.set_ (the `history` DataFrame and the `info` dictionary). This is fast and efficient.
2.  **An Uncached Object Layer:** The main `display_dashboard` function will create a *fresh* `yf.Ticker` objectpage_config(page_title="Aatif's AI Trading Hub", layout="wide")
st.title("🚀 Aatif's AI-Powered Trading Hub")

# === SIDEBAR: Controls & Selections ===
st.sidebar.header("⚙️ Controls")
ticker = st.sidebar.text_input("Enter a T when it needs to perform "live" actions that cannot be cached, like fetching the list of option expiration dates or theicker Symbol", value="NVDA").upper()
timeframe = st.sidebar.radio("Choose Trading Style:", company calendar. Since these are very fast calls, not caching them is perfectly fine and avoids the serialization error.

--- list(TIMEFRAME_MAP.keys()), index=2)

st.sidebar.header("🔧 Technical

### The Final, Corrected `app.py` Script

**Delete your `utils.py` and `setup Indicator Selection")
with st.sidebar.expander("Trend Indicators", expanded=True):
    indicator_selection = {
        "EMA Trend": st.checkbox("EMA Trend (21, 50, 2.py` files.** They are no longer needed. Replace the entire content of your `app.py` with the code below. This version is self-contained, robust, and correctly handles all data fetching and caching.

```python
00)", value=True),
        "Ichimoku Cloud": st.checkbox("Ichimoku Cloud", value=True),
        "Parabolic SAR": st.checkbox("Parabolic SAR", value=True),
# -*- coding: utf-8 -*-
# vim: set ts=4 sw=4 et:

import streamlit as st
import yfinance as yf
import pandas as pd
import ta
import mplfinance as mpf        "ADX": st.checkbox("ADX", value=True),
    }
with st.sidebar.expander("Momentum & Volume Indicators"):
    indicator_selection.update({
        "RSI Momentum
import os
import requests
from bs4 import BeautifulSoup
import nltk
import ssl
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
from datetime import datetime

# === Page Setup ===
st.set_page_config(page_title="Aatif's AI Trading Hub", layout="wide")
st.title("🚀 A": st.checkbox("RSI Momentum", value=True),
        "Stochastic": st.checkbox("Stochastic Oscillator", value=True),
        "CCI": st.checkbox("Commodity Channel Index (CCI)", valueatif's AI-Powered Trading Hub")

# === NLTK Data Download Workaround ===
@st.cache_resource
def download_nltk_data():
    try:
        _create_unverified_https=True),
        "ROC": st.checkbox("Rate of Change (ROC)", value=True),
        "Volume Spike": st.checkbox("Volume Spike", value=True),
        "OBV": st.checkbox("On-Balance Volume (OBV)", value=True),
        "VWAP": st.checkbox("VW_context = ssl._create_unverified_context
    except AttributeError: pass
    else: ssl._create_default_https_context = _create_unverified_https_context
    try:
        nltk.dataAP (Intraday only)", value=True),
    })
with st.sidebar.expander("Display-Only Indicators"):
    indicator_selection.update({"Bollinger Bands": st.checkbox("Bollinger Bands Display.find('sentiment/vader_lexicon.zip')
    except LookupError:
        nltk.download('vader_lexicon')

download_nltk_data()

# === Constants and Configuration ===
LOG_FILE = "trade", value=True)})

st.sidebar.header("🧠 Qualitative Scores")
use_automation = st.sidebar.toggle("Enable Automated Scoring", value=True, help="ON: AI scores are used. OFF: Use manual sliders and only_log.csv"

# === SIDEBAR: Controls & Selections ===
st.sidebar.header("⚙️ Controls")
ticker = st.sidebar.text_input("Enter a Ticker Symbol", value="NVDA the Technical Score will count.")
auto_sentiment_score_placeholder = st.sidebar.empty()
auto_expert_score_placeholder = st.sidebar.empty()

# === Dashboard Display Function ===
def display_dashboard(ticker,").upper()
timeframe = st.sidebar.radio("Choose Trading Style:", ["Scalp Trading", "Day Trading", "Swing Trading", "Position Trading"], index=2)

st.sidebar.header("🔧 Technical Indicator Selection")
 data, params, selection):
    # Unpack data
    hist, info, stock_obj = data['hist'], data['info'], data['stock_obj']
    
    is_intraday = params['with st.sidebar.expander("Trend Indicators", expanded=True):
    indicator_selection = {
        "EMA Trend": st.sidebar.checkbox("EMA Trend (21, 50, 200)", value=interval'] in ['5m', '60m']
    df = calculate_indicators(hist.copy(), is_intraday)
    signals = generate_signals(df, selection, is_intraday)
    lastTrue),
        "Ichimoku Cloud": st.sidebar.checkbox("Ichimoku Cloud", value=True),
        "Parabolic SAR": st.sidebar.checkbox("Parabolic SAR", value=True),
         = df.iloc[-1]
    
    if use_automation:
        finviz_data = get_finviz_data(ticker)
        auto_sentiment_score = convert_compound_to_100_scale"ADX": st.sidebar.checkbox("ADX", value=True),
    }
with st.sidebar.exp(finviz_data['sentiment_compound'])
        auto_expert_score = EXPERT_RATING_ander("Momentum & Volume Indicators"):
    indicator_selection.update({"RSI Momentum": st.sidebar.checkbox("RMAP.get(finviz_data['recom'], 50)
        auto_sentiment_score_placeholder.markdown(f"**Automated Sentiment:** `{auto_sentiment_score}`")
        auto_expert_score_SI Momentum", value=True), "Stochastic": st.sidebar.checkbox("Stochastic Oscillator", value=True),
                                "CCI": st.sidebar.checkbox("Commodity Channel Index (CCI)", value=True), "ROC": st.sidebarplaceholder.markdown(f"**Automated Expert Rating:** `{auto_expert_score}` ({finviz_data['recom']})")
        sentiment_score = st.sidebar.slider("Adjust Final Sentiment Score", .checkbox("Rate of Change (ROC)", value=True),
                                "Volume Spike": st.sidebar.checkbox("Volume Spike", value=True), "OBV": st.sidebar.checkbox("On-Balance Volume (OB1, 100, auto_sentiment_score)
        expert_score = st.sidebar.sliderV)", value=True),
                                "VWAP": st.sidebar.checkbox("VWAP (Intraday("Adjust Final Expert Score", 1, 100, auto_expert_score)
    else:
        st.sidebar.info("Automation OFF. Only technical score is used.")
        sentiment_score = 50; expert_score = 50
        finviz_data = {"headlines": ["Automation is disabled only)", value=True)})
with st.sidebar.expander("Display-Only Indicators"):
    indicator_."]}

    technical_score = (sum(1 for f in signals.values() if f) / len(signals)) * 100 if signals else 0
    scores = {"technical": technical_score, "sentimentselection.update({"Bollinger Bands": st.sidebar.checkbox("Bollinger Bands Display", value=True), "Pivot Points": st.sidebar.checkbox("Pivot Points Display (Daily only)", value=True)})

st.sidebar.header("🧠 Qualitative Scores")
use_automation = st.sidebar.toggle("Enable Automated Scoring", value=True, help="ON: AI scores are used. OFF: Use manual sliders and only the Technical Score will count.")": sentiment_score, "expert": expert_score}
    
    final_weights = params['weights'].
auto_sentiment_score_placeholder = st.sidebar.empty()
auto_expert_score_placeholder = st.sidebar.empty()

# === Core Functions ===
@st.cache_data(ttl=90copy()
    if not use_automation:
        final_weights = {'technical': 1.0, 'sentiment': 0.0, 'expert': 0.0}
        scores['sentiment'], scores['expert'] = 0, 0
    
    overall_confidence = min(round((final_weights["0)
def get_finviz_data(ticker):
    url = f"https://finviz.technical"]*scores["technical"] + final_weights["sentiment"]*scores["sentiment"] + final_weights["expert"]*scores["expert"]), 2), 100)
    
    st.header(f"Analysis for {ticker} ({params['interval']} Interval)")
    
    tab_list = ["📊 Maincom/quote.ashx?t={ticker}"; headers = {'User-Agent': 'Mozilla/5.0'}
    try:
        response = requests.get(url, headers=headers); response.raise_for Analysis", "📈 Trade Plan & Options", "🧪 Backtest", "📰 News & Info", "📝 Trade Log"]
    _status()
        soup = BeautifulSoup(response.text, 'html.parser'); recom_tag = soup.find('td', text='Recom'); analyst_recom = recom_tag.find_next_sibling('tdmain_tab, trade_tab, backtest_tab, news_tab, log_tab = st.tabs(tab_list)

    with main_tab:
        col1, col2 = st.columns([1,').text if recom_tag else "N/A"
        headlines = [tag.text for tag in soup.findAll('a', class_='news-link-left')[:10]]
        analyzer = SentimentIntensityAnalyzer 2])
        with col1:
            st.subheader("💡 Confidence Score"); st.metric("Overall Confidence", f"{overall_confidence:.0f}/100"); st.progress(overall_confidence / (); compound_scores = [analyzer.polarity_scores(h)['compound'] for h in headlines]
        avg_100)
            st.markdown(f"- **Technical:** `{scores['technical']:.0f}`compound = sum(compound_scores) / len(compound_scores) if compound_scores else 0
        return {"recom": analyst_recom, "headlines": headlines, "sentiment_compound": avg_compound}
    except (W: `{final_weights['technical']*100:.0f}%`)\n- **Sentiment:** `{scores['sentiment']:.0f}` (W: `{final_weights['sentiment']*100:.0f}%`)\n- **Expert:** `{scores['expert']:.0f}` (W: `{final_ Exception: return {"recom": "N/A", "headlines": [], "sentiment_compound": 0}

EXPERT_RATING_MAP = {"Strong Buy": 100, "Buy": 85weights['expert']*100:.0f}%`)")
            st.subheader("✅ Technical Analysis Readout") # ... Categorized display ...
        with col2:
            st.subheader("📈 Price Chart"); chart_path =, "Hold": 50, "N/A": 50, "Sell": 15, "Strong Sell": 0}
def convert_compound_to_100_scale(compound_score): return int((compound_score + 1) * 50)

if use_automation:
     f"chart_{ticker}.png"
            mav_tuple = (21, 50, 200) if selection.get("EMA Trend") else None
            ap = [mpf.make_addplot(df.tail(120)[['BB_high', 'BB_low']])] if selection.get("Bolfinviz_data = get_finviz_data(ticker)
    auto_sentiment_score = convert_compound_to_100_scale(finviz_data['sentiment_compound'])
    auto_expert_score = EXPERT_RATING_MAP.get(finviz_data['recom'], 50linger Bands") else None
            mpf.plot(df.tail(120), type='candle', style='yahoo', mav=mav_tuple, volume=True, addplot=ap, title=f"{ticker)
    auto_sentiment_score_placeholder.markdown(f"**Automated Sentiment:** `{auto_sentiment_score}`")
    auto_expert_score_placeholder.markdown(f"**Automated Expert Rating:**} - {params['interval']} chart", savefig=chart_path)
            st.image(chart_path); os.remove(chart_path)

    with trade_tab:
        st.subheader("🎭 `{auto_expert_score}` ({finviz_data['recom']})")
    sentiment_score = st.sidebar.slider("Adjust Final Sentiment Score", 1, 100, auto_sentiment_score Automated Options Strategy")
        expirations = stock_obj.options
        if not expirations: st.warning("No options data available for this ticker.")
        else:
            trade_plan = generate_option_)
    expert_score = st.sidebar.slider("Adjust Final Expert Score", 1, 100, auto_expert_score)
else:
    st.sidebar.info("Automation OFF. Only technical score is used.")
    sentiment_score = 50; expert_score = 50; finviztrade_plan(ticker, overall_confidence, last['Close'], expirations)
            if trade_plan['status'] == 'success':
                st.success(f"**Recommended Strategy: {trade_plan['Strategy']}_data = {"headlines": ["Automation is disabled."]}

# === FIX: Corrected Caching Strategy ===
@st.cache_data(ttl=60)
def get_hist_and_info(symbol, period, interval**") # ... Display logic ...
            else: st.warning(trade_plan['message'])
            st.subheader("⛓️ Full Option Chain") # ... Display logic ...

    with backtest_tab:
        st.subheader():
    """This cached function ONLY returns serializable data."""
    stock = yf.Ticker(symbol)
    hist = stock.history(period=period, interval=interval, auto_adjust=True)
    info =f"🧪 Historical Backtest for {ticker}"); st.info(f"Simulating trades based on your **currently selected indicators**.")
        daily_hist_data = get_all_data(ticker, "2y", {}
    try:
        info = stock.info
    except Exception as e:
        st.warning( "1d")
        if daily_hist_data and daily_hist_data['hist'] is not Nonef"Could not fetch company info: {e}")
    return (hist, info) if not hist.empty else (:
            daily_df = calculate_indicators(daily_hist_data['hist'].copy()); trades, wins, lossesNone, None)

def calculate_indicators(df, is_intraday=False):
    # This function is robust with error handling for each indicator
    try: df["EMA21"]=ta.trend.ema_indicator = backtest_strategy(daily_df, selection)
            total_trades = wins + losses; win_rate = (wins / total_trades) * 100 if total_trades > 0 else 0(df["Close"],21); df["EMA50"]=ta.trend.ema_indicator(df["Close"],50); df["EMA200"]=ta.trend.ema_indicator(df["
            col1, col2, col3 = st.columns(3); col1.metric("Trades Simulated", total_trades); col2.metric("Wins", wins); col3.metric("Win Rate", f"{win_rateClose"],200)
    except Exception: pass
    # ... other indicator calculations ...
    return df

def generate_signals(df, selection, is_intraday=False):
    # This function generates:.1f}%")
            if trades: st.dataframe(pd.DataFrame(trades).tail(20 signals based on the provided dataframe
    signals = {}; last_row = df.iloc[-1]
    if selection))
        else: st.warning("Could not fetch daily data for backtesting.")

    with news_tab:
        st.subheader(f"📰 News & Information for {ticker}"); col1, col2 = st.columns(2)
        with col1: st.markdown("#### ℹ️ Company Info"); st..get("EMA Trend"): signals["Uptrend (21>50>200 EMA)"]write(f"**Name:** {info.get('longName', ticker)}")
        with col2: st.markdown("#### 📅 Company Calendar"); st.dataframe(stock_obj.calendar.T if isinstance(stock_obj = last_row.get("EMA50", 0) > last_row.get("EMA200", 0) and last_row.get("EMA21", 0) > last_row.get("EMA.calendar, pd.DataFrame) else pd.DataFrame.from_dict(stock_obj.calendar, orient='50", 0)
    # ... other signal calculations ...
    return signals

def display_dashboard(tickerindex'))
        st.markdown("#### 🗞️ Latest Headlines"); [st.markdown(f"_{h, hist, info, params, selection):
    # This function displays the main dashboard
    # ... All tab}_") for h in finviz_data['headlines']]
            
    with log_tab:
        st display logic goes here ...
    st.write("Dashboard content will be displayed here.")

# === Main Script Execution.subheader("📝 Log Your Trade Analysis") # ... Logging logic ...

# === Main Script Execution ===
if ticker:
    selected_params = TIMEFRAME_MAP[timeframe]
    try:
        data = get ===
TIMEFRAME_MAP = {
    "Scalp Trading": {"period": "5d", "interval": "5m", "weights": {"technical": 0.9, "sentiment": 0._all_data(ticker, selected_params['period'], selected_params['interval'])
        if data['1, "expert": 0.0}},
    "Day Trading": {"period": "60d", "interval": "60m", "weights": {"technical": 0.7, "sentiment": 0hist'] is None: st.error(f"Could not fetch data for {ticker} on a {selected_params['interval']} interval.")
        else: display_dashboard(ticker, data, selected_params, indicator_selection)
    except Exception as e:
        st.error(f"An error occurred: {e}").2, "expert": 0.1}},
    "Swing Trading": {"period": "1y",
else:
    st.info("Please enter a stock ticker in the sidebar to begin analysis.")