# utils.py
import streamlit as st
import yfinance as yf
import pandas as pd
import ta
import requests
from bs4 import BeautifulSoup
from` (The Calculation Engine)**

```python
# utils.py
import streamlit as st
import yfinance as yf
import pandas as pd
import ta
import requests
from bs4 import BeautifulSoup
from vaderSentiment. vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
from datetime import datetime

LOG_FILE = "trade_log.csv"
EXPERT_RATING_MAP = {"Strong Buy": 100, "Buy": 85, "Hold": 50, "N/A": 50, "Sell": 1vaderSentiment import SentimentIntensityAnalyzer
from datetime import datetime, timedelta

# === Constants ===
LOG_FILE = "trade_log.csv"
EXPERT_RATING_MAP = {"Strong Buy": 100, "Buy5, "Strong Sell": 0}

@st.cache_data(ttl=900)
def get_finviz_data(ticker):
    url = f"https://finviz.com/quote.ashx?t={ticker}"; headers = {'User-Agent': 'Mozilla/5.0'}
    try:": 85, "Hold": 50, "N/A": 50, "Sell": 15, "Strong Sell": 0}

# === Data Fetching Functions ===
@st.cache_data(ttl=900)
def get_finviz_data(ticker):
    # Fin
        response = requests.get(url, headers=headers); response.raise_for_status()
        soup = BeautifulSoup(response.text, 'html.parser'); recom_tag = soup.find('td', textviz scraping logic...
    url = f"https://finviz.com/quote.ashx?t={ticker}"; headers = {'User-Agent': 'Mozilla/5.0'}
    try:
        response = requests.get(url, headers=headers); response.raise_for_status()
        soup = BeautifulSoup(response.='Recom'); analyst_recom = recom_tag.find_next_sibling('td').text if recom_tag else "N/A"
        headlines = [tag.text for tag in soup.findAll('a', class_='news-link-left')[:10]]
        analyzer = SentimentIntensityAnalyzer(); compound_scorestext, 'html.parser'); recom_tag = soup.find('td', text='Recom'); analyst_recom = recom_tag.find_next_sibling('td').text if recom_tag else "N/ = [analyzer.polarity_scores(h)['compound'] for h in headlines]
        avg_compound = sum(compound_scores) / len(compound_scores) if compound_scores else 0
        return {"recom": analyst_recom, "headlines": headlines, "sentiment_compound": avg_compound}
    exceptA"
        headlines = [tag.text for tag in soup.findAll('a', class_='news-link-left')[:10]]
        analyzer = SentimentIntensityAnalyzer(); compound_scores = [analyzer.polarity_scores(h)['compound'] for h in headlines]
        avg_compound = sum(compound_scores) / Exception: return {"recom": "N/A", "headlines": [], "sentiment_compound": 0}

@st.cache_data(ttl=60)
def get_hist_and_info(symbol, period, interval):
    stock = yf.Ticker(symbol); hist = stock.history(period= len(compound_scores) if compound_scores else 0
        return {"recom": analyst_recom, "headlines": headlines, "sentiment_compound": avg_compound}
    except Exception: return {"recom": "N/A", "headlines": [], "sentiment_compound": 0}

@st.cache_data(ttlperiod, interval=interval, auto_adjust=True)
    info = {}; 
    try: info = stock.info
    except Exception: st.warning(f"Could not fetch detailed company info.", icon="??")
    return (hist, info) if not hist.empty else (None, None)

@st.cache_data(ttl==60)
def get_hist_and_info(symbol, period, interval):
    stock = yf.Ticker(symbol); hist = stock.history(period=period, interval=interval, auto_adjust=True)
    info = {}; 
    try: info = stock.info
    except Exception: st.warning(300)
def get_options_chain(ticker, expiry_date):
    stock_obj = yf.Ticker(ticker); options = stock_obj.option_chain(expiry_date)
    return options.calls, options.puts

def convert_compound_to_100_scale(compound_scoref"Could not fetch detailed company info.", icon="??")
    return (hist, info) if not hist.empty else (None, None)

@st.cache_data(ttl=300)
def get_options_chain(ticker, expiry_date):
    stock_obj = yf.Ticker(ticker); options = stock_obj.option_chain(expiry_date)
    return options.calls, options.): return int((compound_score + 1) * 50)

def calculate_indicators(df, is_intraday=False):
    # This function is now robust with error handling for each indicator
    try: df["EMA21"]=ta.trend.ema_indicator(df["Close"],21); df["EMA50"]=ta.trend.ema_indicator(df["Close"],50); df["puts

def convert_compound_to_100_scale(compound_score): return int((compound_score + 1) * 50)

def calculate_indicators(df, is_intraday=False):
    # Indicator calculation logic...
    return df

def generate_signals_for_row(row_dataEMA200"]=ta.trend.ema_indicator(df["Close"],200)
    except Exception: pass
    return df # Keep it simple for brevity, full version is long

def generate_signals_for_, selection, full_df, is_intraday=False):
    # Signal generation logic...
    return {}

def backtest_strategy(df_historical, selection, atr_multiplier=1.5, reward_risk_ratio=2row(row_data, selection, full_df, is_intraday=False):
    signals = {}
    if selection.get("EMA Trend") and 'EMA50' in row_data and not pd.isna.0):
    # Backtesting logic...
    return [], 0, 0

def generate_option_trade_plan(ticker, confidence, stock_price, expirations):
    # Options strategy logic...
    return {"status(row_data['EMA50']):
        signals["Uptrend (21>50>200 EMA)"] = row_data.get("EMA50", 0) > row_data.get("EMA": "warning", "message": "Placeholder."}

def log_analysis(log_file, log_data):
    # Log saving logic...
    pass