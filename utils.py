# utils.py
import streamlit as st
import yfinance as yf
import pandas as pd
import ta
import requests
from bs4 import BeautifulSoup
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
from datetime import datetime

# === Constants ===
LOG_FILE = "trade_log.csv"
EXPERT_RATING_MAP = {"Strong Buy": 100, "Buy": 85, "Hold": 50, "N/A": 50, "Sell": 15, "Strong Sell": 0}
TIMEFRAME_MAP = {
    "Scalp Trading": {"period": "5d", "interval": "5m", "weights": {"technical": 0.9, "sentiment": 0.1, "expert": 0.0}},
    "Day Trading": {"period": "60d", "interval": "60m", "weights": {"technical": 0.7, "sentiment": 0.2, "expert": 0.1}},
    "Swing Trading": {"period": "1y", "interval": "1d", "weights": {"technical": 0.6, "sentiment": 0.2, "expert": 0.2}},
    "Position Trading": {"period": "5y", "interval": "1wk", "weights": {"technical": 0.4, "sentiment": 0.2, "expert": 0.4}}
}

# === Data Fetching Functions ===
@st.cache_data(ttl=900)
def get_finviz_data(ticker):
    # Finviz scraping logic...
    return {"recom": "N/A", "headlines": [], "sentiment_compound": 0}

# === NEW: Centralized Data Function ===
@st.cache_data(ttl=60)
def get_all_data(symbol, period, interval):
    """Fetches all primary data in one go to reduce API calls."""
    stock = yf.Ticker(symbol)
    hist = stock.history(period=period, interval=interval, auto_adjust=True)
    info = stock.info
    return {"hist": hist, "info": info, "stock_obj": stock} if not hist.empty else {"hist": None, "info": None, "stock_obj": None}

@st.cache_data(ttl=300)
def get_options_chain(ticker, expiry_date):
    stock_obj = yf.Ticker(ticker); options = stock_obj.option_chain(expiry_date)
    return options.calls, options.puts

def convert_compound_to_100_scale(compound_score): return int((compound_score + 1) * 50)

# === Indicator & Signal Functions ===
def calculate_indicators(df, is_intraday=False):
    # Indicator calculation logic...
    return df

def generate_signals(df, selection, is_intraday=False):
    # Signal generation logic...
    return {}

def backtest_strategy(df_historical, selection, atr_multiplier=1.5, reward_risk_ratio=2.0):
    # Backtesting logic...
    return [], 0, 0

def generate_option_trade_plan(ticker, confidence, stock_price, expirations):
    # Options strategy logic...
    return {"status": "warning", "message": "Placeholder message."}