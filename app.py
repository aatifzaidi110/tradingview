#==DeepSeekTest===
# -*- coding: utf-8 -*-
# vim: set ts=4 sw=4 et:

from typing import Dict, Tuple, List, Optional, Union
import streamlit as st
import yfinance as yf
import pandas as pd
import ta
import mplfinance as mpf
import matplotlib.pyplot as plt
import requests
from datetime import datetime, timedelta
from textblob import TextBlob
import numpy as np

# === Configuration ===
DEFAULT_WEIGHTS = {
    "technical": 0.6,
    "sentiment": 0.2,
    "expert": 0.2
}

STRATEGY_WEIGHTS = {
    "Scalp Trade": {"technical": 0.5, "sentiment": 0.3, "expert": 0.2},
    "Day Trade": {"technical": 0.6, "sentiment": 0.25, "expert": 0.15},
    "Swing Trade": {"technical": 0.6, "sentiment": 0.2, "expert": 0.2},
    "Position Trade": {"technical": 0.4, "sentiment": 0.2, "expert": 0.4}
}

# === Type Aliases ===
StockData = Tuple[pd.DataFrame, Dict, float, float, Union[str, datetime], Union[str, datetime]]
TradeRecord = Dict[str, Union[str, float, int]]

# === API Configuration ===
NEWS_API_KEY = "your_newsapi_key"  # Replace with actual key
TIPRANKS_API_KEY = "your_tipranks_key"  # Replace with actual key

# === Utility Functions ===
def status(flag: bool) -> str: 
    return "✅" if flag else "❌"

def color_status(flag: bool) -> str: 
    return "🟢 Green" if flag else "🔴 Red"

def calculate_sentiment(ticker: str) -> float:
    """Calculate news sentiment using NewsAPI and TextBlob"""
    try:
        url = f"https://newsapi.org/v2/everything?q={ticker}&apiKey={NEWS_API_KEY}"
        response = requests.get(url)
        news = response.json()
        
        if news.get('articles'):
            polarities = [TextBlob(article['title']).sentiment.polarity 
                         for article in news['articles'][:5]]
            avg_polarity = sum(polarities) / len(polarities)
            return max(0, min(100, (avg_polarity + 1) * 50))
    except Exception as e:
        st.warning(f"NewsAPI fetch failed: {str(e)}")
    return 50  # Neutral if API fails

def get_expert_score(ticker: str) -> float:
    """Get expert score from TipRanks (mock implementation)"""
    try:
        # This is a mock implementation - real API would require subscription
        url = f"https://api.tipranks.com/api/v1/stocks/{ticker}/consensus"
        headers = {"Authorization": f"Bearer {TIPRANKS_API_KEY}"}
        response = requests.get(url, headers=headers)
        if response.status_code == 200:
            data = response.json()
            return data.get("consensus", {}).get("score", 50)
    except Exception as e:
        st.warning(f"TipRanks fetch failed: {str(e)}")
    return 50  # Default score if API fails

def calculate_confidence(technical: float, sentiment: float, expert: float, 
                       weights: Dict[str, float]) -> float:
    """Calculate weighted confidence score"""
    score = round(
        weights["technical"] * technical +
        weights["sentiment"] * sentiment +
        weights["expert"] * expert,
        2
    )
    return min(100, score)  # Cap at 100

# === Data Fetching ===
@st.cache_data(ttl=600)
def get_data(symbol: str) -> StockData:
    """Fetch stock data with enhanced error handling"""
    try:
        stock = yf.Ticker(symbol)
        hist = stock.history(period="1y")
        
        # Fallback if primary fetch fails
        if hist.empty:
            hist = yf.download(symbol, period="1y")
        
        info = stock.info
        price = info.get("currentPrice", hist["Close"].iloc[-1])
        previous_close = info.get("previousClose", hist["Close"].iloc[-2])
        earnings = info.get("earningsDate", "N/A")
        dividend = info.get("dividendDate", "N/A")
        
        return hist, info, price, previous_close, earnings, dividend
    except Exception as e:
        raise ValueError(f"Data fetch failed: {str(e)}")

# === Technical Analysis ===
def calculate_technical_indicators(df: pd.DataFrame) -> pd.DataFrame:
    """Calculate all technical indicators with enhanced set"""
    # Basic indicators
    df["EMA21"] = ta.trend.ema_indicator(df["Close"], 21)
    df["EMA50"] = ta.trend.ema_indicator(df["Close"], 50)
    df["EMA200"] = ta.trend.ema_indicator(df["Close"], 200)
    df["RSI"] = ta.momentum.RSIIndicator(df["Close"]).rsi()
    
    # Volume-weighted MACD
    macd = ta.trend.MACD(df["Close"])
    df["MACD_diff"] = macd.macd_diff()
    df["VWAP"] = ta.volume.VolumeWeightedAveragePrice(
        df["High"], df["Low"], df["Close"], df["Volume"]).volume_weighted_average_price()
    
    # Ichimoku Cloud
    ichimoku = ta.trend.IchimokuIndicator(df["High"], df["Low"])
    df["Ichimoku_Conversion"] = ichimoku.ichimoku_conversion_line()
    df["Ichimoku_Base"] = ichimoku.ichimoku_base_line()
    
    # Bollinger Bands
    bb = ta.volatility.BollingerBands(df["Close"])
    df["BB_low"] = bb.bollinger_lband()
    df["BB_high"] = bb.bollinger_hband()
    
    # Volatility and Volume
    df["ATR"] = ta.volatility.AverageTrueRange(
        df["High"], df["Low"], df["Close"]).average_true_range()
    df["Vol_Avg"] = df["Volume"].rolling(50).mean()
    
    return df

# === Backtesting ===
def backtest_signals(
    df_historical: pd.DataFrame,
    atr_multiplier: float = 1.0,
    reward_multiplier: float = 2.0,
    commission_per_trade: float = 1.00,
    slippage_percent: float = 0.1
) -> List[TradeRecord]:
    """Enhanced backtesting with commissions and slippage"""
    trades = []
    
    # Ensure we have enough data (200 periods minimum)
    if len(df_historical) < 200:
        return trades
    
    # Start from index position 200
    for i in range(200, len(df_historical) - 5):
        row = df_historical.iloc[i]
        
        # Entry signals - properly formatted conditions
        entry_condition = (
            (30 < row["RSI"] < 70) and 
            (row["MACD_diff"] > 0) and 
            (row["EMA21"] > row["EMA50"] > row["EMA200"]) and 
            (row["Close"] > row["VWAP"])
        )
        
        if entry_condition:
            # Calculate prices with slippage
            entry_price = row["Close"] * (1 + slippage_percent/100)
            stop_loss = entry_price - row["ATR"] * atr_multiplier
            take_profit = entry_price + row["ATR"] * reward_multiplier
            
            # Simulate trade over next 5 periods
            for j in range(i+1, min(i+6, len(df_historical))):  # Properly closed parentheses
                future_row = df_historical.iloc[j]
                low_price = future_row["Low"] * (1 - slippage_percent/100)
                high_price = future_row["High"] * (1 + slippage_percent/100)
                
                if low_price <= stop_loss:
                    exit_price = stop_loss
                    result = "Loss"
                    break
                elif high_price >= take_profit:
                    exit_price = take_profit
                    result = "Win"
                    break
            else:
                exit_price = df_historical.iloc[min(i+5, len(df_historical)-1)]["Close"]
                result = "Neutral"
            
            # Calculate P&L with commissions
            pnl = (exit_price - entry_price) - commission_per_trade
            
            trades.append({
                "Date": row.name.strftime('%Y-%m-%d'),
                "Entry": round(entry_price, 2),
                "Exit": round(exit_price, 2),
                "Result": result,
                "PnL": round(pnl, 2),
                "ATR": round(row["ATR"], 2),
                "RSI": round(row["RSI"], 2)
            })
    
    return trades

# === Main Application ===
def main():
    st.set_page_config(page_title="Enhanced Swing Analyzer", layout="centered")
    st.title("📊 Enhanced Swing Trade Analyzer")
    
    # === Ticker Input ===
    ticker = st.text_input("Enter a Ticker Symbol", value="NVDA").strip().upper()
    if not ticker:
        st.info("Please enter a ticker symbol to analyze.")
        st.stop()
    
    # === Data Loading ===
    try:
        hist, info, price, previous_close, earnings, dividend = get_data(ticker)
    except Exception as e:
        st.error(f"⚠️ Data fetch failed: {str(e)}")
        if st.button("🔄 Retry"):
            st.rerun()
        st.stop()
    
    # === Technical Analysis ===
    df = calculate_technical_indicators(hist.copy())
    last = df.iloc[-1]
    
    # === Strategy Selection ===
    st.subheader("🎯 Select Your Trading Style")
    timeframe = st.radio("Choose strategy:", list(STRATEGY_WEIGHTS.keys()))
    selected_strategy = timeframe.split(" ")[0] + " Trade"
    weights = STRATEGY_WEIGHTS[selected_strategy]
    
    # === Signal Calculation ===
    signals = {
        "RSI": 30 < last["RSI"] < 70,
        "MACD": last["MACD_diff"] > 0,
        "EMA": last["EMA21"] > last["EMA50"] > last["EMA200"],
        "Ichimoku": last["Close"] > last["Ichimoku_Conversion"],
        "Volume": last["Volume"] > last["Vol_Avg"] * 1.5,
        "VWAP": last["Close"] > last["VWAP"]
    }
    
    technical_score = sum([20 for signal in signals.values() if signal])
    ml_boost = 10 if last["RSI"] > 50 and price > last["EMA200"] else 0
    technical_score = min(100, technical_score + ml_boost)
    
    # === API Integrations ===
    with st.spinner("Fetching sentiment and expert data..."):
        sentiment_score = calculate_sentiment(ticker)
        expert_score = get_expert_score(ticker)
    
    # === Confidence Calculation ===
    overall_confidence = calculate_confidence(
        technical_score, sentiment_score, expert_score, weights)
    
    # === Display Results ===
    # [Previous visualization code would go here]
    # Add new visualizations for Ichimoku and VWAP-MACD
    
    # === Enhanced Backtest Results ===
    st.subheader("🧪 Enhanced Backtest Results")
    trades = backtest_signals(df, commission_per_trade=1.00, slippage_percent=0.1)
    
    if trades:
        trades_df = pd.DataFrame(trades)
        win_rate = len(trades_df[trades_df["Result"] == "Win"]) / len(trades_df)
        avg_pnl = trades_df["PnL"].mean()
        
        st.write(f"**Backtest Period:** {df.index[0].date()} to {df.index[-1].date()}")
        st.write(f"**Total Trades:** {len(trades_df)}")
        st.write(f"**Win Rate:** {win_rate:.1%}")
        st.write(f"**Average P&L per Trade:** ${avg_pnl:.2f}")
        
        # Display equity curve
        trades_df["Cumulative_PnL"] = trades_df["PnL"].cumsum()
        fig, ax = plt.subplots()
        ax.plot(trades_df["Date"], trades_df["Cumulative_PnL"])
        ax.set_title("Equity Curve")
        ax.set_xlabel("Date")
        ax.set_ylabel("Cumulative P&L ($)")
        st.pyplot(fig)

if __name__ == "__main__":
    main()