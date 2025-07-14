#==GoogleAIStudio==
# -*- coding: utf-8 -*-
# vim: set ts=4 sw=4 et:

import streamlit as st
import yfinance as yf
import pandas as pd
import ta
import mplfinance as mpf
import matplotlib.pyplot as plt
import os

# === Page Setup ===
st.set_page_config(page_title="Aatif's Swing Dashboard", layout="wide")
st.title("📊 Aatif's Advanced Trade Analyzer")

# === SIDEBAR: User Inputs (Placed at the top for correct script flow) ===
st.sidebar.header("⚙️ Controls")
ticker = st.sidebar.text_input("Enter a Ticker Symbol", value="NVDA").upper()
timeframe = st.sidebar.radio("Choose Trading Style:",
    ["Swing Trading", "Day Trading", "Position Trading"],
    index=0, help="This changes the chart's time interval and analysis focus.")

# Map trading styles to yfinance intervals
TIMEFRAME_MAP = {
    "Swing Trading": {"period": "1y", "interval": "1d"},
    "Day Trading": {"period": "60d", "interval": "60m"},
    "Position Trading": {"period": "5y", "interval": "1wk"}
}
selected_period = TIMEFRAME_MAP[timeframe]["period"]
selected_interval = TIMEFRAME_MAP[timeframe]["interval"]

# === Utility Functions ===
def color_status(flag): return "🟢" if flag else "🔴"

# === Caching ===
@st.cache_data(ttl=600)
def get_data(symbol, period, interval):
    """Fetches stock data from Yahoo Finance."""
    stock = yf.Ticker(symbol)
    hist = stock.history(period=period, interval=interval)
    if hist.empty:
        return None, None
    info = stock.info
    return hist, info

def calculate_indicators(df):
    """Calculates all necessary technical indicators."""
    df["EMA21"] = ta.trend.ema_indicator(df["Close"], 21)
    df["EMA50"] = ta.trend.ema_indicator(df["Close"], 50)
    df["EMA200"] = ta.trend.ema_indicator(df["Close"], 200)
    df["RSI"] = ta.momentum.RSIIndicator(df["Close"]).rsi()
    bb = ta.volatility.BollingerBands(df["Close"])
    df["BB_low"] = bb.bollinger_lband()
    df["BB_high"] = bb.bollinger_hband()
    df["MACD_diff"] = ta.trend.macd_diff(df["Close"])
    df["ATR"] = ta.volatility.AverageTrueRange(df["High"], df["Low"], df["Close"]).average_true_range()
    df["Vol_Avg_50"] = df["Volume"].rolling(50).mean()
    return df

def generate_signals(df):
    """Generates trading signals based on improved logic."""
    last = df.iloc[-1]
    is_uptrend = last["EMA50"] > last["EMA200"]

    signals = {
        "Uptrend Confirmation (50>200 EMA)": is_uptrend,
        "Bullish Momentum (RSI > 50)": last["RSI"] > 50,
        "Momentum Reset (RSI 40-60)": 40 < last["RSI"] < 60 and is_uptrend,
        "MACD Bullish Cross": last["MACD_diff"] > 0,
        "Volume Spike (>1.5x Avg)": last["Volume"] > last["Vol_Avg_50"] * 1.5,
        "Price Near Support (Lower BB)": last["Close"] < last["BB_low"] * 1.02 # Within 2% of lower band
    }
    return signals

def calculate_confidence(scores, weights):
    """Calculates the weighted overall confidence score."""
    score = (
        weights["technical"] * scores["technical"] +
        weights["sentiment"] * scores["sentiment"] +
        weights["expert"] * scores["expert"]
    )
    return min(round(score, 2), 100) # Cap score at 100

def display_dashboard(ticker, hist, info, timeframe):
    """Main function to display all UI components."""
    # --- Data Processing ---
    df = calculate_indicators(hist.copy())
    last = df.iloc[-1]
    signals = generate_signals(df)

    # --- Dynamic Weights & Scores ---
    st.sidebar.header("🧠 Qualitative Scores")
    sentiment_score = st.sidebar.slider("Your Sentiment Score (1-100)", 1, 100, 50, help="Your personal feeling about market news, social media, etc.")
    expert_score = st.sidebar.slider("Expert Analysis Score (1-100)", 1, 100, 50, help="Your assessment of analyst ratings, fundamental reports, etc.")

    # Define weights based on strategy
    if timeframe == "Day Trading":
        weights = {"technical": 0.7, "sentiment": 0.2, "expert": 0.1}
    elif timeframe == "Swing Trading":
        weights = {"technical": 0.6, "sentiment": 0.2, "expert": 0.2}
    else: # Position Trading
        weights = {"technical": 0.4, "sentiment": 0.2, "expert": 0.4}

    # Calculate technical score
    signal_weights = {"Uptrend Confirmation (50>200 EMA)": 25, "Bullish Momentum (RSI > 50)": 15,
                        "Momentum Reset (RSI 40-60)": 20, "MACD Bullish Cross": 15, "Volume Spike (>1.5x Avg)": 15,
                        "Price Near Support (Lower BB)": 10}
    technical_score = sum([signal_weights[k] for k, v in signals.items() if v])
    trend_confirmation_bonus = 10 if last["RSI"] > 50 and last["Close"] > last["EMA200"] else 0
    technical_score += trend_confirmation_bonus
    technical_score = min(technical_score, 100)

    scores = {"technical": technical_score, "sentiment": sentiment_score, "expert": expert_score}
    overall_confidence = calculate_confidence(scores, weights)

    # === UI LAYOUT ===
    st.header(f"Analysis for {ticker} ({timeframe})")
    st.metric("Overall Confidence", f"{overall_confidence:.1f}/100")
    st.progress(overall_confidence / 100)

    # --- Key Info & Analysis Columns ---
    col1, col2 = st.columns([1, 2])

    with col1:
        st.subheader("💡 Key Data")
        st.metric("Current Price", f"${last['Close']:.2f}")
        resistance = df["High"][-60:].max()
        support = df["Low"][-60:].min()
        st.write(f"**Support (60-period):** ${support:.2f}")
        st.write(f"**Resistance (60-period):** ${resistance:.2f}")
        st.write(f"**ATR (Volatility):** {last['ATR']:.2f}")
        st.write(f"**Suggested Stop-Loss:** ${last['Close'] - 1.5 * last['ATR']:.2f}")

        st.subheader("📊 Technical Checklist")
        for signal, fired in signals.items():
            st.markdown(f"- {color_status(fired)} {signal}")
        st.markdown(f"- {color_status(trend_confirmation_bonus > 0)} Trend Confirmation Bonus (+10 pts)")

        st.subheader("🔗 Quick Links")
        st.markdown(f"- [Finviz](https://finviz.com/quote.ashx?t={ticker}) | [Barchart](https://www.barchart.com/stocks/quotes/{ticker}/overview) | [TipRanks](https://www.tipranks.com/stocks/{ticker}/forecast)")

    with col2:
        st.subheader("📈 Price Chart")
        chart_path = f"chart_{ticker}.png"
        mpf.plot(df.tail(120), type='candle', style='yahoo',
                 mav=(21, 50, 200), volume=True,
                 addplot=[mpf.make_addplot(df.tail(120)[['BB_high', 'BB_low']])],
                 title=f"{ticker} - {timeframe}",
                 savefig=chart_path)
        st.image(chart_path)
        if os.path.exists(chart_path):
            os.remove(chart_path)

        st.subheader("🧮 Confidence Breakdown")
        score_data = {
            'Component': ['Technical', 'Sentiment', 'Expert'],
            'Weight': [f"{w*100:.0f}%" for w in weights.values()],
            'Raw Score': [f"{s:.0f}/100" for s in scores.values()],
            'Contribution': [
                f"{weights['technical'] * scores['technical']:.1f}",
                f"{weights['sentiment'] * scores['sentiment']:.1f}",
                f"{weights['expert'] * scores['expert']:.1f}"
            ]
        }
        st.table(pd.DataFrame(score_data))


# === Main Script Execution ===
if ticker:
    try:
        hist_data, info_data = get_data(ticker, selected_period, selected_interval)
        if hist_data is None:
            st.error(f"Could not fetch data for {ticker}. The symbol may be invalid or delisted.")
        else:
            display_dashboard(ticker, hist_data, info_data, timeframe)
    except Exception as e:
        st.error(f"An error occurred: {e}")
        st.warning("This may be due to yfinance API rate limits. Please wait a few minutes and try again.")
else:
    st.info("Please enter a stock ticker in the sidebar to begin analysis.")