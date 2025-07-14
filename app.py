#==GoogleAIStudio==
# -*- coding: utf-8 -*-
# vim: set ts=4 sw=4 et:

import streamlit as st
import yfinance as yf
import pandas as pd
import ta
import mplfinance as mpf
import os
import requests
from bs4 import BeautifulSoup

# === Page Setup ===
st.set_page_config(page_title="Aatif's Pro Analyzer", layout="wide")
st.title("📈 Aatif's Controllable Trade Analyzer")

# === SIDEBAR: User Inputs ===
st.sidebar.header("⚙️ Controls")
ticker = st.sidebar.text_input("Enter a Ticker Symbol", value="NVDA").upper()
timeframe = st.sidebar.radio("Choose Trading Style:",
    ["Scalp Trading", "Day Trading", "Swing Trading", "Position Trading"],
    index=2)

# === NEW: Automation On/Off Switch and Reload Button ===
st.sidebar.header("🤖 Automation Settings")
use_automation = st.sidebar.toggle("Enable Automated Scoring", value=True, help="ON: Scrape Finviz for scores. OFF: Use manual sliders.")
if use_automation:
    if st.sidebar.button("🔄 Refresh Automated Scores", help="Force a new data scrape from Finviz, ignoring the cache."):
        st.cache_data.clear() # Clear the cache for all functions
        st.experimental_rerun()

# === Data Scraping & Mapping Definitions ===
@st.cache_data(ttl=900)
def get_finviz_data(ticker):
    url = f"https://finviz.com/quote.ashx?t={ticker}"
    headers = {'User-Agent': 'Mozilla/5.0'}
    try:
        response = requests.get(url, headers=headers)
        response.raise_for_status()
        soup = BeautifulSoup(response.text, 'html.parser')
        recom_tag = soup.find('td', text='Recom')
        analyst_recom = recom_tag.find_next_sibling('td').text if recom_tag else "N/A"
        headlines = [tag.text for tag in soup.findAll('a', class_='news-link-left')[:10]]
        return {"recom": analyst_recom, "headlines": headlines}
    except Exception as e:
        st.sidebar.warning(f"Finviz scrape failed: {e}", icon="⚠️")
        return {"recom": "N/A", "headlines": []}

EXPERT_RATING_MAP = {"Strong Buy": 100, "Buy": 85, "Hold": 50, "N/A": 50, "Sell": 15, "Strong Sell": 0}
SENTIMENT_RATING_MAP = {"Very Positive": 95, "Positive": 75, "Neutral": 50, "Negative": 25, "Very Negative": 5}

# === Sidebar Logic: Automated vs. Manual Scoring ===
st.sidebar.header("🧠 Qualitative Scores")
if use_automation:
    finviz_data = get_finviz_data(ticker)
    expert_options = list(EXPERT_RATING_MAP.keys())
    try:
        default_expert_index = expert_options.index(finviz_data['recom'])
    except ValueError:
        default_expert_index = expert_options.index("N/A")
    
    expert_rating_selection = st.sidebar.selectbox("Expert Analysis (from Finviz)", expert_options, index=default_expert_index)
    sentiment_rating_selection = st.sidebar.selectbox("Your Sentiment (from Headlines)", list(SENTIMENT_RATING_MAP.keys()), index=2)
    expert_score = EXPERT_RATING_MAP[expert_rating_selection]
    sentiment_score = SENTIMENT_RATING_MAP[sentiment_rating_selection]
else:
    st.sidebar.info("Automation is OFF. Using manual sliders.")
    sentiment_score = st.sidebar.slider("Manual Sentiment Score (1-100)", 1, 100, 50)
    expert_score = st.sidebar.slider("Manual Expert Score (1-100)", 1, 100, 50)
    finviz_data = {"headlines": ["Automation is disabled."]} # Placeholder

# Map styles to yfinance intervals and define weights
TIMEFRAME_MAP = {
    "Scalp Trading": {"period": "1d", "interval": "5m", "weights": {"technical": 0.9, "sentiment": 0.1, "expert": 0.0}},
    "Day Trading": {"period": "60d", "interval": "60m", "weights": {"technical": 0.7, "sentiment": 0.2, "expert": 0.1}},
    "Swing Trading": {"period": "1y", "interval": "1d", "weights": {"technical": 0.6, "sentiment": 0.2, "expert": 0.2}},
    "Position Trading": {"period": "5y", "interval": "1wk", "weights": {"technical": 0.4, "sentiment": 0.2, "expert": 0.4}}
}
selected_params = TIMEFRAME_MAP[timeframe]

# === Core Functions (unchanged) ===
@st.cache_data(ttl=60)
def get_data(symbol, period, interval):
    stock = yf.Ticker(symbol); hist = stock.history(period=period, interval=interval, auto_adjust=True)
    return (hist, stock.info) if not hist.empty else (None, None)
def calculate_indicators(df):
    df["EMA21"]=ta.trend.ema_indicator(df["Close"],21); df["EMA50"]=ta.trend.ema_indicator(df["Close"],50); df["EMA200"]=ta.trend.ema_indicator(df["Close"],200)
    df["RSI"]=ta.momentum.RSIIndicator(df["Close"]).rsi(); bb=ta.volatility.BollingerBands(df["Close"]); df["BB_low"]=bb.bollinger_lband(); df["BB_high"]=bb.bollinger_hband()
    df["MACD_diff"]=ta.trend.macd_diff(df["Close"]); df["ATR"]=ta.volatility.AverageTrueRange(df["High"],df["Low"],df["Close"]).average_true_range()
    df["Vol_Avg_50"]=df["Volume"].rolling(50).mean(); return df
def generate_signals(last_row):
    is_uptrend = last_row["EMA50"] > last_row["EMA200"] and last_row["EMA21"] > last_row["EMA50"]
    return {"Uptrend (21>50>200 EMA)": is_uptrend, "Bullish Momentum (RSI > 50)": last_row["RSI"] > 50,
            "MACD Bullish (Diff > 0)": last_row["MACD_diff"] > 0, "Volume Spike (>1.5x Avg)": last_row["Volume"] > last_row["Vol_Avg_50"] * 1.5}
def calculate_confidence(scores, weights):
    score = (weights["technical"] * scores["technical"] + weights["sentiment"] * scores["sentiment"] + weights["expert"] * scores["expert"])
    return min(round(score, 2), 100)
def get_recommendation(timeframe, technical_score, overall_confidence):
    if timeframe == "Scalp Trading":
        if technical_score >= 80: return "success", "⚡ Scalp Signal Met — Quick entry momentum is strong."
        return "warning", "🚫 Weak Momentum — Not ideal for scalping."
    if timeframe == "Day Trading":
        if technical_score >= 75: return "success", "📈 Day Trade Setup Found — Confirm with intraday volume."
        return "info", "⏳ Wait for stronger confirmation."
    if timeframe == "Swing Trading":
        if overall_confidence >= 65: return "success", "🌀 Swing Trade Opportunity — Monitor entry zone."
        return "warning", "⚠️ Setup Weak — Wait for more signals."
    if timeframe == "Position Trading":
        if overall_confidence >= 70: return "success", "📊 Strong Long-Term Outlook — Position entry viable."
        return "info", "💤 Not enough alignment for long-term."
    return "error", "Unknown strategy."

# === Main Dashboard Function ===
def display_dashboard(ticker, hist, info, params):
    df = calculate_indicators(hist.copy()); last = df.iloc[-1]; signals = generate_signals(last)
    technical_score = sum([1 for fired in signals.values() if fired]) / len(signals) * 100
    scores = {"technical": technical_score, "sentiment": sentiment_score, "expert": expert_score}
    overall_confidence = calculate_confidence(scores, params['weights'])
    
    st.header(f"Analysis for {ticker} ({params['interval']} Interval)")
    rec_type, rec_text = get_recommendation(timeframe, technical_score, overall_confidence)
    if rec_type == "success": st.success(rec_text)
    elif rec_type == "warning": st.warning(rec_text)
    else: st.info(rec_text)
    
    tab1, tab2, tab3 = st.tabs(["📊 Main Analysis", "📰 Headlines & News", "ℹ️ Ticker Info"])
    with tab1:
        col1, col2 = st.columns([1, 2])
        with col1:
            st.subheader("💡 Confidence Score"); st.metric("Overall Confidence", f"{overall_confidence:.0f}/100"); st.progress(overall_confidence / 100)
            st.markdown(f"- **Technical Score:** `{scores['technical']:.0f}` (W: `{params['weights']['technical']*100:.0f}%`)\n"
                        f"- **Sentiment Score:** `{scores['sentiment']:.0f}` (W: `{params['weights']['sentiment']*100:.0f}%`)\n"
                        f"- **Expert Score:** `{scores['expert']:.0f}` (W: `{params['weights']['expert']*100:.0f}%`)")
            st.subheader("✅ Technical Checklist")
            for signal, fired in signals.items(): st.markdown(f"- {'🟢' if fired else '🔴'} {signal}")
            st.subheader("🎯 Key Price Levels")
            resistance = df["High"][-60:].max(); support = df["Low"][-60:].min()
            st.write(f"**Support:** ${support:.2f} | **Resistance:** ${resistance:.2f}")
            st.write(f"**ATR:** {last['ATR']:.3f} | **Suggested SL:** ${last['Close'] - 1.5 * last['ATR']:.2f}")
        with col2:
            st.subheader("📈 Price Chart"); chart_path = f"chart_{ticker}.png"
            ap = [mpf.make_addplot(df.tail(120)[['BB_high', 'BB_low']])]
            mpf.plot(df.tail(120), type='candle', style='yahoo', mav=(21, 50, 200), volume=True, addplot=ap,
                     title=f"{ticker} - {params['interval']} chart", savefig=chart_path)
            st.image(chart_path)
            if os.path.exists(chart_path): os.remove(chart_path)
    with tab2:
        st.subheader(f"📰 Latest News Headlines for {ticker}")
        st.info("Use these headlines to inform your 'Sentiment Rating' selection in the sidebar.")
        for i, headline in enumerate(finviz_data['headlines']): st.markdown(f"{i+1}. {headline}")
    with tab3:
        st.subheader(f"ℹ️ About {info.get('longName', ticker)}"); st.write(f"**Sector:** {info.get('sector', 'N/A')} | **Industry:** {info.get('industry', 'N/A')}")
        st.markdown(f"**Business Summary:**"); st.info(f"{info.get('longBusinessSummary', 'No summary available.')}")

# === Main Script Execution ===
if ticker:
    try:
        hist_data, info_data = get_data(ticker, selected_params['period'], selected_params['interval'])
        if hist_data is None or hist_data.empty:
            st.error(f"Could not fetch data for {ticker} on a {selected_params['interval']} interval.")
        else:
            display_dashboard(ticker, hist_data, info_data, selected_params)
    except Exception as e:
        st.error(f"An error occurred: {e}")
else:
    st.info("Please enter a stock ticker in the sidebar to begin analysis.")