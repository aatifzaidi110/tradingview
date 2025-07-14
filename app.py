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
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer

# === Page Setup ===
st.set_page_config(page_title="Aatif's AI Analyzer", layout="wide")
st.title("📈 Aatif's Customizable AI Trade Analyzer")

# === SIDEBAR: Controls & Selections ===
st.sidebar.header("⚙️ Controls")
ticker = st.sidebar.text_input("Enter a Ticker Symbol", value="NVDA").upper()
timeframe = st.sidebar.radio("Choose Trading Style:", ["Scalp Trading", "Day Trading", "Swing Trading", "Position Trading"], index=2)

st.sidebar.header("🔧 Technical Indicator Selection")
st.sidebar.info("Select the indicators you want to use for scoring and display.")
indicator_selection = {
    "EMA Trend": st.sidebar.checkbox("EMA Trend (21, 50, 200)", value=True, help="Use the EMA stack for trend scoring and chart display."),
    "RSI Momentum": st.sidebar.checkbox("RSI Momentum", value=True, help="Use RSI > 50 for bullish momentum scoring."),
    "MACD Crossover": st.sidebar.checkbox("MACD Crossover", value=True, help="Use MACD difference for momentum scoring."),
    "Volume Spike": st.sidebar.checkbox("Volume Spike", value=True, help="Use Volume vs. 50-day average for scoring."),
    "Bollinger Bands": st.sidebar.checkbox("Bollinger Bands Display", value=True, help="Display Bollinger Bands on the chart for visual context (not used in scoring).")
}

st.sidebar.header("🧠 Qualitative Scores")
auto_sentiment_score_placeholder = st.sidebar.empty()
sentiment_score = st.sidebar.slider("Adjust Final Sentiment Score", 1, 100, 50)
auto_expert_score_placeholder = st.sidebar.empty()
expert_score = st.sidebar.slider("Adjust Final Expert Score", 1, 100, 50)

# === Core Functions ===
@st.cache_data(ttl=900)
def get_finviz_data(ticker):
    url = f"https://finviz.com/quote.ashx?t={ticker}"; headers = {'User-Agent': 'Mozilla/5.0'}
    try:
        response = requests.get(url, headers=headers); response.raise_for_status()
        soup = BeautifulSoup(response.text, 'html.parser')
        recom_tag = soup.find('td', text='Recom'); analyst_recom = recom_tag.find_next_sibling('td').text if recom_tag else "N/A"
        headlines = [tag.text for tag in soup.findAll('a', class_='news-link-left')[:10]]
        analyzer = SentimentIntensityAnalyzer()
        compound_scores = [analyzer.polarity_scores(h)['compound'] for h in headlines]
        avg_compound = sum(compound_scores) / len(compound_scores) if compound_scores else 0
        return {"recom": analyst_recom, "headlines": headlines, "sentiment_compound": avg_compound}
    except Exception: return {"recom": "N/A", "headlines": [], "sentiment_compound": 0}

EXPERT_RATING_MAP = {"Strong Buy": 100, "Buy": 85, "Hold": 50, "N/A": 50, "Sell": 15, "Strong Sell": 0}
def convert_compound_to_100_scale(compound_score): return int((compound_score + 1) * 50)

# Update sidebar placeholders with automated data
finviz_data = get_finviz_data(ticker)
auto_sentiment_score = convert_compound_to_100_scale(finviz_data['sentiment_compound'])
auto_expert_score = EXPERT_RATING_MAP.get(finviz_data['recom'], 50)
auto_sentiment_score_placeholder.markdown(f"**Automated Sentiment:** `{auto_sentiment_score}` (from headlines)")
auto_expert_score_placeholder.markdown(f"**Automated Expert Rating:** `{auto_expert_score}` ({finviz_data['recom']})")

@st.cache_data(ttl=60)
def get_data(symbol, period, interval):
    stock = yf.Ticker(symbol); hist = stock.history(period=period, interval=interval, auto_adjust=True)
    return (hist, stock.info) if not hist.empty else (None, None)

def calculate_indicators(df):
    # Always calculate all for potential use, but only score selected ones
    df["EMA21"]=ta.trend.ema_indicator(df["Close"],21); df["EMA50"]=ta.trend.ema_indicator(df["Close"],50); df["EMA200"]=ta.trend.ema_indicator(df["Close"],200)
    df["RSI"]=ta.momentum.RSIIndicator(df["Close"]).rsi(); bb=ta.volatility.BollingerBands(df["Close"]); df["BB_low"]=bb.bollinger_lband(); df["BB_high"]=bb.bollinger_hband()
    df["MACD_diff"]=ta.trend.macd_diff(df["Close"]); df["ATR"]=ta.volatility.AverageTrueRange(df["High"],df["Low"],df["Close"]).average_true_range()
    df["Vol_Avg_50"]=df["Volume"].rolling(50).mean(); return df

def generate_signals(last_row, selection):
    """Generates signals ONLY for the user-selected indicators."""
    signals = {}
    if selection.get("EMA Trend"):
        signals["Uptrend (21>50>200 EMA)"] = last_row["EMA50"] > last_row["EMA200"] and last_row["EMA21"] > last_row["EMA50"]
    if selection.get("RSI Momentum"):
        signals["Bullish Momentum (RSI > 50)"] = last_row["RSI"] > 50
    if selection.get("MACD Crossover"):
        signals["MACD Bullish (Diff > 0)"] = last_row["MACD_diff"] > 0
    if selection.get("Volume Spike"):
        signals["Volume Spike (>1.5x Avg)"] = last_row["Volume"] > last_row["Vol_Avg_50"] * 1.5
    return signals

def calculate_confidence(scores, weights):
    score = (weights["technical"] * scores["technical"] + weights["sentiment"] * scores["sentiment"] + weights["expert"] * scores["expert"])
    return min(round(score, 2), 100)

def get_recommendation(timeframe, technical_score, overall_confidence):
    # This logic remains the same, providing advice based on the final scores
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
def display_dashboard(ticker, hist, info, params, selection):
    df = calculate_indicators(hist.copy()); last = df.iloc[-1]
    
    # --- DYNAMIC SCORING based on selection ---
    signals = generate_signals(last, selection)
    fired_signals = sum(1 for fired in signals.values() if fired)
    selected_signals_count = len(signals)
    technical_score = (fired_signals / selected_signals_count) * 100 if selected_signals_count > 0 else 0

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
            st.markdown(f"- **Technical:** `{scores['technical']:.0f}` (W: `{params['weights']['technical']*100:.0f}%`)\n"
                        f"- **Sentiment:** `{scores['sentiment']:.0f}` (W: `{params['weights']['sentiment']*100:.0f}%`)\n"
                        f"- **Expert:** `{scores['expert']:.0f}` (W: `{params['weights']['expert']*100:.0f}%`)")
            
            st.subheader("✅ Technical Checklist")
            if not signals:
                st.warning("No technical indicators selected for scoring.")
            for signal, fired in signals.items(): 
                st.markdown(f"- {'🟢' if fired else '🔴'} {signal}")

            st.subheader("🎯 Key Price Levels")
            resistance = df["High"][-60:].max(); support = df["Low"][-60:].min()
            st.write(f"**Support:** ${support:.2f} | **Resistance:** ${resistance:.2f}")
            st.write(f"**ATR (Volatility):** {last['ATR']:.3f}")

        with col2:
            st.subheader("📈 Price Chart"); chart_path = f"chart_{ticker}.png"
            # --- DYNAMIC CHART PLOTTING ---
            mav_tuple = (21, 50, 200) if selection.get("EMA Trend") else None
            ap = [mpf.make_addplot(df.tail(120)[['BB_high', 'BB_low']])] if selection.get("Bollinger Bands") else None
            mpf.plot(df.tail(120), type='candle', style='yahoo', mav=mav_tuple, volume=True, addplot=ap,
                     title=f"{ticker} - {params['interval']} chart", savefig=chart_path)
            st.image(chart_path)
            if os.path.exists(chart_path): os.remove(chart_path)

    with tab2:
        st.subheader(f"📰 Latest News Headlines for {ticker}")
        st.info("The AI analyzed these headlines to generate the automated sentiment score.")
        for i, headline in enumerate(finviz_data['headlines']): st.markdown(f"{i+1}. {headline}")
            
    with tab3:
        st.subheader(f"ℹ️ About {info.get('longName', ticker)}"); st.write(f"**Sector:** {info.get('sector', 'N/A')} | **Industry:** {info.get('industry', 'N/A')}")
        st.markdown(f"**Business Summary:**"); st.info(f"{info.get('longBusinessSummary', 'No summary available.')}")

# === Main Script Execution ===
# Map styles to yfinance intervals and define weights
TIMEFRAME_MAP = {
    "Scalp Trading": {"period": "1d", "interval": "5m", "weights": {"technical": 0.9, "sentiment": 0.1, "expert": 0.0}},
    "Day Trading": {"period": "60d", "interval": "60m", "weights": {"technical": 0.7, "sentiment": 0.2, "expert": 0.1}},
    "Swing Trading": {"period": "1y", "interval": "1d", "weights": {"technical": 0.6, "sentiment": 0.2, "expert": 0.2}},
    "Position Trading": {"period": "5y", "interval": "1wk", "weights": {"technical": 0.4, "sentiment": 0.2, "expert": 0.4}}
}
selected_params = TIMEFRAME_MAP[timeframe]

if ticker:
    try:
        hist_data, info_data = get_data(ticker, selected_params['period'], selected_params['interval'])
        if hist_data is None or hist_data.empty:
            st.error(f"Could not fetch data for {ticker} on a {selected_params['interval']} interval.")
        else:
            display_dashboard(ticker, hist_data, info_data, selected_params, indicator_selection)
    except Exception as e:
        st.error(f"An error occurred: {e}")
else:
    st.info("Please enter a stock ticker in the sidebar to begin analysis.")