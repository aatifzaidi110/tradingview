#==GoogleAIStudio==
# -*- coding: utf-8 -*-
# vim: set ts=4 sw=4 et:

import streamlit as st
import yfinance as yf
import pandas as pd
import ta
import mplfinance as mpf
import os

# === Page Setup ===
st.set_page_config(page_title="Aatif's Pro Analyzer", layout="wide")
st.title("📈 Aatif's All-in-One Trade Analyzer")

# === SIDEBAR: User Inputs ===
st.sidebar.header("⚙️ Controls")
ticker = st.sidebar.text_input("Enter a Ticker Symbol", value="NVDA").upper()
timeframe = st.sidebar.radio("Choose Trading Style:",
    ["Scalp Trading", "Day Trading", "Swing Trading", "Position Trading"],
    index=2, help="Changes chart interval and analysis focus.")

# Map styles to yfinance intervals and define weights
TIMEFRAME_MAP = {
    "Scalp Trading": {"period": "1d", "interval": "5m", "weights": {"technical": 0.9, "sentiment": 0.1, "expert": 0.0}},
    "Day Trading": {"period": "60d", "interval": "60m", "weights": {"technical": 0.7, "sentiment": 0.2, "expert": 0.1}},
    "Swing Trading": {"period": "1y", "interval": "1d", "weights": {"technical": 0.6, "sentiment": 0.2, "expert": 0.2}},
    "Position Trading": {"period": "5y", "interval": "1wk", "weights": {"technical": 0.4, "sentiment": 0.2, "expert": 0.4}}
}
selected_params = TIMEFRAME_MAP[timeframe]

st.sidebar.header("🧠 Qualitative Scores")
sentiment_score = st.sidebar.slider("Your Sentiment Score (1-100)", 1, 100, 50, help="Your personal feeling about market news, social media, etc.")
expert_score = st.sidebar.slider("Expert Analysis Score (1-100)", 1, 100, 50, help="Your assessment of analyst ratings, fundamental reports, etc.")

# === Caching & Utility Functions ===
@st.cache_data(ttl=60) # Shorter TTL for intraday data
def get_data(symbol, period, interval):
    stock = yf.Ticker(symbol)
    hist = stock.history(period=period, interval=interval, auto_adjust=True) # auto_adjust=True simplifies column names
    if hist.empty:
        return None, None
    info = stock.info
    return hist, info

def calculate_indicators(df):
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

def generate_signals(last_row):
    is_uptrend = last_row["EMA50"] > last_row["EMA200"] and last_row["EMA21"] > last_row["EMA50"]
    signals = {
        "Uptrend (21>50>200 EMA)": is_uptrend,
        "Bullish Momentum (RSI > 50)": last_row["RSI"] > 50,
        "MACD Bullish (Diff > 0)": last_row["MACD_diff"] > 0,
        "Volume Spike (>1.5x Avg)": last_row["Volume"] > last_row["Vol_Avg_50"] * 1.5,
    }
    return signals

def calculate_confidence(scores, weights):
    score = (
        weights["technical"] * scores["technical"] +
        weights["sentiment"] * scores["sentiment"] +
        weights["expert"] * scores["expert"]
    )
    return min(round(score, 2), 100) # Cap score at 100

def get_recommendation(timeframe, technical_score, overall_confidence):
    """Provides a tailored recommendation based on scores and trading style."""
    if timeframe == "Scalp Trading":
        if technical_score >= 80:
            return "success", "⚡ Scalp Signal Met — Quick entry momentum is strong."
        return "warning", "🚫 Weak Momentum — Not ideal for scalping. Wait for alignment."
    elif timeframe == "Day Trading":
        if technical_score >= 75:
            return "success", "📈 Day Trade Setup Found — Confirm with intraday volume."
        return "info", "⏳ Wait for stronger confirmation or volume spike."
    elif timeframe == "Swing Trading":
        if overall_confidence >= 65:
            return "success", "🌀 Swing Trade Opportunity — Monitor entry zone."
        return "warning", "⚠️ Setup Weak — Wait until more signals fire."
    elif timeframe == "Position Trading":
        if overall_confidence >= 70:
            return "success", "📊 Strong Long-Term Outlook — Position entry viable."
        return "info", "💤 Not enough alignment for long-term entry."
    return "error", "Unknown strategy."

# === Main Dashboard Function ===
def display_dashboard(ticker, hist, info, params):
    df = calculate_indicators(hist.copy())
    last = df.iloc[-1]
    signals = generate_signals(last)

    # --- Scoring ---
    technical_score = sum([1 for fired in signals.values() if fired]) / len(signals) * 100
    scores = {"technical": technical_score, "sentiment": sentiment_score, "expert": expert_score}
    overall_confidence = calculate_confidence(scores, params['weights'])

    # --- UI LAYOUT ---
    st.header(f"Analysis for {ticker} ({params['interval']} Interval)")

    # Recommendation Box
    rec_type, rec_text = get_recommendation(timeframe, technical_score, overall_confidence)
    if rec_type == "success": st.success(rec_text)
    elif rec_type == "warning": st.warning(rec_text)
    elif rec_type == "info": st.info(rec_text)
    else: st.error(rec_text)

    # --- Create Tabs for Organization ---
    tab1, tab2, tab3 = st.tabs(["📊 Main Analysis", "📘 Indicator Guide", "ℹ️ Ticker Info"])

    with tab1:
        col1, col2 = st.columns([1, 2])
        with col1:
            st.subheader("💡 Confidence Score")
            st.metric("Overall Confidence", f"{overall_confidence:.0f}/100")
            st.progress(overall_confidence / 100)
            
            st.markdown(f"""
            - **Technical Score:** `{scores['technical']:.0f}` (Weight: `{params['weights']['technical']*100:.0f}%`)
            - **Sentiment Score:** `{scores['sentiment']:.0f}` (Weight: `{params['weights']['sentiment']*100:.0f}%`)
            - **Expert Score:** `{scores['expert']:.0f}` (Weight: `{params['weights']['expert']*100:.0f}%`)
            """)
            
            st.subheader("✅ Technical Checklist")
            for signal, fired in signals.items():
                st.markdown(f"- {'🟢' if fired else '🔴'} {signal}")
            
            st.subheader("🎯 Key Price Levels")
            resistance = df["High"][-60:].max()
            support = df["Low"][-60:].min()
            st.write(f"**Support (60-period):** ${support:.2f}")
            st.write(f"**Resistance (60-period):** ${resistance:.2f}")
            st.write(f"**ATR (Volatility):** {last['ATR']:.3f}")
            st.write(f"**Suggested Stop-Loss:** ${last['Close'] - 1.5 * last['ATR']:.2f}")

        with col2:
            st.subheader("📈 Price Chart")
            chart_path = f"chart_{ticker}.png"
            ap = [mpf.make_addplot(df.tail(120)[['BB_high', 'BB_low']])]
            mpf.plot(df.tail(120), type='candle', style='yahoo',
                     mav=(21, 50, 200), volume=True, addplot=ap,
                     title=f"{ticker} - {params['interval']} chart", savefig=chart_path)
            st.image(chart_path)
            if os.path.exists(chart_path):
                os.remove(chart_path)

    with tab2:
        st.subheader("📘 Indicator Guide")
        st.markdown("This table explains each indicator, its current value, and its ideal state for a bullish trade.")
        
        indicator_data = [
            {"Indicator": "EMA Stack (21, 50, 200)", 
             "Description": "Shows trend alignment. A stacked formation (short > mid > long) confirms a strong, healthy uptrend.",
             "Current Value": f"21: {last['EMA21']:.2f} | 50: {last['EMA50']:.2f} | 200: {last['EMA200']:.2f}",
             "Ideal Bullish Value": "21 > 50 > 200",
             "Status": '🟢' if signals["Uptrend (21>50>200 EMA)"] else '🔴'},
            {"Indicator": "RSI (14)",
             "Description": "Measures momentum. For a bullish trade, we want to see the RSI in the upper half of its range, confirming buyer strength.",
             "Current Value": f"{last['RSI']:.2f}",
             "Ideal Bullish Value": "> 50",
             "Status": '🟢' if signals["Bullish Momentum (RSI > 50)"] else '🔴'},
            {"Indicator": "MACD Difference",
             "Description": "Highlights momentum direction. A positive value means short-term momentum is stronger than long-term.",
             "Current Value": f"{last['MACD_diff']:.2f}",
             "Ideal Bullish Value": "> 0",
             "Status": '🟢' if signals["MACD Bullish (Diff > 0)"] else '🔴'},
            {"Indicator": "Volume",
             "Description": "Confirms conviction. A volume spike on an up-day shows strong institutional interest.",
             "Current Value": f"{last['Volume']:,.0f}",
             "Ideal Bullish Value": f"> {last['Vol_Avg_50']:,.0f} (Avg)",
             "Status": '🟢' if signals["Volume Spike (>1.5x Avg)"] else '🔴'}
        ]
        st.table(pd.DataFrame(indicator_data).set_index("Indicator"))

    with tab3:
        st.subheader(f"ℹ️ About {info.get('longName', ticker)}")
        st.write(f"**Sector:** {info.get('sector', 'N/A')} | **Industry:** {info.get('industry', 'N/A')}")
        st.write(f"**Website:** {info.get('website', 'N/A')}")
        st.markdown(f"**Business Summary:**")
        st.info(f"{info.get('longBusinessSummary', 'No summary available.')}")
        st.markdown(f"🔗 [Finviz](https://finviz.com/quote.ashx?t={ticker}) | [Barchart](https://www.barchart.com/stocks/quotes/{ticker}/overview) | [TipRanks](https://www.tipranks.com/stocks/{ticker}/forecast)")

# === Main Script Execution ===
if ticker:
    try:
        hist_data, info_data = get_data(ticker, selected_params['period'], selected_params['interval'])
        if hist_data is None or hist_data.empty:
            st.error(f"Could not fetch data for {ticker} on a {selected_params['interval']} interval. This may be an invalid symbol or the interval may not be supported for this asset.")
        else:
            display_dashboard(ticker, hist_data, info_data, selected_params)
    except Exception as e:
        st.error(f"An error occurred: {e}")
        st.warning("This may be due to an invalid ticker, unsupported interval, or yfinance API rate limits. Please wait a moment and try again.")
else:
    st.info("Please enter a stock ticker in the sidebar to begin analysis.")