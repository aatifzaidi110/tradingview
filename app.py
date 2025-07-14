import streamlit as st
import yfinance as yf
import pandas as pd
import ta
import mplfinance as mpf

st.set_page_config(page_title="Aatif's Swing Dashboard", layout="centered")
st.title("📊 Aatif's Swing Trade Analyzer")

# === Ticker Input ===
ticker = st.text_input("Enter a Ticker Symbol", value="NVDA")

# === Utility Functions ===
def status(flag): return "✅" if flag else "❌"
def color_status(flag): return "🟢 Green" if flag else "🔴 Red"

# === Caching ===
@st.cache_data(ttl=600)
def get_data(symbol):
    stock = yf.Ticker(symbol)
    hist = stock.history(period="6mo")
    info = stock.info
    price = info.get("currentPrice", hist["Close"].iloc[-1])
    previous_close = info.get("previousClose", hist["Close"].iloc[-2])
    earnings = info.get("earningsDate", "N/A")
    dividend = info.get("dividendDate", "N/A")
    return hist, info, price, previous_close, earnings, dividend

# === Dashboard ===
if ticker:
    try:
        hist, info, price, previous_close, earnings, dividend = get_data(ticker)
    except Exception:
        st.error("⚠️ Data fetch failed. Wait for Yahoo Finance rate limits to reset.")
        if st.button("🔄 Retry"):
            st.experimental_rerun()
        st.stop()

    df = hist.copy()
    df["EMA21"] = ta.trend.ema_indicator(df["Close"], 21)
    df["EMA50"] = ta.trend.ema_indicator(df["Close"], 50)
    df["EMA200"] = ta.trend.ema_indicator(df["Close"], 200)
    df["RSI"] = ta.momentum.RSIIndicator(df["Close"]).rsi()
    bb = ta.volatility.BollingerBands(df["Close"])
    df["BB_low"] = bb.bollinger_lband()
    df["MACD_diff"] = ta.trend.macd_diff(df["Close"])
    df["ATR"] = ta.volatility.AverageTrueRange(df["High"], df["Low"], df["Close"]).average_true_range()
    df["Vol_Avg"] = df["Volume"].rolling(50).mean()
    last = df.iloc[-1]

    # === Signals & Scores ===
    signals = {
        "RSI": last["RSI"] < 30 or last["RSI"] > 50,
        "MACD": last["MACD_diff"] > 0,
        "EMA": last["EMA21"] > last["EMA50"] > last["EMA200"],
        "ATR": price > previous_close + last["ATR"],
        "Volume": last["Volume"] > last["Vol_Avg"] * 1.5,
        "BB": price < last["BB_low"]
    }
    weights = {"RSI": 20, "MACD": 20, "EMA": 15, "ATR": 15, "Volume": 15, "BB": 15}
    technical_score = sum([weights[k] for k in signals if signals[k]])
    sentiment_score = 10
    expert_score = 10
    overall_confidence = round(0.6 * technical_score + 0.2 * sentiment_score + 0.2 * expert_score, 2)

    stop_loss = round(price - last["ATR"], 2)
    support = df["Low"].rolling(20).min().iloc[-1]
    resistance = df["High"].rolling(20).max().iloc[-1]

    # === Overview Panel ===
    st.subheader(f"📌 {ticker.upper()} Overview")
    col1, col2 = st.columns(2)
    with col1:
        st.write(f"**Description:** {info.get('longBusinessSummary', 'N/A')[:300]}...")
        st.write(f"**Current Price:** ${price:.2f}")
        st.write(f"**Previous Close:** ${previous_close:.2f}")
        st.write(f"**Earnings Date:** {earnings}")
        st.write(f"**Dividend Date:** {dividend}")
    with col2:
        st.write(f"**Support Level:** ${support:.2f}")
        st.write(f"**Resistance Level:** ${resistance:.2f}")
        st.markdown(f"- [📰 Google News](https://news.google.com/search?q={ticker}+stock)")
        st.markdown(f"- [📊 Finviz](https://finviz.com/quote.ashx?t={ticker})")
        st.markdown(f"- [📈 Barchart](https://www.barchart.com/stocks/quotes/{ticker}/overview)")
        st.markdown(f"- [🎯 TipRanks](https://www.tipranks.com/stocks/{ticker}/forecast)")

    # === Chart Snapshot
    st.subheader("🖼️ Chart Snapshot")
    chart_path = "chart.png"
    mpf.plot(df[-60:], type='candle', mav=(21, 50, 200), volume=True, style='yahoo', savefig=chart_path)
    st.image(chart_path, caption=f"{ticker.upper()} - Last 60 Days")

    # === Sentiment & Expert Panel ===
    st.subheader("🧠 Sentiment & Expert Analysis")
    col1, col2 = st.columns(2)
    with col1:
        st.write(f"**Sentiment Score:** {sentiment_score}/100")
        st.markdown(f"- [📰 Latest News](https://news.google.com/search?q={ticker}+stock)")
        st.markdown(f"- [🗂 Finviz Headlines](https://finviz.com/quote.ashx?t={ticker})")
    with col2:
        st.write(f"**Expert Score:** {expert_score}/100")
        st.markdown(f"- [📈 TipRanks](https://www.tipranks.com/stocks/{ticker}/forecast)")
        st.markdown(f"- [📊 Barchart Summary](https://www.barchart.com/stocks/quotes/{ticker}/overview)")

    # === Confidence Summary
    st.subheader("🧮 Confidence Breakdown")
    st.write(f"""
- **Technical Score:** {technical_score}/100  
- **Sentiment Score:** {sentiment_score}/100  
- **Expert Score:** {expert_score}/100  
- ➡️ **Overall Confidence:** **{overall_confidence}/100**
""")
    st.progress(overall_confidence / 100)

    # === Technical Indicator Table with Full Descriptions
    st.subheader("📊 Technical Indicator Breakdown")
    st.markdown(f"""
| **Indicator**     | **Current Value ({ticker.upper()})** | **Meaning & Ideal Range**                                                                                 | **Status** |
|-------------------|--------------------------|------------------------------------------------------------------------------------------------------------|------------|
| **RSI**           | {last['RSI']:.2f}         | Measures overbought/oversold momentum. Ideal: <30 = oversold, >70 = overbought.                          | {color_status(signals["RSI"])} |
| **MACD Diff**     | {last['MACD_diff']:.2f}   | Indicates trend direction & strength. Ideal: >0 for bullish momentum.                                     | {color_status(signals["MACD"])} |
| **EMA Stack**     | 21>{last['EMA21']:.2f} > 50>{last['EMA50']:.2f} > 200>{last['EMA200']:.2f} | Alignment of short- to long-term trends. Ideal: 21 > 50 > 200.         | {color_status(signals["EMA"])} |
| **ATR Breakout**  | {last['ATR']:.2f}         | Gauges volatility. Ideal: price > previous close + ATR for a breakout.                                    | {color_status(signals["ATR"])} |
| **Volume Spike**  | {last['Volume']:.0f} vs Avg(50): {last['Vol_Avg']:.0f} | Highlights interest. Ideal: volume > 1.5× average for strong moves.      | {color_status(signals["Volume"])} |
| **Bollinger Band**| Price < ${last['BB_low']:.2f} | Shows price extremes. Ideal: near lower band may suggest bounce.            | {color_status(signals["BB"])} |
""")

  # === Timeframe Selector
st.subheader("🕰️ Chart Timeframe Selector (Temporarily Disabled)")
st.info("Intraday charting is currently disabled to avoid Yahoo rate limit errors. Full multi-timeframe support will return soon.")

timeframe = st.radio("Choose your trading style:", [
    "Swing Trading (1D)",
    "Day Trading (1H)",
    "Scalp Trading (5Min)",
    "Position Trading (1W)"
])

# === Strategy Tagging
strategy_map = {
    "Swing Trading (1D)": "Swing Trade",
    "Day Trading (1H)": "Day Trade",
    "Scalp Trading (5Min)": "Scalp Trade",
    "Position Trading (1W)": "Position Trade"
}
selected_strategy = strategy_map.get(timeframe, "Unknown")
st.write(f"📌 **Strategy Type Selected:** {selected_strategy}")

# === Adaptive Recommendation
st.subheader("🎯 Strategy Recommendation")

if selected_strategy == "Scalp Trade":
    if technical_score >= 85:
        st.success("⚡ Scalp Signal Met — Quick Entry Suggested")
        st.write(f"Suggested Stop Loss: ${stop_loss}")
    else:
        st.warning("🚫 Weak Momentum — Not Ideal for Scalping")
elif selected_strategy == "Day Trade":
    if technical_score >= 80:
        st.success("📈 Day Trade Setup Found — Confirm with Intraday Flow")
        st.write(f"Suggested Stop Loss: ${stop_loss}")
    else:
        st.info("⏳ Wait for Intraday Confirmation or Volume Spike")
elif selected_strategy == "Swing Trade":
    if technical_score >= 75:
        st.success("🌀 Swing Trade Opportunity — Monitor Entry Zone")
        st.write(f"Suggested Stop Loss: ${stop_loss}")
    else:
        st.warning("⚠️ Setup Weak — Consider Watching Until More Signals Fire")
elif selected_strategy == "Position Trade":
    if overall_confidence >= 70:
        st.success("📊 Strong Long-Term Outlook — Position Entry Viable")
        st.write(f"Use Weekly Support: ${support:.2f}")
    else:
        st.info("💤 Not Enough Alignment for Long-Term Entry")
else:
    st.warning("❔ Unknown strategy type — cannot generate recommendation.")
