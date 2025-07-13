import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import ta
import mplfinance as mpf
import datetime
import os

# === App Setup ===
st.set_page_config(page_title="Aatif's Swing Dashboard", layout="centered")
st.title("📊 Aatif's Swing Trade Analyzer")

# === Ticker Input ===
ticker = st.text_input("Enter a Ticker Symbol", value="NVDA")

def status(flag):
    return "✅" if flag else "❌"

def color_status(flag):
    return "🟢 Green" if flag else "🔴 Red"

# === Data Fetching ===
def get_data(symbol):
    stock = yf.Ticker(symbol)
    hist = stock.history(period="6mo")
    info = stock.info
    price = info.get("currentPrice", hist["Close"].iloc[-1])
    previous_close = info.get("previousClose", hist["Close"].iloc[-2])
    earnings = info.get("earningsDate", "N/A")
    dividend = info.get("dividendDate", "N/A")
    return hist, info, price, previous_close, earnings, dividend

# === Dashboard Logic ===
if ticker:
    hist, info, price, previous_close, earnings, dividend = get_data(ticker)
    df = hist.copy()

    # === Indicators ===
    df["EMA21"] = ta.trend.ema_indicator(df["Close"], 21)
    df["EMA50"] = ta.trend.ema_indicator(df["Close"], 50)
    df["EMA200"] = ta.trend.ema_indicator(df["Close"], 200)
    df["RSI"] = ta.momentum.RSIIndicator(df["Close"]).rsi()
    bb = ta.volatility.BollingerBands(df["Close"])
    df["BB_high"] = bb.bollinger_hband()
    df["BB_low"] = bb.bollinger_lband()
    df["MACD_diff"] = ta.trend.macd_diff(df["Close"])
    df["ATR"] = ta.volatility.AverageTrueRange(df["High"], df["Low"], df["Close"]).average_true_range()
    df["Vol_Avg"] = df["Volume"].rolling(50).mean()

    last = df.iloc[-1]

    # === Technical Signals ===
    signals = {
        "RSI": last["RSI"] < 30 or last["RSI"] > 50,
        "MACD": last["MACD_diff"] > 0,
        "EMA": last["EMA21"] > last["EMA50"] > last["EMA200"],
        "ATR": price > previous_close + last["ATR"],
        "Volume": last["Volume"] > last["Vol_Avg"] * 1.5,
        "BB": price < last["BB_low"]
    }

    weights = {
        "RSI": 20,
        "MACD": 20,
        "EMA": 15,
        "ATR": 15,
        "Volume": 15,
        "BB": 15
    }

    technical_score = sum([weights[key] for key in signals if signals[key]])

    sentiment_score = 10  # Temporary placeholder
    expert_score = 10     # Temporary placeholder

    overall_confidence = round(
        0.6 * technical_score +
        0.2 * sentiment_score +
        0.2 * expert_score,
        2
    )

    stop_loss = round(price - last["ATR"], 2)
    support = df["Low"].rolling(20).min().iloc[-1]
    resistance = df["High"].rolling(20).max().iloc[-1]

    # === Chart Snapshot ===
    st.subheader("🖼️ Chart Snapshot")
    chart_path = "chart.png"
    mpf.plot(df[-60:], type='candle', mav=(21, 50, 200), volume=True, style='yahoo', savefig=chart_path)
    st.image(chart_path, caption=f"{ticker.upper()} - Last 60 Days")

    # === Recommended Timeframes ===
    with st.expander("🕰️ Recommended Chart Timeframes by Strategy"):
        st.markdown("""
        - **Scalp Trading** → 1-min or 5-min for precision  
        - **Day Trading** → 15-min to 1-hour for intraday setups  
        - **Swing Trading** → 1-day charts for multi-session trends  
        - **Position Trades / Investing** → Weekly charts for macro view  
        """)

    # === Overall Confidence Score ===
    st.subheader("🧠 Overall Confidence Score")
    st.write(f"Confidence Level: **{overall_confidence}/100**")
    st.progress(overall_confidence)

    # === Technical Indicator Table ===
    st.subheader("📊 Technical Indicator Breakdown")
    st.markdown(f"""
| **Indicator**     | **Current Value ({ticker.upper()})** | **Meaning & Ideal Range**                                               | **Status** |
|-------------------|--------------------------|------------------------------------------------------------------------|------------|
| **RSI**           | {last['RSI']:.2f}         | Momentum. Ideal: <30 entry, >85 exit                                  | {color_status(signals["RSI"])} |
| **MACD Diff**     | {last['MACD_diff']:.2f}   | Bullish momentum. Ideal: >0                                            | {color_status(signals["MACD"])} |
| **EMA Stack**     | 21>{last['EMA21']:.2f} > 50>{last['EMA50']:.2f} > 200>{last['EMA200']:.2f} | Trend alignment. Ideal: 21 > 50 > 200                     | {color_status(signals["EMA"])} |
| **ATR Breakout**  | {last['ATR']:.2f}         | Volatility. Ideal: Price > Prev Close + ATR                           | {color_status(signals["ATR"])} |
| **Volume Spike**  | {last['Volume']:.0f} vs Avg(50): {last['Vol_Avg']:.0f} | Activity. Ideal: Volume > 1.5× average               | {color_status(signals["Volume"])} |
| **Bollinger Band**| Price < ${last['BB_low']:.2f} | Entry zone. Ideal: Bounce near lower band                         | {color_status(signals["BB"])} |
""")

    # === Indicator Glossary ===
    with st.expander("📘 Indicator Glossary & Strategy Guide"):
        st.markdown("""
- **RSI (Relative Strength Index)**: Momentum indicator  
- **MACD Diff**: Trend momentum signal  
- **EMA Stack**: Measures trend strength  
- **ATR (Average True Range)**: Shows volatility boundaries  
- **Bollinger Bands**: Entry zone indicators  
- **Volume Spike**: Validates interest behind breakout  
        """)

    # === Strategy Recommendations ===
    st.subheader("🎯 Strategy Recommendation")
    if technical_score >= 80:
        st.success("✅ Entry Signal Met — High Conviction")
        st.write(f"Suggested Stop Loss: ${stop_loss}")
    elif technical_score >= 60:
        st.info("⏳ Watchlist Setup — Wait for Confirmation")
    else:
        st.warning("🚫 Weak Signal — Avoid or Monitor")

    # === Support / Resistance ===
    st.subheader("📈 Support & Resistance")
    st.write(f"- Support: ${support:.2f}")
    st.write(f"- Resistance: ${resistance:.2f}")

    # === Sentiment & Expert Links ===
    st.subheader("💬 Sentiment & Expert Scores")
    google_news = f"https://news.google.com/search?q={ticker}+stock"
    finviz = f"https://finviz.com/quote.ashx?t={ticker}"
    barchart = f"https://www.barchart.com/stocks/quotes/{ticker}/overview"
    tipranks = f"https://www.tipranks.com/stocks/{ticker}/forecast"
    st.markdown(f"- [📰 Google News]({google_news})")
    st.markdown(f"- [📊 Finviz]({finviz})")
    st.markdown(f"- [📈 Barchart]({barchart})")
    st.markdown(f"- [🎯 TipRanks]({tipranks})")

    # === Journaling Module ===
    st.subheader("📝 Trade Journal")
    note = st.text_area("Add notes or rationale for this analysis:")

    if st.button("Log Analysis"):
        timestamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M")
        journal_entry = {
            "Timestamp": timestamp,
            "Ticker": ticker.upper(),
            "Confidence": overall_confidence,
            "Technical Score": technical_score,
            "Sentiment Score": sentiment_score,
            "Expert Score": expert_score,
            "Stop Loss": stop_loss,
            "Notes": note
        }
        log_df = pd.DataFrame([journal_entry])
        if os.path.exists("journal.csv"):
            log_df.to_csv("journal.csv", mode="a", header=False, index=False)
        else:
