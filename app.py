import streamlit as st
import yfinance as yf
import pandas as pd
import ta
import mplfinance as mpf
import datetime
import os

st.set_page_config(page_title="Aatif's Swing Dashboard", layout="centered")
st.title("📊 Aatif's Swing Trade Analyzer")

ticker = st.text_input("Enter a Ticker Symbol", value="NVDA")

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

@st.cache_data(ttl=600)
def fetch_intraday(symbol, interval, period):
    df = yf.download(symbol, interval=interval, period=period)
    df.index.name = "Date"
    return df.dropna(subset=["Open", "High", "Low", "Close", "Volume"])

# === Dashboard Logic ===
if ticker:
    try:
        hist, info, price, previous_close, earnings, dividend = get_data(ticker)
    except Exception:
        st.error("⚠️ Yahoo Finance rate limit reached. Please wait a few minutes.")
    
    # ✅ Proper conditional block for retry
        if st.button("🔄 Retry"):
            st.experimental_rerun()
    
    # ✅ Stop further execution
        st.stop()
        
    df = hist.copy()

    # === Indicators ===
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

    # === Technical Signals ===
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

    # === Chart Snapshot
    st.subheader("🖼️ Chart Snapshot")
    chart_path = "chart.png"
    mpf.plot(df[-60:], type='candle', mav=(21, 50, 200), volume=True, style='yahoo', savefig=chart_path)
    st.image(chart_path, caption=f"{ticker.upper()} - Last 60 Days")

    # === Chart Timeframe Selector
    st.subheader("🕰️ Select Chart Timeframe")
    timeframe = st.radio("Choose your trading style:", [
        "Swing Trading (1D)", "Day Trading (1H)", "Scalp Trading (5Min)", "Position Trading (1W)"
    ])

    tf_settings = {
        "Swing Trading (1D)": {"interval": "1d", "period": "6mo"},
        "Day Trading (1H)": {"interval": "1h", "period": "5d"},
        "Scalp Trading (5Min)": {"interval": "5m", "period": "1d"},
        "Position Trading (1W)": {"interval": "1wk", "period": "1y"}
    }

    try:
        selected = tf_settings[timeframe]
        intraday = fetch_intraday(ticker, selected["interval"], selected["period"])
        mpf.plot(intraday, type='candle', mav=(21, 50), volume=True, style='yahoo', savefig=chart_path)
        st.image(chart_path, caption=f"{ticker.upper()} — {selected['interval']} View")
    except Exception:
        st.warning("❌ Unable to load intraday chart. Rate limit or data issue.")
        if st.button("🔄 Retry Chart"):
            st.experimental_rerun()

    # === Recommended Timeframes
    with st.expander("🕰️ Recommended Chart Timeframes by Strategy"):
        st.markdown("""
        - **Scalp Trading** → 1-min or 5-min for precision  
        - **Day Trading** → 15-min to 1-hour for intraday setups  
        - **Swing Trading** → 1-day charts for multi-session trends  
        - **Position Trades / Investing** → Weekly charts for macro view  
        """)

    # === Confidence Score
    st.subheader("🧠 Overall Confidence Score")
    st.write(f"Confidence Level: **{overall_confidence}/100**")
    st.progress(overall_confidence / 100)

    # === Technical Breakdown
    st.subheader("📊 Technical Indicator Breakdown")
    st.markdown(f"""
| **Indicator**     | **Current Value ({ticker.upper()})** | **Meaning & Ideal Range**                       | **Status** |
|-------------------|--------------------------|------------------------------------------------|------------|
| **RSI**           | {last['RSI']:.2f}         | Momentum: <30 entry, >85 exit                  | {color_status(signals["RSI"])} |
| **MACD Diff**     | {last['MACD_diff']:.2f}   | Bullish momentum. >0 indicates strength        | {color_status(signals["MACD"])} |
| **EMA Stack**     | 21>{last['EMA21']:.2f} > 50>{last['EMA50']:.2f} > 200>{last['EMA200']:.2f} | Trend alignment: 21 > 50 > 200 | {color_status(signals["EMA"])} |
| **ATR Breakout**  | {last['ATR']:.2f}         | Volatility signal: Price > Previous + ATR      | {color_status(signals["ATR"])} |
| **Volume Spike**  | {last['Volume']:.0f} vs Avg(50): {last['Vol_Avg']:.0f} | >1.5× avg confirms breakout                    | {color_status(signals["Volume"])} |
| **Bollinger Band**| Price < ${last['BB_low']:.2f} | Below lower band = possible bounce            | {color_status(signals["BB"])} |
""")

    # === Strategy Logic
    st.subheader("🎯 Strategy Recommendation")
    if technical_score >= 80:
        st.success("✅ Entry Signal Met — High Conviction")
        st.write(f"Suggested Stop Loss: ${stop_loss}")
    elif technical_score >= 60:
        st.info("⏳ Watchlist Setup — Wait for Confirmation")
    else:
        st.warning("🚫 Weak Signal — Avoid or Monitor")

    # === Support & Resistance
    st.subheader("📈 Support & Resistance")
    st.write(f"- Support: ${support:.2f}")
    st.write(f"- Resistance: ${resistance:.2f}")
