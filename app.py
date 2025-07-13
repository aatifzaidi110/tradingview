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
    
# === Technical Analysis Table ===
st.subheader("📊 Technical Indicator Breakdown")

def color_status(flag):
    return "🟢 Green" if flag else "🔴 Red"

st.markdown(f"""
| **Indicator**     | **Current Value ({ticker.upper()})** | **Meaning & Ideal Range**                                               | **Status** |
|-------------------|--------------------------|------------------------------------------------------------------------|------------|
| **RSI**           | {last['RSI']:.2f}         | Momentum. Ideal: <30 for entry, >85 signals overbought                | {color_status(rsi_entry and last['RSI'] < 85)} |
| **MACD Diff**     | {last['MACD_diff']:.2f}   | Trend momentum. Ideal: >0 confirms bullish crossover                   | {color_status(macd_bullish)} |
| **EMA Stack**     | 21>{round(last['EMA21'],2)} > 50>{round(last['EMA50'],2)} > 200>{round(last['EMA200'],2)} | Trend strength. Ideal: EMA21 > EMA50 > EMA200                          | {color_status(bull_ema_stack)} |
| **ATR Breakout**  | {round(last['ATR'],2)}    | Volatility. Ideal: Price > Prev Close + ATR                            | {color_status(atr_breakout)} |
| **Volume Spike**  | {last['Volume']:.0f} vs Avg(50): {last['Vol_Avg']:.0f} | Interest. Ideal: Volume > 1.5× 50-day average                          | {color_status(volume_spike)} |
| **Bollinger Band**| Price < ${last['BB_low']:.2f} | Volatility zone. Ideal: Entry if price below lower band               | {color_status(price_below_bb)} |
""")


    # === Technical Passes ===
    bull_ema_stack = last["EMA21"] > last["EMA50"] > last["EMA200"]
    macd_bullish = last["MACD_diff"] > 0
    price_below_bb = price < last["BB_low"]
    rsi_entry = last["RSI"] < 30 or last["RSI"] > 50
    rsi_exit = last["RSI"] > 85
    atr_breakout = price > previous_close + last["ATR"]
    atr_breakdown = price < previous_close - last["ATR"]
    volume_spike = last["Volume"] > last["Vol_Avg"] * 1.5

    entry_conditions = [price_below_bb, bull_ema_stack, macd_bullish, atr_breakout]
    exit_conditions = [rsi_exit, not bull_ema_stack, not macd_bullish, atr_breakdown]

    entry_trigger = all(entry_conditions)
    exit_trigger = all(exit_conditions)
    stop_loss = round(price - last["ATR"], 2)

    support = df["Low"].rolling(20).min().iloc[-1]
    resistance = df["High"].rolling(20).max().iloc[-1]

    # === Chart Snapshot ===
    st.subheader("🖼️ Chart Snapshot")
    chart_path = "chart.png"
    mpf.plot(df[-60:], type='candle', mav=(21,50,200), volume=True, style='yahoo', savefig=chart_path)
    st.image(chart_path, caption=f"{ticker.upper()} - Last 60 Days")

    # === Overall Confidence Score ===
    st.subheader("🧠 Overall Confidence Score")
    technical_score = 70 if entry_trigger else 50
    sentiment_score = 15
    expert_score = 15
    overall_confidence = round(technical_score + sentiment_score + expert_score, 2)
    st.write(f"Confidence Level: **{overall_confidence}/100**")
    st.progress(overall_confidence)

    # === Indicator Glossary ===
    with st.expander("📘 Indicator Glossary & Strategy Guide"):
        st.markdown("""
        - **RSI (Relative Strength Index)**: Momentum oscillator.  
          - *Ideal:* <30 for entry, >85 for exit
        - **MACD**: Trend confirmation.  
          - *Ideal:* Positive MACD diff indicates bullish crossover
        - **EMA Stack (21/50/200)**: Trend strength.  
          - *Ideal:* EMA21 > EMA50 > EMA200 = strong bullish alignment
        - **ATR (Average True Range)**: Measures volatility.  
          - *Ideal:* Price movement beyond ATR suggests breakout
        - **Bollinger Bands**: Identifies volatility zones.  
          - *Ideal:* Price near or below lower band may signal bounce
        - **Volume Spike**: Validates breakout.  
          - *Ideal:* Volume > 1.5× average confirms interest
        """)

    # === Strategy Logic ===
    st.subheader("🎯 Strategy Logic")
    if entry_trigger:
        st.success("✅ Entry Signal Met")
        st.write(f"Suggested Stop Loss: ${stop_loss}")
    elif exit_trigger:
        st.warning("⚠️ Exit Signal Triggered")
    else:
        st.info("⏳ Watchlist Candidate — No Trade Signal")

    # === Support / Resistance ===
    st.subheader("📈 Support & Resistance")
    st.write(f"- Nearest Support: ${support:.2f}")
    st.write(f"- Nearest Resistance: ${resistance:.2f}")

    # === Earnings / Dividends ===
    st.subheader("🗓️ Earnings & Dividends")
    st.write(f"Next Earnings: {earnings}")
    st.write(f"Next Dividend: {dividend}")

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
            "Entry Trigger": entry_trigger,
            "Exit Trigger": exit_trigger,
            "Stop Loss": stop_loss,
            "Notes": note
        }
        log_df = pd.DataFrame([journal_entry])
        if os.path.exists("journal.csv"):
            log_df.to_csv("journal.csv", mode="a", header=False, index=False)
        else:
            log_df.to_csv("journal.csv", index=False)
        st.success("✅ Trade analysis saved to journal")
