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

    # === Technical Passes ===
    bull_ema_stack = last["EMA21"] > last["EMA50"] > last["EMA200"]
    bearish_ema_stack = last["EMA21"] < last["EMA50"] < last["EMA200"]
    macd_bullish = last["MACD_diff"] > 0
    macd_bearish = last["MACD_diff"] < 0
    price_below_bb = price < last["BB_low"]
    price_above_bb = price > last["BB_high"]
    rsi_entry = last["RSI"] < 30 or last["RSI"] > 50
    rsi_exit = last["RSI"] > 85
    atr_breakout = price > previous_close + last["ATR"]
    atr_breakdown = price < previous_close - last["ATR"]
    volume_spike = last["Volume"] > last["Vol_Avg"] * 1.5

    # === Strategy Recommendation ===
    entry_conditions = [price_below_bb, bull_ema_stack, macd_bullish, atr_breakout]
    exit_conditions = [rsi_exit, bearish_ema_stack, macd_bearish, atr_breakdown]

    entry_trigger = all(entry_conditions)
    exit_trigger = all(exit_conditions)
    stop_loss = round(price - last["ATR"], 2)

    # === Support / Resistance ===
    support = df["Low"].rolling(20).min().iloc[-1]
    resistance = df["High"].rolling(20).max().iloc[-1]

    # === Chart Snapshot ===
    st.subheader("🖼️ Chart Snapshot")
    chart_path = "chart.png"
    mpf.plot(df[-60:], type='candle', mav=(21,50,200), volume=True, style='yahoo', savefig=chart_path)
    st.image(chart_path, caption=f"{ticker.upper()} - Last 60 Days")

    # === Indicator Display ===
    st.subheader("📊 Technical Breakdown")
    st.write(f"Price: ${price:.2f} | Previous Close: ${previous_close:.2f}")
    st.write(f"- Bollinger Lower Band: ${last['BB_low']:.2f} {status(price_below_bb)}")
    st.write(f"- RSI: {last['RSI']:.2f} (entry zone <30, exit >85) {status(rsi_entry)}")
    st.write(f"- MACD Diff: {last['MACD_diff']:.2f} {status(macd_bullish)}")
    st.write(f"- EMA Stack: 21>{round(last['EMA21'],2)} > 50>{round(last['EMA50'],2)} > 200>{round(last['EMA200'],2)} {status(bull_ema_stack)}")
    st.write(f"- ATR: {round(last['ATR'],2)} → Breakout: {status(atr_breakout)}")
    st.write(f"- Volume: {last['Volume']:.0f} vs Avg(50): {last['Vol_Avg']:.0f} {status(volume_spike)}")

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

    # === Overall Confidence Score ===
    st.subheader("🧠 Overall Confidence Score")
    technical_score = 70 if entry_trigger else 50
    sentiment_score = 15  # Placeholder for NLP logic
    expert_score = 15     # Placeholder for analyst consensus logic
    overall_confidence = round(technical_score + sentiment_score + expert_score, 2)
    st.write(f"Confidence Level: **{overall_confidence}/100**")
    st.progress(overall_confidence)

    # === Simulated Backtest ===
    st.subheader("🔁 Simulated Backtest (Last 30 Bars)")
    recent = df[-30:]
    trades, in_trade, entry_price = [], False, 0

    for i in range(len(recent)):
        row = recent.iloc[i]
        cond_entry = row["Close"] < row["BB_low"] and row["EMA21"] > row["EMA50"] > row["EMA200"] and row["MACD_diff"] > 0
        cond_exit = row["RSI"] > 85 and row["MACD_diff"] < 0

        if cond_entry and not in_trade:
            entry_price = row["Open"]
            in_trade = True
        elif cond_exit and in_trade:
            exit_price = row["Open"]
            trades.append(exit_price - entry_price)
            in_trade = False

    if trades:
        win_rate = round(100 * len([t for t in trades if t > 0]) / len(trades), 2)
        avg_pl = round(np.mean(trades), 2)
        st.write(f"Trades: {len(trades)} | Win Rate: {win_rate}% | Avg P/L: ${avg_pl}")
    else:
        st.write("No trades triggered in last 30 bars.")

    # === Indicator Glossary ===
    with st.expander("📘 Indicator Glossary & Strategy Guide"):
        st.markdown("""
        - **RSI (Relative Strength Index)**: Momentum oscillator.  
          *Ideal:*"")