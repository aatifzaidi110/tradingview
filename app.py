import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import ta
import datetime
import mplfinance as mpf

# === CONFIG ===
st.set_page_config(page_title="Aatif's Swing Trade Analyzer", layout="centered")
st.title("📊 Swing Trade Analyzer by Aatif")
ticker = st.text_input("Enter a Ticker Symbol", value="NVDA")

# === FETCH DATA ===
def get_data(symbol):
    stock = yf.Ticker(symbol)
    hist = stock.history(period="6mo")
    info = stock.info
    earnings = info.get("nextEarningsDate", "N/A")
    dividend = info.get("dividendDate", "N/A")
    price = info.get("currentPrice", hist['Close'].iloc[-1])
    previous_close = info.get("previousClose", hist['Close'].iloc[-2])
    return hist, info, earnings, dividend, price, previous_close

if ticker:
    hist, info, earnings, dividend, price, previous_close = get_data(ticker)

    # === INDICATORS ===
    df = hist.copy()
    df['ema21'] = ta.trend.ema_indicator(df['Close'], window=21)
    df['ema50'] = ta.trend.ema_indicator(df['Close'], window=50)
    df['ema200'] = ta.trend.ema_indicator(df['Close'], window=200)
    df['rsi'] = ta.momentum.RSIIndicator(df['Close']).rsi()
    bb = ta.volatility.BollingerBands(df['Close'])
    df['bb_high'] = bb.bollinger_hband()
    df['bb_low'] = bb.bollinger_lband()
    df['atr'] = ta.volatility.AverageTrueRange(df['High'], df['Low'], df['Close']).average_true_range()
    df['macd'] = ta.trend.macd_diff(df['Close'])
    df['volume_avg'] = df['Volume'].rolling(window=50).mean()

    last = df.iloc[-1]
    bull_stack = last['ema21'] > last['ema50'] > last['ema200']
    bear_stack = last['ema21'] < last['ema50'] < last['ema200']
    macd_bull = last['macd'] > 0
    macd_bear = last['macd'] < 0
    atr_breakout = price > previous_close + last['atr']
    atr_breakdown = price < previous_close - last['atr']
    vol_spike = last['Volume'] > last['volume_avg'] * 1.5

    # === STRATEGY LOGIC ===
    entry_pass = (price < last['bb_low']) and bull_stack and macd_bull and atr_breakout
    exit_pass = (last['rsi'] > 85) and bear_stack and macd_bear and atr_breakdown
    stop_loss = round(price - last['atr'], 2)

    # === CHART SNAPSHOT ===
    st.subheader("🖼️ Chart Snapshot")
    mpf.plot(hist[-60:], type='candle', mav=(21,50,200), volume=True, style='yahoo', savefig='chart.png')
    st.image("chart.png", caption=f"{ticker.upper()} - Last 60 Days")

    # === TECHNICALS ===
    st.subheader("📊 Technical Indicators")
    def status(flag): return "✅ Pass" if flag else "❌ Fail"
    st.write(f"Price: ${price:.2f}")
    st.write(f"Previous Close: ${previous_close:.2f}")
    st.write(f"Bollinger Lower Band: ${last['bb_low']:.2f} → {status(price < last['bb_low'])}")
    st.write(f"RSI: {last['rsi']:.2f} → {status(last['rsi'] < 30 or last['rsi'] > 50)}")
    st.write(f"MACD Diff: {last['macd']:.2f} → {status(macd_bull)}")
    st.write(f"EMA Stack: {round(last['ema21'],2)} > {round(last['ema50'],2)} > {round(last['ema200'],2)} → {status(bull_stack)}")
    st.write(f"ATR: {last['atr']:.2f} → Breakout: {status(atr_breakout)}")
    st.write(f"Volume: {last['Volume']:.0f} | Avg(50): {last['volume_avg']:.0f} → {status(vol_spike)}")

    # === STRATEGY SIGNAL ===
    st.subheader("🎯 Strategy Recommendation")
    if entry_pass:
        st.success("✅ Swing Entry Triggered")
        st.write(f"Suggested Stop Loss: ${stop_loss}")
    elif exit_pass:
        st.warning("⚠️ Exit Signal Active")
    else:
        st.info("⏳ No Entry/Exit Signal - Watchlist Candidate")

    # === SUPPORT / RESISTANCE ===
    st.subheader("📈 Support / Resistance")
    support = df['Low'].rolling(window=20).min().iloc[-1]
    resistance = df['High'].rolling(window=20).max().iloc[-1]
    st.write(f"Nearest Support: ${support:.2f}")
    st.write(f"Nearest Resistance: ${resistance:.2f}")

    # === EVENTS ===
    st.subheader("🗓️ Earnings & Dividends")
    st.write(f"Next Earnings Date: {earnings}")
    st.write(f"Dividend Info: {dividend if dividend != 'N/A' else 'No upcoming dividend'}")

    # === SENTIMENT & LINKS ===
    st.subheader("💬 Sentiment & Expert Analysis")
    google_news = f"https://news.google.com/search?q={ticker}+stock"
    finviz = f"https://finviz.com/quote.ashx?t={ticker}"
    barchart = f"https://www.barchart.com/stocks/quotes/{ticker}/overview"
    tipranks = f"https://www.tipranks.com/stocks/{ticker}/forecast"
    yahoo = f"https://finance.yahoo.com/quote/{ticker}"
    st.markdown(f"- [📰 News Headlines]({google_news})")
    st.markdown(f"- [📊 Finviz Overview]({finviz})")
    st.markdown(f"- [📈 Barchart Opinion]({barchart})")
    st.markdown(f"- [🎯 TipRanks Forecast]({tipranks})")
    st.markdown(f"- [💼 Yahoo Finance Summary]({yahoo})")

    # === BACKTESTING (Simulated Logic) ===
    st.subheader("🔁 Backtest Summary (Simulated)")
    recent = df[-30:]
    trades = []
    in_trade = False
    entry_price = 0

    for i in range(len(recent)):
        row = recent.iloc[i]
        cond_entry = row['Close'] < row['bb_low'] and row['ema21'] > row['ema50'] > row['ema200'] and row['macd'] > 0
        cond_exit = row['rsi'] > 85 and row['macd'] < 0

        if cond_entry and not in_trade:
            entry_price = row['Open']
            in_trade = True
        elif cond_exit and in_trade:
            exit_price = row['Open']
            trades.append(exit_price - entry_price)
            in_trade = False

    if trades:
        win_rate = round(100 * len([p for p in trades if p > 0]) / len(trades), 2)
        avg_pl = round(np.mean(trades), 2)
        st.write(f"Trades Simulated: {len(trades)}")
        st.write(f"Win Rate: {win_rate}% → {status(win_rate > 60)}")
        st.write(f"Avg Profit/Loss: ${avg_pl} → {status(avg_pl > 0)}")
    else:
        st.write("No trades triggered in last 30 bars")
