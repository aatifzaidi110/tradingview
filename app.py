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
from datetime import datetime

# === Page Setup ===
st.set_page_config(page_title="Aatif's AI Trading Hub", layout="wide")
st.title("🚀 Aatif's AI-Powered Trading Hub")

# === Constants and Configuration ===
LOG_FILE = "trade_log.csv"

# === SIDEBAR: Controls & Selections ===
st.sidebar.header("⚙️ Controls")
ticker = st.sidebar.text_input("Enter a Ticker Symbol", value="NVDA").upper()
timeframe = st.sidebar.radio("Choose Trading Style:", ["Scalp Trading", "Day Trading", "Swing Trading", "Position Trading"], index=2)

st.sidebar.header("🔧 Technical Indicator Selection")
indicator_selection = {
    "EMA Trend": st.sidebar.checkbox("EMA Trend (21, 50, 200)", value=True),
    "RSI Momentum": st.sidebar.checkbox("RSI Momentum", value=True),
    "MACD Crossover": st.sidebar.checkbox("MACD Crossover", value=True),
    "Volume Spike": st.sidebar.checkbox("Volume Spike", value=True),
    "Bollinger Bands": st.sidebar.checkbox("Bollinger Bands Display", value=True)
}

st.sidebar.header("🧠 Qualitative Scores")
use_automation = st.sidebar.toggle("Enable Automated Scoring", value=True, help="ON: AI scores are used. OFF: Use manual sliders and only the Technical Score will count.")
auto_sentiment_score_placeholder = st.sidebar.empty()
auto_expert_score_placeholder = st.sidebar.empty()

# === Core Functions ===
@st.cache_data(ttl=900)
def get_finviz_data(ticker):
    url = f"https://finviz.com/quote.ashx?t={ticker}"; headers = {'User-Agent': 'Mozilla/5.0'}
    try:
        response = requests.get(url, headers=headers); response.raise_for_status()
        soup = BeautifulSoup(response.text, 'html.parser'); recom_tag = soup.find('td', text='Recom')
        analyst_recom = recom_tag.find_next_sibling('td').text if recom_tag else "N/A"
        headlines = [tag.text for tag in soup.findAll('a', class_='news-link-left')[:10]]
        analyzer = SentimentIntensityAnalyzer()
        compound_scores = [analyzer.polarity_scores(h)['compound'] for h in headlines]
        avg_compound = sum(compound_scores) / len(compound_scores) if compound_scores else 0
        return {"recom": analyst_recom, "headlines": headlines, "sentiment_compound": avg_compound}
    except Exception: return {"recom": "N/A", "headlines": [], "sentiment_compound": 0}

EXPERT_RATING_MAP = {"Strong Buy": 100, "Buy": 85, "Hold": 50, "N/A": 50, "Sell": 15, "Strong Sell": 0}
def convert_compound_to_100_scale(compound_score): return int((compound_score + 1) * 50)

if use_automation:
    finviz_data = get_finviz_data(ticker)
    auto_sentiment_score = convert_compound_to_100_scale(finviz_data['sentiment_compound'])
    auto_expert_score = EXPERT_RATING_MAP.get(finviz_data['recom'], 50)
    auto_sentiment_score_placeholder.markdown(f"**Automated Sentiment:** `{auto_sentiment_score}`")
    auto_expert_score_placeholder.markdown(f"**Automated Expert Rating:** `{auto_expert_score}` ({finviz_data['recom']})")
    sentiment_score = st.sidebar.slider("Adjust Final Sentiment Score", 1, 100, auto_sentiment_score)
    expert_score = st.sidebar.slider("Adjust Final Expert Score", 1, 100, auto_expert_score)
else:
    st.sidebar.info("Automation OFF. Only technical score is used.")
    sentiment_score = 50 # Set to neutral, will be ignored
    expert_score = 50 # Set to neutral, will be ignored
    finviz_data = {"headlines": ["Automation is disabled."]}

@st.cache_data(ttl=60)
def get_data(symbol, period, interval):
    stock = yf.Ticker(symbol); hist = stock.history(period=period, interval=interval, auto_adjust=True)
    return (hist, stock.info) if not hist.empty else (None, None)

def calculate_indicators(df):
    df["EMA21"]=ta.trend.ema_indicator(df["Close"],21); df["EMA50"]=ta.trend.ema_indicator(df["Close"],50); df["EMA200"]=ta.trend.ema_indicator(df["Close"],200)
    df["RSI"]=ta.momentum.RSIIndicator(df["Close"]).rsi(); bb=ta.volatility.BollingerBands(df["Close"]); df["BB_low"]=bb.bollinger_lband(); df["BB_high"]=bb.bollinger_hband()
    df["MACD_diff"]=ta.trend.macd_diff(df["Close"]); df["ATR"]=ta.volatility.AverageTrueRange(df["High"],df["Low"],df["Close"]).average_true_range()
    df["Vol_Avg_50"]=df["Volume"].rolling(50).mean(); return df

def generate_signals(last_row, selection):
    signals = {}
    if selection.get("EMA Trend"): signals["Uptrend (21>50>200 EMA)"] = last_row["EMA50"] > last_row["EMA200"] and last_row["EMA21"] > last_row["EMA50"]
    if selection.get("RSI Momentum"): signals["Bullish Momentum (RSI > 50)"] = last_row["RSI"] > 50
    if selection.get("MACD Crossover"): signals["MACD Bullish (Diff > 0)"] = last_row["MACD_diff"] > 0
    if selection.get("Volume Spike"): signals["Volume Spike (>1.5x Avg)"] = last_row["Volume"] > last_row["Vol_Avg_50"] * 1.5
    return signals

def backtest_strategy(df_historical, selection, atr_multiplier=1.5, reward_risk_ratio=2.0):
    trades = []; in_trade = False
    for i in range(1, len(df_historical) - 1):
        if in_trade:
            if df_historical['Low'].iloc[i] <= stop_loss: trades.append({"Date": df_historical.index[i].strftime('%Y-%m-%d'), "Type": "Exit (Loss)", "Price": stop_loss}); in_trade = False
            elif df_historical['High'].iloc[i] >= take_profit: trades.append({"Date": df_historical.index[i].strftime('%Y-%m-%d'), "Type": "Exit (Win)", "Price": take_profit}); in_trade = False
        if not in_trade:
            row = df_historical.iloc[i-1]
            if pd.isna(row.get('EMA200')): continue
            signals = generate_signals(row, selection)
            if signals and all(signals.values()):
                entry_price = df_historical['Open'].iloc[i]
                stop_loss = entry_price - (row['ATR'] * atr_multiplier); take_profit = entry_price + (row['ATR'] * atr_multiplier * reward_risk_ratio)
                trades.append({"Date": df_historical.index[i].strftime('%Y-%m-%d'), "Type": "Entry", "Price": entry_price}); in_trade = True
    wins = len([t for t in trades if t['Type'] == 'Exit (Win)']); losses = len([t for t in trades if t['Type'] == 'Exit (Loss)'])
    return trades, wins, losses

def display_dashboard(ticker, hist, info, params, selection):
    df = calculate_indicators(hist.copy()); last = df.iloc[-1]
    signals = generate_signals(last, selection)
    
    technical_score = (sum(1 for f in signals.values() if f) / len(signals)) * 100 if signals else 0
    scores = {"technical": technical_score, "sentiment": sentiment_score, "expert": expert_score}
    
    # === FIX: Dynamic Weighting based on Toggle ===
    final_weights = params['weights'].copy()
    if not use_automation:
        final_weights['technical'] = 1.0; final_weights['sentiment'] = 0.0; final_weights['expert'] = 0.0
        scores['sentiment'] = 0; scores['expert'] = 0 # Set to 0 for display
    
    overall_confidence = min(round((final_weights["technical"] * scores["technical"] + final_weights["sentiment"] * scores["sentiment"] + final_weights["expert"] * scores["expert"]), 2), 100)
    
    st.header(f"Analysis for {ticker} ({params['interval']} Interval)")
    
    tab_list = ["📊 Main Analysis", "📈 Trade Plan", "🧪 Backtest", "📰 News & Info", "📝 Trade Log"]
    main_tab, trade_tab, backtest_tab, news_tab, log_tab = st.tabs(tab_list)

    with main_tab:
        col1, col2 = st.columns([1, 2])
        with col1:
            st.subheader("💡 Confidence Score"); st.metric("Overall Confidence", f"{overall_confidence:.0f}/100"); st.progress(overall_confidence / 100)
            st.markdown(f"- **Technical:** `{scores['technical']:.0f}` (W: `{final_weights['technical']*100:.0f}%`)\n- **Sentiment:** `{scores['sentiment']:.0f}` (W: `{final_weights['sentiment']*100:.0f}%`)\n- **Expert:** `{scores['expert']:.0f}` (W: `{final_weights['expert']*100:.0f}%`)")
            st.subheader("🎯 Key Price Levels"); current_price = last['Close']; prev_close = df['Close'].iloc[-2]; price_delta = current_price - prev_close
            st.metric(label="Current Price", value=f"${current_price:.2f}", delta=f"${price_delta:.2f}")
            wk52_high = info.get('fiftyTwoWeekHigh', 'N/A'); wk52_low = info.get('fiftyTwoWeekLow', 'N/A')
            st.write(f"**52W High/Low:** `${wk52_high:.2f} / ${wk52_low:.2f}`" if isinstance(wk52_high, float) else "")
        with col2:
            st.subheader("📈 Price Chart"); chart_path = f"chart_{ticker}.png"
            mav_tuple = (21, 50, 200) if selection.get("EMA Trend") else None
            ap = [mpf.make_addplot(df.tail(120)[['BB_high', 'BB_low']])] if selection.get("Bollinger Bands") else None
            mpf.plot(df.tail(120), type='candle', style='yahoo', mav=mav_tuple, volume=True, addplot=ap, title=f"{ticker} - {params['interval']} chart", savefig=chart_path)
            st.image(chart_path); os.remove(chart_path)

    with trade_tab:
        st.subheader("📋 Suggested Stock Trade Plan (Bullish Swing)")
        entry_zone_start = last['EMA21'] * 0.99; entry_zone_end = last['EMA21'] * 1.01
        stop_loss = last['Low'] - last['ATR']; profit_target = last['Close'] + (2 * (last['Close'] - stop_loss))
        st.info(f"**Entry Zone:** Between **${entry_zone_start:.2f}** and **${entry_zone_end:.2f}**.\n"
                f"**Stop-Loss:** A close below **${stop_loss:.2f}**.\n"
                f"**Profit Target:** Around **${profit_target:.2f}** (2:1 Reward/Risk).")

    with backtest_tab:
        st.subheader(f"🧪 Historical Backtest for {ticker}"); st.info(f"Simulating trades based on your **currently selected indicators**. Entry is triggered if ALL selected signals are positive.")
        daily_hist_for_backtest, _ = get_data(ticker, "2y", "1d")
        if daily_hist_for_backtest is not None:
            daily_df = calculate_indicators(daily_hist_for_backtest.copy())
            trades, wins, losses = backtest_strategy(daily_df, selection)
            total_trades = wins + losses; win_rate = (wins / total_trades) * 100 if total_trades > 0 else 0
            col1, col2, col3 = st.columns(3)
            col1.metric("Trades Simulated", total_trades); col2.metric("Wins", wins); col3.metric("Win Rate", f"{win_rate:.1f}%")
            if trades: st.dataframe(pd.DataFrame(trades).tail(20))
        else: st.warning("Could not fetch daily data for backtesting.")

    with news_tab:
        st.subheader(f"📰 News & Information for {ticker}")
        col1, col2 = st.columns(2)
        with col1:
            st.markdown("#### ℹ️ Company Info"); st.write(f"**Name:** {info.get('longName', ticker)}"); st.write(f"**Sector:** {info.get('sector', 'N/A')}")
            st.markdown("#### 🔗 External Research Links")
            st.markdown(f"- [Yahoo Finance]({info.get('website', 'https://finance.yahoo.com')}) | [Finviz](https://finviz.com/quote.ashx?t={ticker})")
        with col2:
            st.markdown("#### 📅 Company Calendar")
            stock_obj_for_cal = yf.Ticker(ticker)
            # FIX: Corrected calendar check
            if stock_obj_for_cal and hasattr(stock_obj_for_cal, 'calendar') and isinstance(stock_obj_for_cal.calendar, pd.DataFrame) and not stock_obj_for_cal.calendar.empty:
                st.dataframe(stock_obj_for_cal.calendar.T)
            else: st.info("No upcoming calendar events found.")
            st.markdown("#### 🗞️ Latest Headlines")
            for h in finviz_data['headlines']: st.markdown(f"_{h}_")
            
    with log_tab:
        st.subheader("📝 Log Your Trade Analysis"); user_notes = st.text_area("Add your personal notes or trade thesis here:")
        if st.button("💾 Save Analysis to Log"):
            log_data = {"Timestamp": datetime.now().strftime("%Y-%m-%d %H:%M"), "Ticker": ticker, "Confidence": f"{overall_confidence:.0f}",
                        "Tech": f"{scores['technical']:.0f}", "Sent": f"{scores['sentiment']:.0f}", "Exp": f"{scores['expert']:.0f}",
                        "Price": f"{last['Close']:.2f}", "Notes": user_notes.replace("\n", " ")}
            log_analysis(log_data)
        st.markdown("---"); st.subheader("📋 View Saved Log")
        if os.path.exists(LOG_FILE): st.dataframe(pd.read_csv(LOG_FILE).sort_values(by="Timestamp", ascending=False))
        else: st.warning("No log file found.")

# === Main Script Execution ===
TIMEFRAME_MAP = {
    "Scalp Trading": {"period": "5d", "interval": "5m", "weights": {"technical": 0.9, "sentiment": 0.1, "expert": 0.0}},
    "Day Trading": {"period": "60d", "interval": "60m", "weights": {"technical": 0.7, "sentiment": 0.2, "expert": 0.1}},
    "Swing Trading": {"period": "1y", "interval": "1d", "weights": {"technical": 0.6, "sentiment": 0.2, "expert": 0.2}},
    "Position Trading": {"period": "5y", "interval": "1wk", "weights": {"technical": 0.4, "sentiment": 0.2, "expert": 0.4}}
}
selected_params_main = TIMEFRAME_MAP[timeframe]

if ticker:
    try:
        hist_data, info_data = get_data(ticker, selected_params_main['period'], selected_params_main['interval'])
        if hist_data is None: st.error(f"Could not fetch data for {ticker} on a {selected_params_main['interval']} interval.")
        else: display_dashboard(ticker, hist_data, info_data, selected_params_main, indicator_selection)
    except Exception as e:
        st.error(f"An error occurred: {e}")
else:
    st.info("Please enter a stock ticker in the sidebar to begin analysis.")