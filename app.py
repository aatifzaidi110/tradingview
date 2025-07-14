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
st.set_page_config(page_title="Aatif's AI Trading Hub", layout="wide")
st.title("🚀 Aatif's AI-Powered Trading Hub")

# === SIDEBAR: Controls & Selections ===
st.sidebar.header("⚙️ Controls")
ticker = st.sidebar.text_input("Enter a Ticker Symbol", value="NVDA").upper()
# FIX: Re-added all trading styles
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

finviz_data = get_finviz_data(ticker)
auto_sentiment_score = convert_compound_to_100_scale(finviz_data['sentiment_compound'])
auto_expert_score = EXPERT_RATING_MAP.get(finviz_data['recom'], 50)
auto_sentiment_score_placeholder.markdown(f"**Automated Sentiment:** `{auto_sentiment_score}` (from headlines)")
auto_expert_score_placeholder.markdown(f"**Automated Expert Rating:** `{auto_expert_score}` ({finviz_data['recom']})")

@st.cache_data(ttl=60)
def get_data(symbol, period, interval):
    stock = yf.Ticker(symbol); hist = stock.history(period=period, interval=interval, auto_adjust=True)
    return (hist, stock.info) if not hist.empty else (None, None)

@st.cache_data(ttl=300)
def get_options_chain(ticker, expiry_date):
    stock_obj = yf.Ticker(ticker)
    options = stock_obj.option_chain(expiry_date)
    return options.calls, options.puts

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

def get_options_suggestion(confidence, current_price, calls):
    if confidence >= 75:
        suggestion = "💡 **Strategy Suggestion: Buy an ITM Call.**"; reason = "High confidence suggests a strong directional move. An ITM call (Delta > 0.60) has a higher probability of success."
        target_call = calls[calls['inTheMoney']].iloc[0] if not calls[calls['inTheMoney']].empty else None
        return "success", suggestion, reason, target_call
    elif 60 <= confidence < 75:
        suggestion = "💡 **Strategy Suggestion: Consider a Bull Call (Debit) Spread.**"; reason = "Moderate confidence favors a defined-risk strategy, which lowers cost and caps risk."
        target_call = calls.iloc[(calls['strike'] - current_price).abs().argsort()[:1]].iloc[0]
        return "info", suggestion, reason, target_call
    else:
        suggestion = "💡 **Strategy Suggestion: Caution Advised.**"; reason = "Low confidence suggests avoiding leveraged directional bets."
        return "warning", suggestion, reason, None

def display_dashboard(ticker, hist, info, params, selection):
    df = calculate_indicators(hist.copy()); last = df.iloc[-1]
    signals = generate_signals(last, selection)
    
    technical_score = (sum(1 for f in signals.values() if f) / len(signals)) * 100 if signals else 0
    scores = {"technical": technical_score, "sentiment": sentiment_score, "expert": expert_score}
    weights = params['weights']
    overall_confidence = min(round((weights["technical"] * scores["technical"] + weights["sentiment"] * scores["sentiment"] + weights["expert"] * scores["expert"]), 2), 100)
    
    st.header(f"Analysis for {ticker} ({params['interval']} Interval)")
    
    tab_list = ["📊 Main Analysis", "📈 Trade Plan & Options", "📘 Indicator Guide", "📰 Headlines", "ℹ️ Ticker Info"]
    main_tab, trade_tab, guide_tab, news_tab, info_tab = st.tabs(tab_list)

    with main_tab:
        col1, col2 = st.columns([1, 2])
        with col1:
            st.subheader("💡 Confidence Score"); st.metric("Overall Confidence", f"{overall_confidence:.0f}/100"); st.progress(overall_confidence / 100)
            st.markdown(f"- **Technical:** `{scores['technical']:.0f}` (W: `{weights['technical']*100:.0f}%`)\n- **Sentiment:** `{scores['sentiment']:.0f}` (W: `{weights['sentiment']*100:.0f}%`)\n- **Expert:** `{scores['expert']:.0f}` (W: `{weights['expert']*100:.0f}%`)")
            st.subheader("🎯 Key Price Levels")
            current_price = last['Close']; prev_close = df['Close'].iloc[-2]; price_delta = current_price - prev_close
            st.metric(label="Current Price", value=f"${current_price:.2f}", delta=f"${price_delta:.2f}")
            wk52_high = info.get('fiftyTwoWeekHigh', 'N/A'); wk52_low = info.get('fiftyTwoWeekLow', 'N/A')
            st.write(f"**52W High:** ${wk52_high:.2f}" if isinstance(wk52_high, float) else f"**52W High:** {wk52_high}")
            st.write(f"**52W Low:** ${wk52_low:.2f}" if isinstance(wk52_low, float) else f"**52W Low:** {wk52_low}")
            st.write(f"**ATR (Volatility):** {last['ATR']:.3f}")
        with col2:
            st.subheader("📈 Price Chart"); chart_path = f"chart_{ticker}.png"
            mav_tuple = (21, 50, 200) if selection.get("EMA Trend") else None
            ap = [mpf.make_addplot(df.tail(120)[['BB_high', 'BB_low']])] if selection.get("Bollinger Bands") else None
            mpf.plot(df.tail(120), type='candle', style='yahoo', mav=mav_tuple, volume=True, addplot=ap, title=f"{ticker} - {params['interval']} chart", savefig=chart_path)
            st.image(chart_path); os.remove(chart_path)

    with trade_tab:
        st.subheader("📋 Suggested Trade Plan (Bullish Swing)")
        entry_zone_start = last['EMA21'] * 0.99; entry_zone_end = last['EMA21'] * 1.01
        stop_loss = last['Low'] - last['ATR']
        profit_target = last['Close'] + (2 * (last['Close'] - stop_loss))
        st.info(f"**Entry Zone:** Between **${entry_zone_start:.2f}** and **${entry_zone_end:.2f}**.\n"
                f"**Stop-Loss:** A close below **${stop_loss:.2f}**.\n"
                f"**Profit Target:** Around **${profit_target:.2f}** (2:1 Reward/Risk).")
        st.markdown("---")

        st.subheader("🎭 Options Analysis")
        stock_obj = yf.Ticker(ticker)
        expirations = stock_obj.options
        if not expirations:
            st.warning("No options data available for this ticker.")
        else:
            exp_date_str = st.selectbox("Select Option Expiration Date:", expirations)
            if exp_date_str:
                calls, puts = get_options_chain(ticker, exp_date_str)
                
                rec_type, suggestion, reason, target_call = get_options_suggestion(overall_confidence, last['Close'], calls)
                if rec_type == "success": st.success(suggestion)
                elif rec_type == "info": st.info(suggestion)
                else: st.warning(suggestion)
                st.write(reason)
                if target_call is not None:
                    st.write("**Example Target Option:**"); st.json(target_call.to_dict())

                st.markdown(f"[**🔗 Analyze this chain on OptionCharts.io**](https://optioncharts.io/options/{ticker}/chain/{exp_date_str})")
                
                # --- FIX: Robust column selection ---
                st.write("#### Call Options")
                desired_cols = ['strike', 'lastPrice', 'volume', 'openInterest', 'impliedVolatility', 'delta', 'theta']
                available_cols = [col for col in desired_cols if col in calls.columns]
                if available_cols:
                    st.dataframe(calls[available_cols].set_index('strike'))
                else:
                    st.warning("Basic options data is available, but Greeks (Delta, Theta) are missing for this ticker.")
                
                with st.expander("📘 Understanding Option Greeks & IV"):
                    st.markdown("- **Implied Volatility (IV):** Market's expectation of future swings.\n- **Delta:** Option's price change per $1 stock price change.\n- **Theta:** Time decay; value lost per day.")

    with guide_tab:
        st.subheader("📘 Dynamic Indicator Guide"); st.info("This guide explains the indicators you have selected.")
        guide_data = []
        if selection.get("EMA Trend"): guide_data.append({"Indicator": "EMA Trend", "Description": "Shows trend alignment. Stacked EMAs (21>50>200) confirm a strong uptrend.", "Current": f"21: {last['EMA21']:.2f}", "Ideal": "21 > 50 > 200", "Status": '🟢' if signals.get("Uptrend (21>50>200 EMA)") else '🔴'})
        if selection.get("RSI Momentum"): guide_data.append({"Indicator": "RSI (14)", "Description": "Measures momentum. We want RSI > 50 to confirm buyer strength.", "Current": f"{last['RSI']:.2f}", "Ideal": "> 50", "Status": '🟢' if signals.get("Bullish Momentum (RSI > 50)") else '🔴'})
        if selection.get("MACD Crossover"): guide_data.append({"Indicator": "MACD Diff", "Description": "Highlights momentum direction. Positive value = bullish.", "Current": f"{last['MACD_diff']:.2f}", "Ideal": "> 0", "Status": '🟢' if signals.get("MACD Bullish (Diff > 0)") else '🔴'})
        if selection.get("Volume Spike"): guide_data.append({"Indicator": "Volume", "Description": "Confirms conviction via institutional interest.", "Current": f"{last['Volume']:,.0f}", "Ideal": f"> {last['Vol_Avg_50']:,.0f}", "Status": '🟢' if signals.get("Volume Spike (>1.5x Avg)") else '🔴'})
        if guide_data: st.table(pd.DataFrame(guide_data).set_index("Indicator"))
        else: st.warning("No indicators selected.")
    with news_tab:
        st.subheader(f"📰 Latest News Headlines for {ticker}")
        st.info("The AI analyzed these headlines to generate the automated sentiment score.")
        for i, h in enumerate(finviz_data['headlines']): st.markdown(f"{i+1}. {h}")
    with info_tab:
        st.subheader(f"ℹ️ About {info.get('longName', ticker)}"); st.write(f"**Sector:** {info.get('sector', 'N/A')} | **Industry:** {info.get('industry', 'N/A')}")
        st.markdown(f"**Business Summary:**"); st.info(f"{info.get('longBusinessSummary', 'No summary available.')}")
    st.sidebar.markdown("---"); st.sidebar.warning("Disclaimer: This tool is for educational purposes only. Not investment advice.")

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
        if hist_data is None or hist_data.empty:
            st.error(f"Could not fetch data for {ticker} on a {selected_params_main['interval']} interval.")
        else:
            display_dashboard(ticker, hist_data, info_data, selected_params_main, indicator_selection)
    except Exception as e:
        st.error(f"An error occurred: {e}")
else:
    st.info("Please enter a stock ticker in the sidebar to begin analysis.")