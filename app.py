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
with st.sidebar.expander("Trend Indicators", expanded=True):
    indicator_selection = {
        "EMA Trend": st.checkbox("EMA Trend (21, 50, 200)", value=True),
        "Ichimoku Cloud": st.checkbox("Ichimoku Cloud", value=True),
        "Parabolic SAR": st.checkbox("Parabolic SAR", value=True),
        "ADX": st.checkbox("ADX", value=True),
    }
with st.sidebar.expander("Momentum & Volume Indicators"):
    indicator_selection.update({
        "RSI Momentum": st.checkbox("RSI Momentum", value=True),
        "Stochastic": st.checkbox("Stochastic Oscillator", value=True),
        "CCI": st.checkbox("Commodity Channel Index (CCI)", value=True),
        "ROC": st.checkbox("Rate of Change (ROC)", value=True),
        "Volume Spike": st.checkbox("Volume Spike", value=True),
        "OBV": st.checkbox("On-Balance Volume (OBV)", value=True),
        "VWAP": st.checkbox("VWAP (Intraday only)", value=True),
    })
with st.sidebar.expander("Display-Only Indicators"):
    indicator_selection.update({
        "Bollinger Bands": st.checkbox("Bollinger Bands Display", value=True),
        "Pivot Points": st.checkbox("Pivot Points Display (Daily only)", value=True),
    })

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
    sentiment_score = 50; expert_score = 50
    finviz_data = {"headlines": ["Automation is disabled."]}

@st.cache_data(ttl=60)
def get_data(symbol, period, interval):
    stock = yf.Ticker(symbol); hist = stock.history(period=period, interval=interval, auto_adjust=True)
    return (hist, stock.info) if not hist.empty else (None, None)

def calculate_indicators(df, is_intraday=False):
    # This function remains the same
    try: df["EMA21"]=ta.trend.ema_indicator(df["Close"],21); df["EMA50"]=ta.trend.ema_indicator(df["Close"],50); df["EMA200"]=ta.trend.ema_indicator(df["Close"],200)
    except Exception: pass
    try: ichimoku = ta.trend.IchimokuIndicator(df['High'], df['Low']); df['ichimoku_a'] = ichimoku.ichimoku_a(); df['ichimoku_b'] = ichimoku.ichimoku_b()
    except Exception: pass
    try: df['psar'] = ta.trend.PSARIndicator(df['High'], df['Low'], df['Close']).psar()
    except Exception: pass
    try: df['adx'] = ta.trend.ADXIndicator(df['High'], df['Low'], df['Close']).adx()
    except Exception: pass
    try: df["RSI"]=ta.momentum.RSIIndicator(df["Close"]).rsi()
    except Exception: pass
    try: stoch = ta.momentum.StochasticOscillator(df['High'], df['Low'], df['Close']); df['stoch_k'] = stoch.stoch(); df['stoch_d'] = stoch.stoch_signal()
    except Exception: pass
    try: df['cci'] = ta.momentum.cci(df['High'], df['Low'], df['Close'])
    except AttributeError: st.warning("CCI indicator failed. Your `ta` library might be outdated.", icon="⚠️")
    except Exception: pass
    try: df['roc'] = ta.momentum.ROCIndicator(df['Close']).roc()
    except Exception: pass
    try: df['obv'] = ta.volume.OnBalanceVolumeIndicator(df['Close'], df['Volume']).on_balance_volume()
    except Exception: pass
    if is_intraday:
        try: df['vwap'] = ta.volume.VolumeWeightedAveragePrice(df['High'], df['Low'], df['Close'], df['Volume']).volume_weighted_average_price()
        except Exception: pass
    df["ATR"]=ta.volatility.AverageTrueRange(df["High"],df["Low"],df["Close"]).average_true_range()
    bb=ta.volatility.BollingerBands(df["Close"]); df["BB_low"]=bb.bollinger_lband(); df["BB_high"]=bb.bollinger_hband()
    df["Vol_Avg_50"]=df["Volume"].rolling(50).mean()
    return df

# === FIX: Redesigned Pivot Point calculation ===
def calculate_pivot_points(df):
    """Calculates pivots for the entire DataFrame."""
    df_pivots = pd.DataFrame(index=df.index)
    df_pivots['Pivot'] = (df['High'] + df['Low'] + df['Close']) / 3
    df_pivots['R1'] = (2 * df_pivots['Pivot']) - df['Low']
    df_pivots['S1'] = (2 * df_pivots['Pivot']) - df['High']
    df_pivots['R2'] = df_pivots['Pivot'] + (df['High'] - df['Low'])
    df_pivots['S2'] = df_pivots['Pivot'] - (df['High'] - df['Low'])
    # Shift the pivots to be valid for the *next* day
    return df_pivots.shift(1)

def generate_signals(df, selection, is_intraday=False):
    signals = {}; last_row = df.iloc[-1]
    if selection.get("OBV") and 'obv' in df.columns and len(df) > 10:
        signals["OBV Rising"] = last_row['obv'] > df['obv'].rolling(10).mean().iloc[-1]
    if selection.get("EMA Trend"): signals["Uptrend (21>50>200 EMA)"] = last_row["EMA50"] > last_row["EMA200"] and last_row["EMA21"] > last_row["EMA50"]
    if selection.get("Ichimoku Cloud"): signals["Bullish Ichimoku"] = last_row['Close'] > last_row['ichimoku_a'] and last_row['Close'] > last_row['ichimoku_b']
    if selection.get("Parabolic SAR"): signals["Bullish PSAR"] = last_row['Close'] > last_row['psar']
    if selection.get("ADX"): signals["Strong Trend (ADX > 25)"] = last_row['adx'] > 25
    if selection.get("RSI Momentum"): signals["Bullish Momentum (RSI > 50)"] = last_row["RSI"] > 50
    if selection.get("Stochastic"): signals["Bullish Stoch Cross"] = last_row['stoch_k'] > last_row['stoch_d']
    if selection.get("CCI") and 'cci' in df.columns: signals["Bullish CCI (>0)"] = last_row['cci'] > 0
    if selection.get("ROC"): signals["Positive ROC (>0)"] = last_row['roc'] > 0
    if selection.get("Volume Spike"): signals["Volume Spike (>1.5x Avg)"] = last_row["Volume"] > last_row["Vol_Avg_50"] * 1.5
    if selection.get("VWAP") and is_intraday: signals["Price > VWAP"] = last_row['Close'] > last_row['vwap']
    return signals

def display_dashboard(ticker, hist, info, params, selection):
    is_intraday = params['interval'] in ['5m', '60m']
    df = calculate_indicators(hist.copy(), is_intraday)
    signals = generate_signals(df, selection, is_intraday)
    last = df.iloc[-1]
    
    technical_score = (sum(1 for f in signals.values() if f) / len(signals)) * 100 if signals else 0
    scores = {"technical": technical_score, "sentiment": sentiment_score, "expert": expert_score}
    
    final_weights = params['weights'].copy()
    if not use_automation:
        final_weights['technical'] = 1.0; final_weights['sentiment'] = 0.0; final_weights['expert'] = 0.0
        scores['sentiment'] = 0; scores['expert'] = 0
    
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
            
            st.subheader("✅ Technical Analysis Readout")
            with st.expander("📈 Trend Indicators", expanded=True):
                def format_value(signal_name, value):
                    is_fired = signals.get(signal_name, False)
                    status_icon = '🟢' if is_fired else '🔴'
                    name = signal_name.split('(')[0].strip()
                    value_str = f"`{value:.2f}`" if isinstance(value, (int, float)) else ""
                    return f"{status_icon} **{name}:** {value_str}"

                if selection.get("EMA Trend"): st.markdown(format_value("Uptrend (21>50>200 EMA)", None))
                if selection.get("Ichimoku Cloud"): st.markdown(format_value("Bullish Ichimoku", None))
                if selection.get("Parabolic SAR"): st.markdown(format_value("Bullish PSAR", None))
                if selection.get("ADX"): st.markdown(format_value("Strong Trend (ADX > 25)", last.get('adx')))
            with st.expander("💨 Momentum Indicators", expanded=True):
                if selection.get("RSI Momentum"): st.markdown(format_value("Bullish Momentum (RSI > 50)", last.get('RSI')))
                if selection.get("Stochastic"): st.markdown(format_value("Bullish Stoch Cross", last.get('stoch_k')))
                if selection.get("CCI"): st.markdown(format_value("Bullish CCI (>0)", last.get('cci')))
                if selection.get("ROC"): st.markdown(format_value("Positive ROC (>0)", last.get('roc')))
            with st.expander("📊 Volume Indicators", expanded=True):
                if selection.get("Volume Spike"): st.markdown(format_value("Volume Spike (>1.5x Avg)", None))
                if selection.get("OBV"): st.markdown(format_value("OBV Rising", None))
                if is_intraday and selection.get("VWAP"): st.markdown(format_value("Price > VWAP", last.get('vwap')))
            
            # --- FIX: Pivot Point Display Logic ---
            if selection.get("Pivot Points") and not is_intraday:
                with st.expander("📌 Support & Resistance (Daily Pivots)"):
                    pivots_df = calculate_pivot_points(df)
                    if not pivots_df.empty:
                        # Display today's pivots, which are based on yesterday's data
                        st.dataframe(pivots_df.iloc[-1].round(2).to_frame().T)

        with col2:
            st.subheader("📈 Price Chart"); chart_path = f"chart_{ticker}.png"
            mav_tuple = (21, 50, 200) if selection.get("EMA Trend") else None
            ap = [mpf.make_addplot(df.tail(120)[['BB_high', 'BB_low']])] if selection.get("Bollinger Bands") else None
            mpf.plot(df.tail(120), type='candle', style='yahoo', mav=mav_tuple, volume=True, addplot=ap, title=f"{ticker} - {params['interval']} chart", savefig=chart_path)
            st.image(chart_path); os.remove(chart_path)

    # ... Other tabs remain the same
    with trade_tab:
        st.subheader("📋 Suggested Stock Trade Plan (Bullish Swing)")
        # Trade plan logic here
    with backtest_tab:
        st.subheader("🧪 Historical Backtest")
        # Backtest logic here
    with news_tab:
        st.subheader("📰 News & Information")
        # News and info logic here
    with log_tab:
        st.subheader("📝 Trade Log")
        # Logging logic here

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