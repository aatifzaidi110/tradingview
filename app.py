#==Gemini test==
# -*- coding: utf-8 -*-
# vim: set ts=4 sw=4 et:

# === Streamlit Imports ===
import streamlit as st
import yfinance as yf
import pandas as pd
import ta
import mplfinance as mpf
import matplotlib.pyplot as plt
import os # For cleaning up chart file if needed, though direct plot is preferred now

# === Page Setup ===
st.set_page_config(page_title="Aatif's Swing Dashboard", layout="centered")
st.title("📊 Aatif's Swing Trade Analyzer")

# === Ticker Input ===
# Added input validation and standardization
ticker = st.text_input("Enter a Ticker Symbol", value="NVDA").strip().upper()

# If no ticker is entered, stop the app
if not ticker:
    st.info("Please enter a ticker symbol to analyze.")
    st.stop()

# === Utility Functions ===
def status(flag): return "✅" if flag else "❌"
def color_status(flag): return "🟢 Green" if flag else "🔴 Red"

def calculate_confidence(technical, sentiment, expert, weights):
    """Calculates the overall confidence score based on weighted components."""
    score = round(
        weights["technical"] * technical +
        weights["sentiment"] * sentiment +
        weights["expert"] * expert,
        2
    )
    return score

# === Caching ===
@st.cache_data(ttl=600)
def get_data(symbol):
    """Fetches historical stock data and basic info from Yahoo Finance, with caching."""
    stock = yf.Ticker(symbol)
    # Fetch 1 year of data to ensure enough history for 200 EMA and longer backtesting
    hist = stock.history(period="1y") # Changed to 1 year for more robust EMA200 and backtesting
    info = stock.info
    price = info.get("currentPrice", hist["Close"].iloc[-1] if not hist.empty else None)
    previous_close = info.get("previousClose", hist["Close"].iloc[-2] if len(hist) > 1 else None)
    earnings = info.get("earningsDate", "N/A")
    dividend = info.get("dividendDate", "N/A")

    if hist.empty or price is None or previous_close is None:
        raise ValueError("Could not fetch sufficient historical data or current price.")

    return hist, info, price, previous_close, earnings, dividend

# === Dashboard ===
try:
    hist, info, price, previous_close, earnings, dividend = get_data(ticker)
except Exception as e:
    st.error(f"⚠️ Data fetch failed for '{ticker}'. This might be due to an invalid ticker, rate limits, or no data available. Error: {e}")
    if st.button("🔄 Retry", key="retry_button"): # Added key for uniqueness
        st.rerun() # Replaced st.experimental_rerun()
    st.stop()

df = hist.copy()

# Ensure we have enough data for all indicators, especially 200 EMA
if len(df) < 200:
    st.warning(f"Not enough historical data for {ticker} to calculate all indicators (requires at least 200 data points). Displaying available data.")
    # Stop execution if critical data is missing for calculations
    st.stop()

# === Technical Indicator Calculations ===
df["EMA21"] = ta.trend.ema_indicator(df["Close"], 21)
df["EMA50"] = ta.trend.ema_indicator(df["Close"], 50)
df["EMA200"] = ta.trend.ema_indicator(df["Close"], 200)
df["RSI"] = ta.momentum.RSIIndicator(df["Close"]).rsi()
bb = ta.volatility.BollingerBands(df["Close"])
df["BB_low"] = bb.bollinger_lband()
df["BB_high"] = bb.bollinger_hband() # Added for more complete BB info
df["MACD_diff"] = ta.trend.macd_diff(df["Close"])
df["ATR"] = ta.volatility.AverageTrueRange(df["High"], df["Low"], df["Close"]).average_true_range()
df["Vol_Avg"] = df["Volume"].rolling(50).mean()
last = df.iloc[-1]

# Ensure 'last' row has all calculated values (important for indicators with NaN at start)
if last.isnull().any():
    st.warning("Some technical indicators could not be calculated for the most recent period due to insufficient data. Displaying results for the last complete period.")
    # Fallback to the last row where all indicators are valid
    last = df.dropna().iloc[-1]
    if last.empty:
        st.error("Could not calculate any valid technical indicators.")
        st.stop()

# === Timeframe Selector (Moved Up to influence weights) ===
st.subheader("🎯 Select Your Trading Style")
timeframe = st.radio("Choose your trading style:", [
    "Swing Trading (1D)",
    "Day Trading (1H)", # Note: Current data is 1D, so these are conceptual for now
    "Scalp Trading (5Min)", # Note: Current data is 1D, so these are conceptual for now
    "Position Trading (1W)"
], index=0) # Set default to Swing Trading

# === Strategy Tagging ===
strategy_map = {
    "Swing Trading (1D)": "Swing Trade",
    "Day Trading (1H)": "Day Trade",
    "Scalp Trading (5Min)": "Scalp Trade",
    "Position Trading (1W)": "Position Trade"
}
selected_strategy = strategy_map.get(timeframe, "Unknown")
st.write(f"📌 **Strategy Type Selected:** {selected_strategy}")

# === Dynamic Confidence Weights (Applied before confidence calculation) ===
if selected_strategy == "Scalp Trade":
    weights = {"technical": 0.5, "sentiment": 0.3, "expert": 0.2}
elif selected_strategy == "Day Trade":
    weights = {"technical": 0.6, "sentiment": 0.25, "expert": 0.15}
elif selected_strategy == "Swing Trade":
    weights = {"technical": 0.6, "sentiment": 0.2, "expert": 0.2}
elif selected_strategy == "Position Trade":
    weights = {"technical": 0.4, "sentiment": 0.2, "expert": 0.4}
else:
    weights = {"technical": 0.6, "sentiment": 0.2, "expert": 0.2} # Default weights

# === Signals & Scores ===
signals = {
    "RSI": (last["RSI"] < 30) or (last["RSI"] > 70), # More typical RSI extreme ranges
    "MACD": last["MACD_diff"] > 0,
    "EMA": last["EMA21"] > last["EMA50"] > last["EMA200"],
    "ATR": price > previous_close + last["ATR"] * 0.5, # Adjusted ATR breakout logic, can be fine-tuned
    "Volume": last["Volume"] > last["Vol_Avg"] * 1.5,
    "BB": (price < last["BB_low"] and last["RSI"] < 35) # Combined with RSI for stronger signal
}
signal_weights = {"RSI": 20, "MACD": 20, "EMA": 15, "ATR": 15, "Volume": 15, "BB": 15}

# === ML Boost Logic (before scoring)
# ML Boost calculation
ml_boost = 10 if last["RSI"] > 50 and price > last["EMA200"] else 0
technical_score = sum([signal_weights[k] for k in signals if signals[k]]) + ml_boost

# Placeholder for Sentiment and Expert Scores - Suggestion: implement actual fetching/user input
st.subheader("⚙️ Manual Score Adjustments (for testing)")
sentiment_score = st.slider("Sentiment Score (0-100)", 0, 100, 10, help="Manually adjust the sentiment score (e.g., from news analysis).")
expert_score = st.slider("Expert Score (0-100)", 0, 100, 10, help="Manually adjust the expert score (e.g., from analyst ratings).")

# === Calculate final confidence score using dynamic weights ===
overall_confidence = calculate_confidence(technical_score, sentiment_score, expert_score, weights)

stop_loss = round(price - last["ATR"], 2)
# Calculate support and resistance based on recent highs/lows
support = df["Low"].rolling(window=20, min_periods=1).min().iloc[-1]
resistance = df["High"].rolling(window=20, min_periods=1).max().iloc[-1]

# === Overview Panel ===
st.subheader(f"📌 {ticker.upper()} Overview")
col1, col2 = st.columns(2)
with col1:
    description = info.get('longBusinessSummary', 'N/A')
    if len(description) > 300:
        truncated_desc = description[:300]
        last_space = truncated_desc.rfind(' ')
        if last_space != -1:
            truncated_desc = truncated_desc[:last_space] + "..."
        else:
            truncated_desc += "..."
        st.write(f"**Description:** {truncated_desc}")
    else:
        st.write(f"**Description:** {description}")

    st.write(f"**Current Price:** ${price:.2f}")
    st.write(f"**Previous Close:** ${previous_close:.2f}")
    st.write(f"**Earnings Date:** {earnings}")
    st.write(f"**Dividend Date:** {dividend}")
with col2:
    st.write(f"**Support Level (20D):** ${support:.2f}")
    st.write(f"**Resistance Level (20D):** ${resistance:.2f}")
    st.markdown(f"- [📰 Google News](https://news.google.com/search?q={ticker}+stock)")
    st.markdown(f"- [📊 Finviz](https://finviz.com/quote.ashx?t={ticker})")
    st.markdown(f"- [📈 Barchart](https://www.barchart.com/stocks/quotes/{ticker}/overview)")
    st.markdown(f"- [🎯 TipRanks](https://www.tipranks.com/stocks/{ticker}/forecast)")

# === Chart Snapshot ===
st.subheader("🖼️ Chart Snapshot")
fig, axlist = mpf.plot(df[-60:], type='candle', mav=(21, 50, 200), volume=True, style='yahoo', returnfig=True)
st.pyplot(fig, clear_figure=True) # Pass figure directly and clear it
plt.close(fig) # Explicitly close the figure to free memory

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

# === Technical Indicator Table with Full Descriptions ===
st.subheader("📊 Technical Indicator Breakdown")
st.markdown(f"""
| **Indicator** | **Current Value ({ticker.upper()})** | **Meaning & Ideal Range** | **Status** |
|-------------------|--------------------------|------------------------------------------------------------------------------------------------------------|------------|
| **RSI** | {last['RSI']:.2f}         | Measures overbought/oversold momentum. Ideal: <30 (oversold, potential bounce), >70 (overbought, potential reversal).                          | {color_status(signals["RSI"])} |
| **MACD Diff** | {last['MACD_diff']:.2f}   | Indicates trend direction & strength. Ideal: >0 for bullish momentum, rising for strong up-trend.                                     | {color_status(signals["MACD"])} |
| **EMA Stack** | 21:{last['EMA21']:.2f} > 50:{last['EMA50']:.2f} > 200:{last['EMA200']:.2f} | Alignment of short- to long-term trends. Ideal: 21 EMA > 50 EMA > 200 EMA (bullish alignment).         | {color_status(signals["EMA"])} |
| **ATR Breakout** | {last['ATR']:.2f}         | Gauges volatility. Ideal: Price showing significant movement (e.g., price > prev_close + 0.5*ATR) for a breakout.                                    | {color_status(signals["ATR"])} |
| **Volume Spike** | {last['Volume']:.0f} vs Avg(50): {last['Vol_Avg']:.0f} | Highlights interest. Ideal: current volume > 1.5× average for strong moves.      | {color_status(signals["Volume"])} |
| **Bollinger Band**| Price < ${last['BB_low']:.2f} | Shows price extremes. Ideal: Price near lower band, combined with oversold RSI, may suggest bounce.            | {color_status(signals["BB"])} |
| **ML Boost (Bonus)**| —              | A bonus applied when RSI > 50 and Price > EMA200, indicating strong bullish momentum.                 | {ml_boost} points |
""")

# === Confidence Scoring Table (Only one instance) ===
st.markdown("### 🧮 Confidence Scoring Table")
st.markdown(f"""
| **Component** | **Weight (%)** | **Raw Score** | **Contribution** |
|---------------------|----------------|---------------|------------------|
| **Technical Score** | {weights['technical']*100:.0f}% | {technical_score}/100 | {weights['technical']*technical_score:.1f} |
| **Sentiment Score** | {weights['sentiment']*100:.0f}% | {sentiment_score}/100 | {weights['sentiment']*sentiment_score:.1f} |
| **Expert Score** | {weights['expert']*100:.0f}%    | {expert_score}/100 | {weights['expert']*expert_score:.1f} |
|                     |                |               |                  |
| **➡️ Overall Confidence** |       —        |       —       | **{overall_confidence}/100** |
""")

# === Confidence Pie Chart ===
st.subheader("📊 Confidence Weight Distribution")

labels = ["Technical", "Sentiment", "Expert"]
raw_scores = [technical_score, sentiment_score, expert_score]
contributions = [
    round(raw_scores[0] * weights["technical"], 1),
    round(raw_scores[1] * weights["sentiment"], 1),
    round(raw_scores[2] * weights["expert"], 1)
]

fig_pie, ax_pie = plt.subplots() # Use a different variable name for the pie chart figure
colors = ["#4CAF50", "#2196F3", "#FFC107"]
ax_pie.pie(
    contributions,
    labels=[f"{labels[i]} ({contributions[i]})" for i in range(3)],
    autopct="%1.1f%%",
    startangle=90,
    colors=colors
)
ax_pie.axis("equal")
st.pyplot(fig_pie, clear_figure=True)
plt.close(fig_pie) # Explicitly close the figure to free memory

# === Backtest Function ===
def backtest_signals(df_historical, atr_multiplier=1.0, reward_multiplier=2.0, signal_weights=None):
    """
    Simulates trades based on defined signals and calculates win rate.
    Uses a more comprehensive set of signals for entry.
    """
    if signal_weights is None:
        signal_weights = {"RSI": 20, "MACD": 20, "EMA": 15, "ATR": 15, "Volume": 15, "BB": 15}

    trades = []
    # Start backtesting from where all indicators are valid (e.g., after 200 periods)
    start_idx = df_historical.dropna().index.min()
    if pd.isna(start_idx): # Check if dropna resulted in an empty df
        return []
    start_pos = df_historical.index.get_loc(start_idx)

    # Ensure we have enough future data for potential exit (e.g., 5 days)
    for i in range(max(start_pos, 200), len(df_historical) - 5): # Ensure min 200 for EMA
        row = df_historical.iloc[i]
        next_rows = df_historical.iloc[i+1:i+6] # Look 5 days ahead for exit

        # Re-evaluate signals for each backtest point
        current_signals = {
            "RSI": (row["RSI"] < 30) or (row["RSI"] > 70),
            "MACD": row["MACD_diff"] > 0,
            "EMA": row["EMA21"] > row["EMA50"] > row["EMA200"],
            # FIX APPLIED HERE: Access previous close from df_historical
            "ATR": row["Close"] > df_historical.iloc[i-1]["Close"] + row["ATR"] * 0.5 if i > 0 else False,
            "Volume": row["Volume"] > row["Vol_Avg"] * 1.5,
            "BB": (row["Close"] < row["BB_low"] and row["RSI"] < 35)
        }

        # Calculate technical score for this backtest point
        boost = 10 if row["RSI"] > 50 and row["Close"] > row["EMA200"] else 0
        current_technical_score = sum([signal_weights[k] for k in current_signals if current_signals[k]]) + boost

        # Entry condition (example: requires a certain technical score)
        # You can make this more complex based on your specific strategy
        if current_technical_score >= 70: # Example threshold for entry
            entry_price = row["Close"]
            stop_loss_price = entry_price - row["ATR"] * atr_multiplier
            take_profit_price = entry_price + row["ATR"] * reward_multiplier
            result = "Neutral"
            exit_price = entry_price # Default exit if no condition met

            for k, future_row in enumerate(next_rows.itertuples()):
                # Check for stop loss
                if future_row.Low < stop_loss_price:
                    exit_price = stop_loss_price
                    result = "Loss"
                    break
                # Check for take profit
                elif future_row.High > take_profit_price:
                    exit_price = take_profit_price
                    result = "Win"
                    break
                # Exit after 5 days if no stop/target hit (time-based exit)
                if k == len(next_rows) - 1:
                    exit_price = future_row.Close # Exit at close of last day considered
                    result = "Time Expiry"


            trades.append({
                "Date": row.name.strftime('%Y-%m-%d'), # Add date for context
                "Entry": round(entry_price, 2),
                "Stop Loss": round(stop_loss_price, 2),
                "Target": round(take_profit_price, 2),
                "Exit": round(exit_price, 2),
                "Result": result,
                "ML Boost Used": boost,
                "Technical Score": current_technical_score
            })
    return trades

# === Backtest Summary ===
st.subheader("🧪 Historical Signal Backtest")
st.info("Note: This backtest is a simplified simulation based on daily data and specific entry/exit conditions. It does not account for slippage, commissions, or real-time intraday price action.")

trades = backtest_signals(df, signal_weights=signal_weights) # Pass signal weights
wins = sum(1 for t in trades if t["Result"] == "Win")
losses = sum(1 for t in trades if t["Result"] == "Loss")
neutral_or_time = sum(1 for t in trades if t["Result"] in ["Neutral", "Time Expiry"])
total = len(trades)
win_rate = round((wins / total) * 100, 2) if total else 0

st.write(f"- 📈 Total Trades Simulated: **{total}** (over last ~{len(df)} days)")
st.write(f"- ✅ Wins: {wins}")
st.write(f"- ❌ Losses: {losses}")
st.write(f"- ⏳ Neutral/Time Expiry: {neutral_or_time}")
st.write(f"- 🏆 **Win Rate: {win_rate}%**")

if trades:
    trades_df = pd.DataFrame(trades)
    st.subheader("Detailed Backtest Trades")
    st.dataframe(trades_df.tail(10)) # Show last 10 trades

# === Strategy Recommendations ===
st.subheader("💡 Strategy Recommendation")
if selected_strategy == "Scalp Trade":
    if technical_score >= 85 and sentiment_score > 60:
        st.success("⚡ **Scalp Signal Met — Quick Entry Suggested!**")
        st.write(f"Consider rapid entry given strong technicals and positive sentiment.")
        st.write(f"Suggested Stop Loss: ${stop_loss:.2f} (tight for scalping)")
    else:
        st.warning("🚫 **Weak Momentum — Not Ideal for Scalping.**")
        st.info("Scalping requires extremely strong, immediate momentum. Current signals are not sufficient.")
elif selected_strategy == "Day Trade":
    if technical_score >= 80 and sentiment_score > 50:
        st.success("📈 **Day Trade Setup Found — Confirm with Intraday Flow!**")
        st.write(f"The daily chart shows a promising setup. Look for confirming signals on smaller timeframes (e.g., 5-min, 15-min) and strong intraday volume.")
        st.write(f"Suggested Stop Loss: ${stop_loss:.2f}")
    else:
        st.info("⏳ **Wait for Intraday Confirmation or Volume Spike.**")
        st.warning("Daily signals are not strong enough for a confident day trade. Monitor for better intraday conditions.")
elif selected_strategy == "Swing Trade":
    if technical_score >= 75 and overall_confidence >= 60: # Using overall confidence for swing
        st.success("🌀 **Swing Trade Opportunity — Monitor Entry Zone!**")
        st.write(f"The technical setup looks good for a multi-day hold. Consider entering on a pull-back to support or a clear breakout retest.")
        st.write(f"Suggested Stop Loss: ${stop_loss:.2f}")
    else:
        st.warning("⚠️ **Setup Weak — Consider Watching Until More Signals Fire.**")
        st.info("Current signals are not robust enough for a high-conviction swing trade. Patience is key.")
elif selected_strategy == "Position Trade":
    if overall_confidence >= 70:
        st.success("📊 **Strong Long-Term Outlook — Position Entry Viable!**")
        st.write(f"The overall analysis suggests a favorable long-term hold. Consider building a position over time, looking for dips.")
        st.write(f"Use Weekly Support: ${support:.2f} as a key level.")
    else:
        st.info("💤 **Not Enough Alignment for Long-Term Entry.**")
        st.warning("Long-term positions require strong fundamental and technical alignment. Current conditions are not ideal.")
else:
    st.warning("❔ **Unknown strategy type — cannot generate specific recommendation.**")

# === Confidence Breakdown ===
st.subheader("💡 Confidence Breakdown")
st.progress(overall_confidence / 100)
st.write(f"✅ Technical Score: **{technical_score}/100**")
st.write(f"🔮 Sentiment Score: **{sentiment_score}/100** (Adjusted manually or by future integrations)")
st.write(f"📚 Expert Score: **{expert_score}/100** (Adjusted manually or by future integrations)")
st.write(f"📈 ML Boost Applied: **{ml_boost} points** (Bonus for strong RSI and Price > EMA200)")