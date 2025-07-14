import streamlit as st
import yfinance as yf
import pandas as pd
import ta
import mplfinance as mpf
import matplotlib.pyplot as plt


st.set_page_config(page_title="Aatif's Swing Dashboard", layout="centered")
st.title("📊 Aatif's Swing Trade Analyzer")

# === Ticker Input ===
ticker = st.text_input("Enter a Ticker Symbol", value="NVDA")

# === Utility Functions ===
def status(flag): return "✅" if flag else "❌"
def color_status(flag): return "🟢 Green" if flag else "🔴 Red"

#====Calculate Confidenece====
def calculate_confidence(technical, sentiment, expert, weights=None):
    if weights is None:
        weights = {"technical": 0.6, "sentiment": 0.2, "expert": 0.2}
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
    weights = {"technical": 0.6, "sentiment": 0.2, "expert": 0.2}
  # Calculate final confidence score
    overall_confidence = calculate_confidence(technical_score, sentiment_score, expert_score, weights)


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
#=====Confidence Scoring Table=====
    st.markdown("### 🧮 Confidence Scoring Table")
    st.markdown(f"""
| **Component**       | **Weight (%)** | **Raw Score** | **Contribution** |
|---------------------|----------------|---------------|------------------|
| **Technical Score** | {weights['technical']*100:.0f}% | {technical_score}/100 | {weights['technical']*technical_score:.1f} |
| **Sentiment Score** | {weights['sentiment']*100:.0f}% | {sentiment_score}/100 | {weights['sentiment']*sentiment_score:.1f} |
| **Expert Score**    | {weights['expert']*100:.0f}%    | {expert_score}/100 | {weights['expert']*expert_score:.1f} |
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

fig, ax = plt.subplots()
colors = ["#4CAF50", "#2196F3", "#FFC107"]
ax.pie(
    contributions,
    labels=[f"{labels[i]} ({contributions[i]})" for i in range(3)],
    autopct="%1.1f%%",
    startangle=90,
    colors=colors
)
ax.axis("equal")
st.pyplot(fig)


#=======Backtest function========
def backtest_signals(df, atr_multiplier=1.0, reward_multiplier=2.0):
    trades = []
    for i in range(60, len(df) - 5):  # Skip initial buffer
        row = df.iloc[i]
        next_rows = df.iloc[i+1:i+5]

        if (
            row["RSI"] < 30 or row["RSI"] > 50 and
            row["MACD_diff"] > 0 and
            row["EMA21"] > row["EMA50"] > row["EMA200"]
        ):
            entry = row["Close"]
            stop = entry - row["ATR"] * atr_multiplier
            target = entry + row["ATR"] * reward_multiplier
            exit = entry
            result = "Neutral"

            for r in next_rows.itertuples():
                if r.Low < stop:
                    exit = stop
                    result = "Loss"
                    break
                elif r.High > target:
                    exit = target
                    result = "Win"
                    break

            trades.append({
                "Entry": round(entry, 2),
                "Exit": round(exit, 2),
                "Result": result
            })
    return trades

   # === Backtest Summary ===
trades = backtest_signals(df)
wins = sum(1 for t in trades if t["Result"] == "Win")
losses = sum(1 for t in trades if t["Result"] == "Loss")
total = len(trades)
win_rate = round((wins / total) * 100, 2) if total else 0

st.subheader("🧪 Historical Signal Backtest")
st.write(f"- 📈 Total Trades Simulated: **{total}**")
st.write(f"- ✅ Wins: {wins}, ❌ Losses: {losses}")
st.write(f"- 🏆 Win Rate: **{win_rate}%**")



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

# === Dynamic Confidence Weights
if selected_strategy == "Scalp Trade":
    weights = {"technical": 0.5, "sentiment": 0.3, "expert": 0.2}
elif selected_strategy == "Day Trade":
    weights = {"technical": 0.6, "sentiment": 0.25, "expert": 0.15}
elif selected_strategy == "Swing Trade":
    weights = {"technical": 0.6, "sentiment": 0.2, "expert": 0.2}
elif selected_strategy == "Position Trade":
    weights = {"technical": 0.4, "sentiment": 0.2, "expert": 0.4}
else:
    weights = {"technical": 0.6, "sentiment": 0.2, "expert": 0.2}


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
#=====Journaling Function =======

if "journal" not in st.session_state:
    st.session_state["journal"] = []

if st.button("📝 Log This Trade Setup"):
    journal_entry = {
        "Ticker": ticker.upper(),
        "Date": pd.Timestamp.now().strftime("%Y-%m-%d %H:%M"),
        "Strategy": selected_strategy,
        "Confidence": overall_confidence,
        "Signals Fired": [k for k in signals if signals[k]],
        "Recommendation": selected_strategy  # could be expanded later
    }
    st.session_state["journal"].append(journal_entry)
    st.success(f"✅ Trade for {ticker.upper()} logged successfully!")
#==== Journaling View======
if st.session_state["journal"]:
    st.subheader("📚 Logged Trade Setups")
    st.dataframe(pd.DataFrame(st.session_state["journal"]))
