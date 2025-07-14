#==GoogleAIStudio==
# -*- coding: utf-8 -*-
# vim: set ts=4 sw=4 et:

import streamlit as st
import yfinance as yf
import pandas as pd
import ta
import mplfinance as mpf
import os

# === Page Setup ===
st.set_page_config(page_title="Aatif's Pro Analyzer", layout="wide")
st.title("📈 Aatif's Professional Trade Analyzer")

# === SIDEBAR: User Inputs ===
st.sidebar.header("⚙️ Controls")
ticker = st.sidebar.text_input("Enter a Ticker Symbol", value="NVDA").upper()
timeframe = st.sidebar.radio("Choose Trading Style:",
    ["Swing Trading", "Position Trading"],
    index=0, help="Changes chart interval and analysis focus. Backtest is always on Daily data.")

# Map styles to yfinance intervals
TIMEFRAME_MAP = {
    "Swing Trading": {"period": "1y", "interval": "1d"},
    "Position Trading": {"period": "5y", "interval": "1wk"}
}
selected_period = TIMEFRAME_MAP[timeframe]["period"]
selected_interval = TIMEFRAME_MAP[timeframe]["interval"]

# === Caching & Utility Functions ===
@st.cache_data(ttl=600)
def get_data(symbol, period, interval):
    stock = yf.Ticker(symbol)
    hist = stock.history(period=period, interval=interval)
    if hist.empty:
        return None, None
    info = stock.info
    return hist, info

def calculate_indicators(df):
    df["EMA21"] = ta.trend.ema_indicator(df["Close"], 21)
    df["EMA50"] = ta.trend.ema_indicator(df["Close"], 50)
    df["EMA200"] = ta.trend.ema_indicator(df["Close"], 200)
    df["RSI"] = ta.momentum.RSIIndicator(df["Close"]).rsi()
    bb = ta.volatility.BollingerBands(df["Close"])
    df["BB_low"] = bb.bollinger_lband()
    df["BB_high"] = bb.bollinger_hband()
    df["MACD_diff"] = ta.trend.macd_diff(df["Close"])
    df["ATR"] = ta.volatility.AverageTrueRange(df["High"], df["Low"], df["Close"]).average_true_range()
    df["Vol_Avg_50"] = df["Volume"].rolling(50).mean()
    return df

def generate_signals(last_row):
    is_uptrend = last_row["EMA50"] > last_row["EMA200"] and last_row["EMA21"] > last_row["EMA50"]
    signals = {
        "Uptrend (21>50>200 EMA)": is_uptrend,
        "Bullish Momentum (RSI > 50)": last_row["RSI"] > 50,
        "Momentum Reset (RSI 40-60)": 40 < last_row["RSI"] < 60 and is_uptrend,
        "MACD Bullish (Diff > 0)": last_row["MACD_diff"] > 0,
        "Volume Spike (>1.5x Avg)": last_row["Volume"] > last_row["Vol_Avg_50"] * 1.5,
    }
    return signals

def backtest_strategy(df, atr_multiplier=1.5, reward_risk_ratio=2.0):
    """Backtests the signal strategy with realistic entry/exit points."""
    trades = []
    in_trade = False
    
    for i in range(1, len(df) - 1):
        if in_trade:
            # Check exit conditions
            if df['Low'].iloc[i] <= stop_loss:
                trades.append({"Date": df.index[i], "Type": "Exit (Loss)", "Price": stop_loss})
                in_trade = False
            elif df['High'].iloc[i] >= take_profit:
                trades.append({"Date": df.index[i], "Type": "Exit (Win)", "Price": take_profit})
                in_trade = False

        if not in_trade:
            # Check entry conditions
            row = df.iloc[i-1] # Use previous day's data to decide
            signals = generate_signals(row)
            # Define a strong entry condition for the backtest
            if signals["Uptrend (21>50>200 EMA)"] and signals["MACD Bullish (Diff > 0)"] and signals["Momentum Reset (RSI 40-60)"]:
                entry_price = df['Open'].iloc[i] # Enter on next day's open
                stop_loss = entry_price - (row['ATR'] * atr_multiplier)
                take_profit = entry_price + (row['ATR'] * atr_multiplier * reward_risk_ratio)
                trades.append({"Date": df.index[i], "Type": "Entry", "Price": entry_price})
                in_trade = True
    
    wins = len([t for i, t in enumerate(trades) if t['Type'] == 'Exit (Win)'])
    losses = len([t for t in trades if t['Type'] == 'Exit (Loss)'])
    return trades, wins, losses

# === Main Dashboard Function ===
def display_dashboard(ticker, hist, info, timeframe):
    # --- Data Processing ---
    df = calculate_indicators(hist.copy())
    last = df.iloc[-1]
    signals = generate_signals(last)

    # --- UI LAYOUT ---
    st.header(f"Analysis for {ticker} ({timeframe})")

    # --- Create Tabs for Organization ---
    tab1, tab2, tab3, tab4 = st.tabs(["📊 Main Analysis", "🧪 Backtest", "📘 Indicator Guide", "ℹ️ Ticker Info"])

    with tab1:
        col1, col2 = st.columns([1, 2])
        with col1:
            st.subheader("💡 Confidence Score")
            technical_score = sum([1 for signal, fired in signals.items() if fired]) / len(signals) * 100
            st.metric("Technical Signal Strength", f"{technical_score:.0f}/100")
            st.progress(technical_score / 100)
            
            st.subheader("✅ Technical Checklist")
            for signal, fired in signals.items():
                st.markdown(f"- {'🟢' if fired else '🔴'} {signal}")
            
            st.subheader("🎯 Key Price Levels")
            resistance = df["High"][-60:].max()
            support = df["Low"][-60:].min()
            st.write(f"**Support (60-period):** ${support:.2f}")
            st.write(f"**Resistance (60-period):** ${resistance:.2f}")
            st.write(f"**ATR (Volatility):** {last['ATR']:.2f}")
            st.write(f"**Suggested Stop-Loss:** ${last['Close'] - 1.5 * last['ATR']:.2f}")

        with col2:
            st.subheader("📈 Price Chart")
            chart_path = f"chart_{ticker}.png"
            ap = [mpf.make_addplot(df.tail(120)[['BB_high', 'BB_low']])]
            mpf.plot(df.tail(120), type='candle', style='yahoo',
                     mav=(21, 50, 200), volume=True, addplot=ap,
                     title=f"{ticker} - {timeframe}", savefig=chart_path)
            st.image(chart_path)
            if os.path.exists(chart_path):
                os.remove(chart_path)

    with tab2:
        st.subheader(f"🧪 Historical Signal Backtest for {ticker}")
        st.info("This backtest simulates the 'Uptrend + MACD + RSI Reset' strategy on daily data over the last year with a 2:1 Reward/Risk ratio.")
        
        # Always use daily data for consistent backtesting
        daily_hist, _ = get_data(ticker, "1y", "1d")
        if daily_hist is not None:
            daily_df = calculate_indicators(daily_hist.copy())
            trades, wins, losses = backtest_strategy(daily_df)
            total_trades = wins + losses
            win_rate = (wins / total_trades) * 100 if total_trades > 0 else 0

            st.write(f"- **Trades Simulated:** {total_trades}")
            st.write(f"- ✅ **Wins:** {wins}")
            st.write(f"- ❌ **Losses:** {losses}")
            st.metric("Historical Win Rate", f"{win_rate:.1f}%")

            if trades:
                st.write("#### Trade Log")
                st.dataframe(pd.DataFrame(trades).set_index("Date"))
        else:
            st.warning("Could not fetch daily data for backtesting.")

    with tab3:
        st.subheader("📘 Indicator Guide")
        st.markdown("""
        This table explains what each indicator measures, its current value for the ticker, and the ideal condition for a bullish swing trade.
        """)
        
        indicator_data = [
            {"Indicator": "EMA Stack (21, 50, 200)", 
             "Description": "Shows trend alignment. A stacked formation (short > mid > long) confirms a strong, healthy uptrend across multiple timeframes.",
             "Current Value": f"21: {last['EMA21']:.2f} | 50: {last['EMA50']:.2f} | 200: {last['EMA200']:.2f}",
             "Ideal Bullish Value": "21 EMA > 50 EMA > 200 EMA",
             "Status": '🟢' if signals["Uptrend (21>50>200 EMA)"] else '🔴'},
            {"Indicator": "RSI (14)",
             "Description": "Measures momentum. For a pullback trade in an uptrend, we want to see the RSI 'reset' to a neutral zone, not be overbought.",
             "Current Value": f"{last['RSI']:.2f}",
             "Ideal Bullish Value": "Between 40 and 60 (in an uptrend)",
             "Status": '🟢' if signals.get("Momentum Reset (RSI 40-60)", False) else '🔴'},
            {"Indicator": "MACD Difference",
             "Description": "Highlights the direction and strength of momentum. A positive value indicates that recent momentum is stronger than past momentum.",
             "Current Value": f"{last['MACD_diff']:.2f}",
             "Ideal Bullish Value": "> 0",
             "Status": '🟢' if signals["MACD Bullish (Diff > 0)"] else '🔴'},
            {"Indicator": "Volume",
             "Description": "Confirms the conviction behind a price move. A spike in volume on an up-day shows strong institutional interest.",
             "Current Value": f"{last['Volume']:,.0f}",
             "Ideal Bullish Value": f"> {last['Vol_Avg_50']:,.0f} (1.5x Avg)",
             "Status": '🟢' if signals["Volume Spike (>1.5x Avg)"] else '🔴'}
        ]
        st.table(pd.DataFrame(indicator_data).set_index("Indicator"))

    with tab4:
        st.subheader(f"ℹ️ About {info.get('longName', ticker)}")
        st.write(f"**Sector:** {info.get('sector', 'N/A')}")
        st.write(f"**Industry:** {info.get('industry', 'N/A')}")
        st.write(f"**Website:** {info.get('website', 'N/A')}")
        st.markdown(f"**Business Summary:**")
        st.info(f"{info.get('longBusinessSummary', 'No summary available.')}")
        st.markdown(f"🔗 [Finviz](https://finviz.com/quote.ashx?t={ticker}) | [Barchart](https://www.barchart.com/stocks/quotes/{ticker}/overview) | [TipRanks](https://www.tipranks.com/stocks/{ticker}/forecast)")

# === Main Script Execution ===
if ticker:
    try:
        hist_data, info_data = get_data(ticker, selected_period, selected_interval)
        if hist_data is None:
            st.error(f"Could not fetch data for {ticker}. The symbol may be invalid or delisted.")
        else:
            display_dashboard(ticker, hist_data, info_data, timeframe)
    except Exception as e:
        st.error(f"An error occurred: {e}")
        st.warning("This may be due to yfinance API rate limits or an invalid ticker. Please wait a moment and try again.")
else:
    st.info("Please enter a stock ticker in the sidebar to begin analysis.")