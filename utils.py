# utils.py
import streamlit as st
import yfinance as yf
import pandas as pd
import ta
import requests
from bs4 import BeautifulSoup
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
from datetime import datetime

# === Constants ===
LOG_FILE = "trade_log.csv"
EXPERT_RATING_MAP = {"Strong Buy": 100, "Buy": 85, "Hold": 50, "N/A": 50, "Sell": 15, "Strong Sell": 0}
TIMEFRAME_MAP = {
    "Scalp Trading": {"period": "5d", "interval": "5m", "weights": {"technical": 0.9, "sentiment": 0.1, "expert": 0.0}},
    "Day Trading": {"period": "60d", "interval": "60m", "weights": {"technical": 0.7, "sentiment": 0.2, "expert": 0.1}},
    "Swing Trading": {"period": "1y", "interval": "1d", "weights": {"technical": 0.6, "sentiment": 0.2, "expert": 0.2}},
    "Position Trading": {"period": "5y", "interval": "1wk", "weights": {"technical": 0.4, "sentiment": 0.2, "expert": 0.4}}
}

# === Data Fetching Functions ===
@st.cache_data(ttl=900)
def get_finviz_data(ticker):
    # Finviz scraping logic...
    url = f"https://finviz.com/quote.ashx?t={ticker}"; headers = {'User-Agent': 'Mozilla/5.0'}
    try:
        response = requests.get(url, headers=headers); response.raise_for_status()
        soup = BeautifulSoup(response.text, 'html.parser'); recom_tag = soup.find('td', text='Recom'); analyst_recom = recom_tag.find_next_sibling('td').text if recom_tag else "N/A"
        headlines = [tag.text for tag in soup.findAll('a', class_='news-link-left')[:10]]
        analyzer = SentimentIntensityAnalyzer(); compound_scores = [analyzer.polarity_scores(h)['compound'] for h in headlines]
        avg_compound = sum(compound_scores) / len(compound_scores) if compound_scores else 0
        return {"recom": analyst_recom, "headlines": headlines, "sentiment_compound": avg_compound}
    except Exception: return {"recom": "N/A", "headlines": [], "sentiment_compound": 0}

# === NEW: Centralized Data Function ===
@st.cache_data(ttl=60)
def get_all_data(symbol, period, interval):
    """Fetches all primary data in one go to reduce API calls."""
    stock = yf.Ticker(symbol)
    hist = stock.history(period=period, interval=interval, auto_adjust=True)
    info = {}
    # Use a try-except block for info as it can sometimes fail
    try:
        info = stock.info
    except Exception as e:
        st.warning(f"Could not fetch company info: {e}")
    return {"hist": hist, "info": info, "stock_obj": stock} if not hist.empty else {"hist": None, "info": None, "stock_obj": None}

@st.cache_data(ttl=300)
def get_options_chain(ticker, expiry_date):
    stock_obj = yf.Ticker(ticker); options = stock_obj.option_chain(expiry_date)
    return options.calls, options.puts

def convert_compound_to_100_scale(compound_score): return int((compound_score + 1) * 50)

# === Indicator & Signal Functions ===
def calculate_indicators(df, is_intraday=False):
    # Indicator calculation logic...
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
    except AttributeError: st.warning("CCI indicator failed. Your `ta` library might be outdated.", icon="??")
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

def generate_signals(df, selection, is_intraday=False):
    # Signal generation logic...
    signals = {}; last_row = df.iloc[-1]
    if selection.get("OBV") and 'obv' in df.columns and len(df) > 10: signals["OBV Rising"] = last_row['obv'] > df['obv'].rolling(10).mean().iloc[-1]
    if selection.get("EMA Trend") and 'EMA50' in df.columns: signals["Uptrend (21>50>200 EMA)"] = last_row["EMA50"] > last_row["EMA200"] and last_row["EMA21"] > last_row["EMA50"]
    if selection.get("Ichimoku Cloud") and 'ichimoku_a' in df.columns: signals["Bullish Ichimoku"] = last_row['Close'] > last_row['ichimoku_a'] and last_row['Close'] > last_row['ichimoku_b']
    if selection.get("Parabolic SAR") and 'psar' in df.columns: signals["Bullish PSAR"] = last_row['Close'] > last_row['psar']
    if selection.get("ADX") and 'adx' in df.columns: signals["Strong Trend (ADX > 25)"] = last_row['adx'] > 25
    if selection.get("RSI Momentum") and 'RSI' in df.columns: signals["Bullish Momentum (RSI > 50)"] = last_row["RSI"] > 50
    if selection.get("Stochastic") and 'stoch_k' in df.columns: signals["Bullish Stoch Cross"] = last_row['stoch_k'] > last_row['stoch_d']
    if selection.get("CCI") and 'cci' in df.columns: signals["Bullish CCI (>0)"] = last_row['cci'] > 0
    if selection.get("ROC") and 'roc' in df.columns: signals["Positive ROC (>0)"] = last_row['roc'] > 0
    if selection.get("Volume Spike") and 'Vol_Avg_50' in df.columns: signals["Volume Spike (>1.5x Avg)"] = last_row["Volume"] > last_row["Vol_Avg_50"] * 1.5
    if selection.get("VWAP") and is_intraday and 'vwap' in df.columns: signals["Price > VWAP"] = last_row['Close'] > last_row['vwap']
    return signals

def backtest_strategy(df_historical, selection, atr_multiplier=1.5, reward_risk_ratio=2.0):
    # Backtesting logic...
    trades = []; in_trade = False
    for i in range(1, len(df_historical) - 1):
        if in_trade:
            if df_historical['Low'].iloc[i] <= stop_loss: trades.append({"Date": df_historical.index[i].strftime('%Y-%m-%d'), "Type": "Exit (Loss)", "Price": stop_loss}); in_trade = False
            elif df_historical['High'].iloc[i] >= take_profit: trades.append({"Date": df_historical.index[i].strftime('%Y-%m-%d'), "Type": "Exit (Win)", "Price": take_profit}); in_trade = False
        if not in_trade:
            row = df_historical.iloc[i-1]
            if pd.isna(row.get('EMA200')): continue
            signals = generate_signals(df_historical.iloc[:i], selection)
            if signals and all(signals.values()):
                entry_price = df_historical['Open'].iloc[i]
                stop_loss = entry_price - (row['ATR'] * atr_multiplier); take_profit = entry_price + (row['ATR'] * atr_multiplier * reward_risk_ratio)
                trades.append({"Date": df_historical.index[i].strftime('%Y-%m-%d'), "Type": "Entry", "Price": entry_price}); in_trade = True
    wins = len([t for t in trades if t['Type'] == 'Exit (Win)']); losses = len([t for t in trades if t['Type'] == 'Exit (Loss)'])
    return trades, wins, losses

def generate_option_trade_plan(ticker, confidence, stock_price, expirations):
    # Options strategy logic...
    if confidence < 60: return {"status": "warning", "message": "Confidence score is too low. No options trade is recommended."}
    today = datetime.now()
    target_exp_date = None
    for exp_str in expirations:
        exp_date = datetime.strptime(exp_str, '%Y-%m-%d')
        if 45 <= (exp_date - today).days <= 90:
            target_exp_date = exp_str
            break
    if not target_exp_date: return {"status": "warning", "message": "Could not find a suitable expiration date (45-90 days out)."}
    calls, _ = get_options_chain(ticker, target_exp_date)
    if calls.empty: return {"status": "error", "message": f"No call options found for {target_exp_date}."}
    strategy = "Buy Call"; reason = ""
    target_options = pd.DataFrame()
    if confidence >= 75:
        strategy = "Buy ITM Call"; reason = "High confidence suggests a strong directional move. An ITM call (Delta > 0.60) provides good leverage with a higher probability of success."
        target_options = calls[(calls['inTheMoney']) & (calls.get('delta', 0) > 0.60)]
    elif 60 <= confidence < 75:
        strategy = "Buy ATM Call"; reason = "Moderate confidence favors an At-the-Money call to balance cost and potential upside."
        target_options = calls.iloc[[(calls['strike'] - stock_price).abs().idxmin()]]
    if target_options.empty: target_options = calls.iloc[[(calls['strike'] - stock_price).abs().idxmin()]]; reason += " (Fell back to nearest ATM option)."
    if target_options.empty: return {"status": "error", "message": "Could not find any suitable options."}
    recommended_option = target_options.iloc[0]
    entry_price = recommended_option.get('ask', recommended_option.get('lastPrice'))
    if entry_price == 0: entry_price = recommended_option.get('lastPrice')
    risk_per_share = entry_price * 0.50; stop_loss = entry_price - risk_per_share; profit_target = entry_price + (risk_per_share * 2)
    return {"status": "success", "Strategy": strategy, "Reason": reason, "Expiration": target_exp_date,
            "Strike": f"${recommended_option['strike']:.2f}", "Entry Price": f"~${entry_price:.2f}",
            "Stop-Loss": f"~${stop_loss:.2f} (50% loss)", "Profit Target": f"~${profit_target:.2f} (100% gain)",
            "Max Risk / Share": f"${risk_per_share:.2f}", "Reward / Risk": "2 to 1", "Contract": recommended_option}