# utils.py
import streamlit as st
import yfinance as yf
import pandas as pd
import ta
import requests
from bs4 import BeautifulSoup
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
from datetime import datetime, timedelta

# === Constants ===
LOG_FILE = "trade_log.csv"
EXPERT_RATING_MAP = {"Strong Buy": 100, "Buy": 85, "Hold": 50, "N/A": 50, "Sell": 15, "Strong Sell": 0}

# === Data Fetching Functions ===
@st.cache_data(ttl=900)
def get_finviz_data(ticker):
    url = f"https://finviz.com/quote.ashx?t={ticker}"; headers = {'User-Agent': 'Mozilla/5.0'}
    try:
        response = requests.get(url, headers=headers); response.raise_for_status()
        soup = BeautifulSoup(response.text, 'html.parser'); recom_tag = soup.find('td', text='Recom'); analyst_recom = recom_tag.find_next_sibling('td').text if recom_tag else "N/A"
        headlines = [tag.text for tag in soup.findAll('a', class_='news-link-left')[:10]]
        analyzer = SentimentIntensityAnalyzer(); compound_scores = [analyzer.polarity_scores(h)['compound'] for h in headlines]
        avg_compound = sum(compound_scores) / len(compound_scores) if compound_scores else 0
        return {"recom": analyst_recom, "headlines": headlines, "sentiment_compound": avg_compound}
    except Exception: return {"recom": "N/A", "headlines": [], "sentiment_compound": 0}

@st.cache_data(ttl=60)
def get_data(symbol, period, interval):
    stock = yf.Ticker(symbol); hist = stock.history(period=period, interval=interval, auto_adjust=True)
    return (hist, stock.info) if not hist.empty else (None, None)

@st.cache_data(ttl=300)
def get_options_chain(ticker, expiry_date):
    stock_obj = yf.Ticker(ticker); options = stock_obj.option_chain(expiry_date)
    return options.calls, options.puts

# === Calculation Functions ===
def convert_compound_to_100_scale(compound_score): return int((compound_score + 1) * 50)

def calculate_indicators(df, is_intraday=False):
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

def generate_signals_for_row(row_data, selection, full_df, is_intraday=False):
    signals = {}
    if selection.get("OBV") and 'obv' in row_data and not pd.isna(row_data['obv']) and len(full_df) > 10:
        signals["OBV Rising"] = row_data['obv'] > full_df['obv'].rolling(10).mean().loc[row_data.name]
    if selection.get("EMA Trend") and 'EMA50' in row_data and not pd.isna(row_data['EMA50']):
        signals["Uptrend (21>50>200 EMA)"] = row_data.get("EMA50", 0) > row_data.get("EMA200", 0) and row_data.get("EMA21", 0) > row_data.get("EMA50", 0)
    if selection.get("Ichimoku Cloud") and 'ichimoku_a' in row_data: signals["Bullish Ichimoku"] = row_data['Close'] > row_data['ichimoku_a'] and row_data['Close'] > row_data['ichimoku_b']
    if selection.get("Parabolic SAR") and 'psar' in row_data: signals["Bullish PSAR"] = row_data['Close'] > row_data['psar']
    if selection.get("ADX") and 'adx' in row_data: signals["Strong Trend (ADX > 25)"] = row_data['adx'] > 25
    if selection.get("RSI Momentum") and 'RSI' in row_data: signals["Bullish Momentum (RSI > 50)"] = row_data["RSI"] > 50
    if selection.get("Stochastic") and 'stoch_k' in row_data: signals["Bullish Stoch Cross"] = row_data['stoch_k'] > row_data['stoch_d']
    if selection.get("CCI") and 'cci' in row_data: signals["Bullish CCI (>0)"] = row_data['cci'] > 0
    if selection.get("ROC") and 'roc' in row_data: signals["Positive ROC (>0)"] = row_data['roc'] > 0
    if selection.get("Volume Spike") and 'Vol_Avg_50' in row_data: signals["Volume Spike (>1.5x Avg)"] = row_data["Volume"] > row_data["Vol_Avg_50"] * 1.5
    if selection.get("VWAP") and is_intraday and 'vwap' in row_data: signals["Price > VWAP"] = row_data['Close'] > row_data['vwap']
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
            signals = generate_signals_for_row(row, selection, df_historical.iloc[:i])
            if signals and all(signals.values()):
                entry_price = df_historical['Open'].iloc[i]
                stop_loss = entry_price - (row['ATR'] * atr_multiplier); take_profit = entry_price + (row['ATR'] * atr_multiplier * reward_risk_ratio)
                trades.append({"Date": df_historical.index[i].strftime('%Y-%m-%d'), "Type": "Entry", "Price": entry_price}); in_trade = True
    wins = len([t for t in trades if t['Type'] == 'Exit (Win)']); losses = len([t for t in trades if t['Type'] == 'Exit (Loss)'])
    return trades, wins, losses

def generate_option_trade_plan(ticker, confidence, stock_price, expirations):
    if confidence < 60: return {"status": "warning", "message": "Confidence score is too low."}
    today = datetime.now()
    target_exp_date = next((exp for exp in expirations if 45 <= (datetime.strptime(exp, '%Y-%m-%d') - today).days <= 90), None)
    if not target_exp_date: return {"status": "warning", "message": "No suitable expiration found (45-90 days)."}
    
    calls, _ = get_options_chain(ticker, target_exp_date)
    if calls.empty: return {"status": "error", "message": f"No call options for {target_exp_date}."}
    
    strategy = "Buy ATM Call"; reason = "Moderate confidence favors an At-the-Money call to balance cost and potential."
    target_options = calls.iloc[[(calls['strike'] - stock_price).abs().idxmin()]]
    if confidence >= 75:
        strategy = "Buy ITM Call"; reason = "High confidence suggests a directional play. An ITM call (Delta > 0.60) offers good leverage."
        itm_options = calls[(calls['inTheMoney']) & (calls.get('delta', 0) > 0.60)]
        if not itm_options.empty: target_options = itm_options
    
    if target_options.empty: return {"status": "error", "message": "Could not find a suitable option contract."}
    
    rec_option = target_options.iloc[0]
    entry_price = rec_option.get('ask', rec_option.get('lastPrice', 0))
    if entry_price == 0: entry_price = rec_option.get('lastPrice')
    if not isinstance(entry_price, (int, float)) or entry_price == 0: return {"status": "error", "message": "Could not determine a valid entry price for the option."}
    
    risk_per_share = entry_price * 0.50; stop_loss = entry_price - risk_per_share; profit_target = entry_price + (risk_per_share * 2)
    return {"status": "success", "Strategy": strategy, "Reason": reason, "Expiration": target_exp_date,
            "Strike": f"${rec_option['strike']:.2f}", "Entry Price": f"~${entry_price:.2f}",
            "Stop-Loss": f"~${stop_loss:.2f}", "Profit Target": f"~${profit_target:.2f}",
            "Max Risk / Share": f"${risk_per_share:.2f}", "Reward / Risk": "2 to 1", "Contract": rec_option}

def log_analysis(log_file, log_data):
    log_df = pd.DataFrame([log_data])
    file_exists = os.path.isfile(log_file)
    log_df.to_csv(log_file, mode='a', header=not file_exists, index=False)
    st.success(f"Analysis for {log_data['Ticker']} logged successfully!")

# The file should end here. The extra 'return' statement was the error.