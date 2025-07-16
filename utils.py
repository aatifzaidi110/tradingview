# utils.py

import streamlit as st
import yfinance as yf
import pandas as pd
import ta
import requests
from bs4 import BeautifulSoup
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
from datetime import datetime, timedelta

# === Data Fetching Functions ===
@st.cache_data(ttl=900)
def get_finviz_data(ticker):
    """Fetches analyst recommendations and news sentiment from Finviz."""
    url = f"https://finviz.com/quote.ashx?t={ticker}"
    headers = {'User-Agent': 'Mozilla/5.0'}
    try:
        with st.spinner(f"Fetching Finviz data for {ticker}..."):
            response = requests.get(url, headers=headers)
            response.raise_for_status()
            soup = BeautifulSoup(response.text, 'html.parser')
            recom_tag = soup.find('td', text='Recom')
            analyst_recom = recom_tag.find_next_sibling('td').text if recom_tag else "N/A"
            headlines = [tag.text for tag in soup.findAll('a', class_='news-link-left')[:10]]
            analyzer = SentimentIntensityAnalyzer()
            compound_scores = [analyzer.polarity_scores(h)['compound'] for h in headlines]
            avg_compound = sum(compound_scores) / len(compound_scores) if compound_scores else 0
            return {"recom": analyst_recom, "headlines": headlines, "sentiment_compound": avg_compound}
    except Exception as e:
        st.error(f"Error fetching Finviz data: {e}", icon="🚫")
        return {"recom": "N/A", "headlines": [], "sentiment_compound": 0}

@st.cache_data(ttl=60)
def get_data(symbol, period, interval):
    """Fetches historical stock data and basic info from Yahoo Finance."""
    with st.spinner(f"Fetching {period} of {interval} data for {symbol}..."):
        stock = yf.Ticker(symbol)
        hist = stock.history(period=period, interval=interval, auto_adjust=True)
        return (hist, stock.info) if not hist.empty else (None, None)

@st.cache_data(ttl=300)
def get_options_chain(ticker, expiry_date):
    """Fetches call and put options data for a given ticker and expiry."""
    with st.spinner(f"Fetching options chain for {ticker} ({expiry_date})..."):
        stock_obj = yf.Ticker(ticker)
        options = stock_obj.option_chain(expiry_date)
        return options.calls, options.puts

# === Indicator Calculation Functions ===
def calculate_indicators(df, is_intraday=False):
    """Calculates various technical indicators for a given DataFrame."""
    required_cols = ['Open', 'High', 'Low', 'Close', 'Volume']
    if not all(col in df.columns for col in required_cols):
        st.error(f"Missing one or more required columns for indicator calculation: {required_cols}", icon="🚫")
        return df

    df_cleaned = df.dropna(subset=['High', 'Low', 'Close', 'Volume']).copy()
    if df_cleaned.empty:
        st.warning("DataFrame is empty after dropping NaN values, cannot calculate indicators.", icon="⚠️")
        return df_cleaned

    # Use .loc for safe assignment to avoid SettingWithCopyWarning
    try: df_cleaned.loc[:, "EMA21"]=ta.trend.ema_indicator(df_cleaned["Close"],21); df_cleaned.loc[:, "EMA50"]=ta.trend.ema_indicator(df_cleaned["Close"],50); df_cleaned.loc[:, "EMA200"]=ta.trend.ema_indicator(df_cleaned["Close"],200)
    except Exception as e: st.warning(f"Could not calculate EMA indicators: {e}", icon="⚠️")
    try: ichimoku = ta.trend.IchimokuIndicator(df_cleaned['High'], df_cleaned['Low'], window1=9, window2=26, window3=52, window4=26); df_cleaned.loc[:, 'ichimoku_a'] = ichimoku.ichimoku_a(); df_cleaned.loc[:, 'ichimoku_b'] = ichimoku.ichimoku_b()
    except Exception as e: st.warning(f"Could not calculate Ichimoku Cloud: {e}", icon="⚠️")
    try: df_cleaned.loc[:, 'psar'] = ta.trend.PSARIndicator(df_cleaned['High'], df_cleaned['Low'], df_cleaned['Close']).psar()
    except Exception as e: st.warning(f"Could not calculate Parabolic SAR: {e}", icon="⚠️")
    try: df_cleaned.loc[:, 'adx'] = ta.trend.ADXIndicator(df_cleaned['High'], df_cleaned['Low'], df_cleaned['Close']).adx()
    except Exception as e: st.warning(f"Could not calculate ADX: {e}", icon="⚠️")
    try: df_cleaned.loc[:, "RSI"]=ta.momentum.RSIIndicator(df_cleaned["Close"]).rsi()
    except Exception as e: st.warning(f"Could not calculate RSI: {e}", icon="⚠️")
    try: stoch = ta.momentum.StochasticOscillator(df_cleaned['High'], df_cleaned['Low'], df_cleaned['Close']); df_cleaned.loc[:, 'stoch_k'] = stoch.stoch(); df_cleaned.loc[:, 'stoch_d'] = stoch.stoch_signal()
    except Exception as e: st.warning(f"Could not calculate Stochastic Oscillator: {e}", icon="⚠️")
    try:
        if not (df_cleaned['High'] == df_cleaned['Low']).all() and not (df_cleaned['High'] == df_cleaned['Close']).all():
            df_cleaned.loc[:, 'cci'] = ta.momentum.cci(df_cleaned['High'], df_cleaned['Low'], df_cleaned['Close'])
        else:
            st.warning("CCI cannot be calculated due to invariant High/Low/Close prices.", icon="⚠️")
    except Exception as e: st.warning(f"Could not calculate CCI: {e}", icon="⚠️")
    try: df_cleaned.loc[:, 'roc'] = ta.momentum.ROCIndicator(df_cleaned['Close']).roc()
    except Exception as e: st.warning(f"Could not calculate ROC: {e}", icon="⚠️")
    try: df_cleaned.loc[:, 'obv'] = ta.volume.OnBalanceVolumeIndicator(df_cleaned['Close'], df_cleaned['Volume']).on_balance_volume()
    except Exception as e: st.warning(f"Could not calculate OBV: {e}", icon="⚠️")
    if is_intraday:
        try: df_cleaned.loc[:, 'vwap'] = ta.volume.VolumeWeightedAveragePrice(df_cleaned['High'], df_cleaned['Low'], df_cleaned['Close'], df_cleaned['Volume']).volume_weighted_average_price()
        except Exception as e: st.warning(f"Could not calculate VWAP: {e}", icon="⚠️")
    
    try: df_cleaned.loc[:, "ATR"]=ta.volatility.AverageTrueRange(df_cleaned["High"],df_cleaned["Low"],df_cleaned["Close"]).average_true_range()
    except Exception as e: st.warning(f"Could not calculate ATR: {e}", icon="⚠️")
    
    try: bb=ta.volatility.BollingerBands(df_cleaned["Close"]); df_cleaned.loc[:, "BB_low"]=bb.bollinger_lband(); df_cleaned.loc[:, "BB_high"]=bb.bollinger_hband()
    except Exception as e: st.warning(f"Could not calculate Bollinger Bands: {e}", icon="⚠️")
    
    try: df_cleaned.loc[:, "Vol_Avg_50"]=df_cleaned["Volume"].rolling(50).mean()
    except Exception as e: st.warning(f"Could not calculate Volume Average: {e}", icon="⚠️")
    
    return df_cleaned

def calculate_pivot_points(df):
    """Calculates classical pivot points for a DataFrame."""
    if df.empty or not all(col in df.columns for col in ['High', 'Low', 'Close']):
        return pd.DataFrame(index=df.index)

    df_pivots = pd.DataFrame(index=df.index)
    df_pivots['Pivot'] = (df['High'] + df['Low'] + df['Close']) / 3
    df_pivots['R1'] = (2 * df_pivots['Pivot']) - df['Low']
    df_pivots['S1'] = (2 * df_pivots['Pivot']) - df['High']
    df_pivots['R2'] = df_pivots['Pivot'] + (df['High'] - df['Low'])
    df_pivots['S2'] = df_pivots['Pivot'] - (df['High'] - df['Low'])
    return df_pivots.shift(1)

# === Signal Generation ===
def generate_signals_for_row(row_data, selection, full_df=None, is_intraday=False):
    """Generates bullish/bearish signals for a single row of data based on selected indicators."""
    signals = {}
    
    if selection.get("OBV") and 'obv' in row_data and full_df is not None and len(full_df) >= 10:
        try:
            current_index_loc = full_df.index.get_loc(row_data.name)
            if current_index_loc >= 10:
                prev_data_for_rolling = full_df.iloc[current_index_loc - 10 : current_index_loc]
                if not prev_data_for_rolling.empty and 'obv' in prev_data_for_rolling.columns:
                    signals["OBV Rising"] = row_data['obv'] > prev_data_for_rolling['obv'].rolling(10).mean().iloc[-1]
                else:
                    signals["OBV Rising"] = False
            else:
                signals["OBV Rising"] = False
        except KeyError:
            signals["OBV Rising"] = False
    else:
        signals["OBV Rising"] = False

    if selection.get("EMA Trend") and 'EMA50' in row_data and not pd.isna(row_data["EMA50"]):
        signals["Uptrend (21>50>200 EMA)"] = row_data["EMA50"] > row_data["EMA200"] and row_data["EMA21"] > row_data["EMA50"]
    if selection.get("Ichimoku Cloud") and 'ichimoku_a' in row_data and not pd.isna(row_data["ichimoku_a"]):
        signals["Bullish Ichimoku"] = row_data['Close'] > row_data['ichimoku_a'] and row_data['Close'] > row_data['ichimoku_b']
    if selection.get("Parabolic SAR") and 'psar' in row_data and not pd.isna(row_data["psar"]):
        signals["Bullish PSAR"] = row_data['Close'] > row_data['psar']
    if selection.get("ADX") and 'adx' in row_data and not pd.isna(row_data["adx"]):
        signals["Strong Trend (ADX > 25)"] = row_data['adx'] > 25
    if selection.get("RSI Momentum") and 'RSI' in row_data and not pd.isna(row_data["RSI"]):
        signals["Bullish Momentum (RSI > 50)"] = row_data["RSI"] > 50
    if selection.get("Stochastic") and 'stoch_k' in row_data and not pd.isna(row_data["stoch_k"]):
        signals["Bullish Stoch Cross"] = row_data['stoch_k'] > row_data['stoch_d']
    if selection.get("CCI") and 'cci' in row_data and not pd.isna(row_data["cci"]):
        signals["Bullish CCI (>0)"] = row_data['cci'] > 0
    if selection.get("ROC") and 'roc' in row_data and not pd.isna(row_data["roc"]):
        signals["Positive ROC (>0)"] = row_data['roc'] > 0
    if selection.get("Volume Spike") and 'Vol_Avg_50' in row_data and not pd.isna(row_data["Vol_Avg_50"]):
        signals["Volume Spike (>1.5x Avg)"] = row_data["Volume"] > row_data["Vol_Avg_50"] * 1.5
    if selection.get("VWAP") and is_intraday and 'vwap' in row_data and not pd.isna(row_data["vwap"]):
        signals["Price > VWAP"] = row_data['Close'] > row_data['vwap']
    return signals

# === Backtesting Logic ===
def backtest_strategy(df_historical_calculated, selection, atr_multiplier=1.5, reward_risk_ratio=2.0):
    """
    Simulates trades based on selected indicators and a simple entry/exit strategy.
    Assumes df_historical_calculated has all indicators pre-calculated and NaNs handled.
    """
    trades = []
    in_trade = False
    entry_price = 0
    stop_loss = 0
    take_profit = 0

    min_data_points_for_backtest = 200 # Safe minimum for EMA200 and other lookbacks

    if len(df_historical_calculated) < min_data_points_for_backtest + 1:
        st.info(f"Not enough complete historical data for robust backtesting after indicator calculation. (Need at least {min_data_points_for_backtest+1} data points after NaN removal). Found: {len(df_historical_calculated)}")
        return [], 0, 0

    required_cols_for_signals = [
        col for key, selected in selection.items() if selected for col in {
            "EMA Trend": ["EMA21", "EMA50", "EMA200"], "Ichimoku Cloud": ["ichimoku_a", "ichimoku_b"],
            "Parabolic SAR": ["psar"], "ADX": ["adx"], "RSI Momentum": ["RSI"],
            "Stochastic": ["stoch_k", "stoch_d"], "CCI": ["cci"], "ROC": ["roc"],
            "Volume Spike": ["Volume", "Vol_Avg_50"], "OBV": ["obv"], "VWAP": ["vwap"]
        }.get(key, [])
    ]
    if "ATR" not in required_cols_for_signals: required_cols_for_signals.append("ATR")
    # In utils.py, inside backtest_strategy
# ...
print("Columns in df_historical_calculated:", df_historical_calculated.columns)
print("Required columns for signals:", required_cols_for_signals)
# ...
first_valid_idx = df_historical_calculated[required_cols_for_signals].first_valid_index()
    
    first_valid_idx = df_historical_calculated[required_cols_for_signals].first_valid_index()
    if first_valid_idx is None:
        st.warning("No valid data points found after indicator calculation for backtesting.", icon="⚠️")
        return [], 0, 0

    start_i = df_historical_calculated.index.get_loc(first_valid_idx)
    if start_i == 0: start_i = 1

    for i in range(start_i, len(df_historical_calculated)):
        current_day_data = df_historical_calculated.iloc[i]
        prev_day_data = df_historical_calculated.iloc[i-1]

        if pd.isna(prev_day_data.get('ATR')) or prev_day_data['ATR'] == 0:
            continue

        if in_trade:
            if current_day_data['Low'] <= stop_loss:
                pnl = stop_loss - entry_price
                trades.append({"Date": current_day_data.name.strftime('%Y-%m-%d'), "Type": "Exit (Loss)", "Price": round(stop_loss, 2), "Entry Price": round(entry_price, 2), "PnL": round(pnl, 2)})
                in_trade = False
            elif current_day_data['High'] >= take_profit:
                pnl = take_profit - entry_price
                trades.append({"Date": current_day_data.name.strftime('%Y-%m-%d'), "Type": "Exit (Win)", "Price": round(take_profit, 2), "Entry Price": round(entry_price, 2), "PnL": round(pnl, 2)})
                in_trade = False

        if not in_trade:
            signals = generate_signals_for_row(prev_day_data, selection, df_historical_calculated.iloc[:i], is_intraday=False)

            selected_and_fired_count = 0
            selected_indicator_count = 0
            signal_indicator_keys = [k for k in selection.keys() if k not in ["Bollinger Bands", "Pivot Points", "VWAP"]]

            for indicator_key in signal_indicator_keys:
                if selection.get(indicator_key):
                    selected_indicator_count += 1
                    actual_signal_name = ""
                    if indicator_key == "EMA Trend": actual_signal_name = "Uptrend (21>50>200 EMA)"
                    elif indicator_key == "Ichimoku Cloud": actual_signal_name = "Bullish Ichimoku"
                    elif indicator_key == "Parabolic SAR": actual_signal_name = "Bullish PSAR"
                    elif indicator_key == "ADX": actual_signal_name = "Strong Trend (ADX > 25)"
                    elif indicator_key == "RSI Momentum": actual_signal_name = "Bullish Momentum (RSI > 50)"
                    elif indicator_key == "Stochastic": actual_signal_name = "Bullish Stoch Cross"
                    elif indicator_key == "CCI": actual_signal_name = "Bullish CCI (>0)"
                    elif indicator_key == "ROC": actual_signal_name = "Positive ROC (>0)"
                    elif indicator_key == "Volume Spike": actual_signal_name = "Volume Spike (>1.5x Avg)"
                    elif indicator_key == "OBV": actual_signal_name = "OBV Rising"
                    elif indicator_key == "VWAP": actual_signal_name = "Price > VWAP" # VWAP is handled by is_intraday in generate_signals_for_row

                    if actual_signal_name and signals.get(actual_signal_name, False):
                        selected_and_fired_count += 1
            
            if selected_indicator_count > 0 and selected_and_fired_count == selected_indicator_count:
                entry_price = current_day_data['Open']
                if not pd.isna(prev_day_data['ATR']) and prev_day_data['ATR'] > 0:
                    stop_loss = entry_price - (prev_day_data['ATR'] * atr_multiplier)
                    take_profit = entry_price + (prev_day_data['ATR'] * atr_multiplier * reward_risk_ratio)
                    trades.append({"Date": current_day_data.name.strftime('%Y-%m-%d'), "Type": "Entry", "Price": round(entry_price, 2)})
                    in_trade = True

    if in_trade:
        final_exit_price = df_historical_calculated.iloc[-1]['Close']
        pnl = final_exit_price - entry_price
        trades.append({"Date": df_historical_calculated.index[-1].strftime('%Y-%m-%d'), "Type": "Exit (End of Backtest)", "Price": round(final_exit_price, 2), "Entry Price": round(entry_price, 2), "PnL": round(pnl, 2)})

    wins = len([t for t in trades if t['Type'] == 'Exit (Win)'])
    losses = len([t for t in trades if t['Type'] == 'Exit (Loss)'])
    return trades, wins, losses

# === Options Strategy Logic ===
EXPERT_RATING_MAP = {"Strong Buy": 100, "Buy": 85, "Hold": 50, "N/A": 50, "Sell": 15, "Strong Sell": 0}

def convert_compound_to_100_scale(compound_score): return int((compound_score + 1) * 50)

def generate_option_trade_plan(ticker, confidence, stock_price, expirations):
    """Generates an options trade plan based on confidence and available expirations."""
    if confidence < 60:
        return {"status": "warning", "message": "Confidence score is too low. No options trade is recommended."}
    
    today = datetime.now()
    suitable_expirations = []
    for exp_str in expirations:
        exp_date = datetime.strptime(exp_str, '%Y-%m-%d')
        days_to_expiry = (exp_date - today).days
        if 45 <= days_to_expiry <= 365: # Up to 1 year
            suitable_expirations.append((days_to_expiry, exp_str))
            
    if not suitable_expirations:
        return {"status": "warning", "message": "Could not find a suitable expiration date (45-365 days out)."}
    
    suitable_expirations.sort()
    target_exp_date = suitable_expirations[0][1]

    calls, _ = get_options_chain(ticker, target_exp_date)
    if calls.empty:
        return {"status": "error", "message": f"No call options found for {target_exp_date}."}

    strategy = "Buy Call"
    reason = ""
    target_options = pd.DataFrame()

    if confidence >= 75:
        # Attempt Bull Call Spread
        itm_calls = calls[(calls['inTheMoney']) & (calls['delta'] > 0.60)].sort_values(by='strike', ascending=False)
        if not itm_calls.empty:
            buy_leg = itm_calls.iloc[0]
            otm_calls_for_spread = calls[(calls['strike'] > buy_leg['strike']) & (calls['inTheMoney'] == False)].sort_values(by='strike', ascending=True)
            otm_calls_for_spread = otm_calls_for_spread[(otm_calls_for_spread['volume'] > 5) | (otm_calls_for_spread['openInterest'] > 10)]

            if not otm_calls_for_spread.empty and len(otm_calls_for_spread) > 0:
                sell_leg = None
                for j in range(1, min(len(otm_calls_for_spread), 5)):
                    if otm_calls_for_spread.iloc[j]['strike'] > buy_leg['strike'] * 1.02:
                        sell_leg = otm_calls_for_spread.iloc[j]
                        break
                
                if sell_leg is not None:
                    strategy = "Bull Call Spread"
                    reason = f"High confidence ({confidence:.0f}% bullish) suggests a strong directional move with defined risk. A Bull Call Spread limits both upside and downside, reducing premium cost."
                    
                    buy_price = buy_leg.get('ask', buy_leg.get('lastPrice', 0))
                    sell_price = sell_leg.get('bid', sell_leg.get('lastPrice', 0))

                    if buy_price == 0 or sell_price == 0:
                         strategy = "Buy ITM Call" # Fallback
                         reason = f"High confidence ({confidence:.0f}% bullish) suggests a strong directional move. An ITM call (Delta > 0.60) provides good leverage with a higher probability of success. (Bull Call Spread not feasible due to illiquid prices)."
                         target_options = itm_calls
                    else:
                        spread_cost = buy_price - sell_price
                        strike_difference = sell_leg['strike'] - buy_leg['strike']
                        max_profit = (strike_difference) - spread_cost
                        max_risk = spread_cost

                        if spread_cost > 0 and max_risk > 0 and strike_difference > 0:
                            return {"status": "success", "Strategy": strategy, "Reason": reason, "Expiration": target_exp_date,
                                    "Buy Strike": f"${buy_leg['strike']:.2f}", "Sell Strike": f"${sell_leg['strike']:.2f}",
                                    "Net Debit": f"~${spread_cost:.2f}",
                                    "Max Profit": f"~${max_profit:.2f}", "Max Risk": f"~${max_risk:.2f}", "Reward / Risk": f"{max_profit/max_risk:.1f} to 1" if max_risk > 0 else "N/A",
                                    "Contracts": {"Buy": buy_leg, "Sell": sell_leg}}
                        else:
                            strategy = "Buy ITM Call" # Fallback
                            reason = f"High confidence ({confidence:.0f}% bullish) suggests a strong directional move. An ITM call (Delta > 0.60) provides good leverage with a higher probability of success. (Bull Call Spread not feasible due to invalid spread metrics)."
                            target_options = itm_calls
                else: # No suitable sell leg found
                    strategy = "Buy ITM Call"
                    reason = f"High confidence ({confidence:.0f}% bullish) suggests a strong directional move. An ITM call (Delta > 0.60) provides good leverage with a higher probability of success. (No suitable OTM calls for spread found)."
                    target_options = itm_calls
            else: # No OTM calls for spread
                strategy = "Buy ITM Call"
                reason = f"High confidence ({confidence:.0f}% bullish) suggests a strong directional move. An ITM call (Delta > 0.60) provides good leverage with a higher probability of success. (No suitable OTM calls for spread to create spread)."
                target_options = itm_calls
        else: # No ITM calls for spread
            strategy = "Buy ITM Call"
            reason = f"High confidence ({confidence:.0f}% bullish) suggests a strong directional move. An ITM call (Delta > 0.60) provides good leverage with a higher probability of success. (No suitable ITM calls for spread)."
            target_options = calls[(calls['inTheMoney'])]

    elif 60 <= confidence < 75:
        strategy = "Buy ATM Call"
        reason = f"Moderate confidence ({confidence:.0f}% bullish) favors an At-the-Money call to balance cost and potential upside."
        if 'strike' in calls.columns:
            target_options = calls.iloc[[(calls['strike'] - stock_price).abs().idxmin()]]
        else:
            return {"status": "error", "message": "Could not find strike price column in options data."}

    if target_options.empty:
        if 'strike' in calls.columns:
            target_options = calls.iloc[[(calls['strike'] - stock_price).abs().idxmin()]]
            reason += " (Fell back to nearest ATM option)."
        else:
            return {"status": "error", "message": "Could not find any suitable options or strike price column is missing."}

    if target_options.empty:
        return {"status": "error", "message": "Could not find any suitable options."}

    if strategy == "Buy ITM Call" or strategy == "Buy ATM Call":
        recommended_option = target_options.iloc[0]
        entry_price = recommended_option.get('ask', recommended_option.get('lastPrice'))
        if entry_price is None or entry_price == 0:
            entry_price = recommended_option.get('lastPrice')
            if entry_price is None or entry_price == 0:
                return {"status": "error", "message": "Could not determine a valid entry price for the recommended option."}
        
        risk_per_share = entry_price * 0.50
        stop_loss = entry_price - risk_per_share
        profit_target = entry_price + (risk_per_share * 2)

        return {"status": "success", "Strategy": strategy, "Reason": reason, "Expiration": target_exp_date,
                "Strike": f"${recommended_option['strike']:.2f}", "Entry Price": f"~${entry_price:.2f}",
                "Stop-Loss": f"~${stop_loss:.2f} (50% loss)", "Profit Target": f"~${profit_target:.2f} (100% gain)",
                "Max Risk / Share": f"${risk_per_share:.2f}", "Reward / Risk": "2 to 1", "Contract": recommended_option}
    
    return {"status": "error", "message": "No suitable option strategy found."}

def get_options_suggestion(confidence, stock_price, calls_df):
    """Placeholder for more specific options chain suggestions."""
    if calls_df.empty:
        return "warning", "No call options available for detailed suggestion.", "", None

    if confidence >= 75:
        if 'inTheMoney' in calls_df.columns and 'volume' in calls_df.columns:
            itm_calls = calls_df[calls_df['inTheMoney'] & (calls_df['volume'] > 10)]
            if not itm_calls.empty:
                target_call = itm_calls.iloc[0]
                return "success", f"High Confidence ({confidence:.0f}%): Consider a deep In-The-Money (ITM) call for strong directional play.", "Look for calls with high delta and good liquidity.", target_call
        return "info", f"High Confidence ({confidence:.0f}%), but specific ITM call not found.", "Consider ATM calls or further research.", None
    elif 60 <= confidence < 75:
        atm_call = calls_df.iloc[[(calls_df['strike'] - stock_price).abs().idxmin()]]
        if not atm_call.empty:
            return "info", f"Moderate Confidence ({confidence:.0f}%): An At-The-Money (ATM) call balances cost and potential upside.", "This is a good general strategy for moderate bullishness.", atm_call.iloc[0]
        return "warning", f"Moderate Confidence ({confidence:.0f}%), but ATM call not found.", "Consider OTM calls or re-evaluate.", None
    else:
        return "warning", f"Low Confidence ({confidence:.0f}%): Options trading is not recommended at this time due to low overall confidence.", "Focus on further analysis or paper trading.", None