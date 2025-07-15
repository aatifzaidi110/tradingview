#===GoogleAIStudio 4 ====
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
from datetime import datetime, timedelta
import matplotlib.pyplot as plt

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
        # Dynamically disable/enable based on timeframe
        "VWAP": st.checkbox("VWAP (Intraday only)", value=True, disabled=(timeframe not in ["Scalp Trading", "Day Trading"])),
    })
with st.sidebar.expander("Display-Only Indicators"):
    indicator_selection.update({
        "Bollinger Bands": st.checkbox("Bollinger Bands Display", value=True),
        # Dynamically disable/enable based on timeframe
        "Pivot Points": st.checkbox("Pivot Points Display (Daily only)", value=True, disabled=(timeframe not in ["Swing Trading", "Position Trading"])),
    })

st.sidebar.header("🧠 Qualitative Scores")
use_automation = st.sidebar.toggle("Enable Automated Scoring", value=True, help="ON: AI scores are used. OFF: Use manual sliders and only the Technical Score will count.")
auto_sentiment_score_placeholder = st.sidebar.empty()
auto_expert_score_placeholder = st.sidebar.empty()

# === Core Functions ===
@st.cache_data(ttl=900)
def get_finviz_data(ticker):
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
            analyzer = SentimentIntensityAnalyser()
            compound_scores = [analyzer.polarity_scores(h)['compound'] for h in headlines]
            avg_compound = sum(compound_scores) / len(compound_scores) if compound_scores else 0
            return {"recom": analyst_recom, "headlines": headlines, "sentiment_compound": avg_compound}
    except Exception as e:
        st.error(f"Error fetching Finviz data: {e}", icon="🚫")
        return {"recom": "N/A", "headlines": [], "sentiment_compound": 0}

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
    sentiment_score = 50
    expert_score = 50
    finviz_data = {"headlines": ["Automation is disabled."]}

@st.cache_data(ttl=60)
def get_data(symbol, period, interval):
    with st.spinner(f"Fetching {period} of {interval} data for {symbol}..."):
        stock = yf.Ticker(symbol)
        hist = stock.history(period=period, interval=interval, auto_adjust=True)
        return (hist, stock.info) if not hist.empty else (None, None)

@st.cache_data(ttl=300)
def get_options_chain(ticker, expiry_date):
    with st.spinner(f"Fetching options chain for {ticker} ({expiry_date})..."):
        stock_obj = yf.Ticker(ticker)
        options = stock_obj.option_chain(expiry_date)
        return options.calls, options.puts

def calculate_indicators(df, is_intraday=False):
    # This function remains the same, with robust error handling
    try: df["EMA21"]=ta.trend.ema_indicator(df["Close"],21); df["EMA50"]=ta.trend.ema_indicator(df["Close"],50); df["EMA200"]=ta.trend.ema_indicator(df["Close"],200)
    except Exception as e: st.warning(f"Could not calculate EMA indicators: {e}", icon="⚠️")
    try: ichimoku = ta.trend.IchimokuIndicator(df['High'], df['Low']); df['ichimoku_a'] = ichimoku.ichimoku_a(); df['ichimoku_b'] = ichimoku.ichimoku_b()
    except Exception as e: st.warning(f"Could not calculate Ichimoku Cloud: {e}", icon="⚠️")
    try: df['psar'] = ta.trend.PSARIndicator(df['High'], df['Low'], df['Close']).psar()
    except Exception as e: st.warning(f"Could not calculate Parabolic SAR: {e}", icon="⚠️")
    try: df['adx'] = ta.trend.ADXIndicator(df['High'], df['Low'], df['Close']).adx()
    except Exception as e: st.warning(f"Could not calculate ADX: {e}", icon="⚠️")
    try: df["RSI"]=ta.momentum.RSIIndicator(df["Close"]).rsi()
    except Exception as e: st.warning(f"Could not calculate RSI: {e}", icon="⚠️")
    try: stoch = ta.momentum.StochasticOscillator(df['High'], df['Low'], df['Close']); df['stoch_k'] = stoch.stoch(); df['stoch_d'] = stoch.stoch_signal()
    except Exception as e: st.warning(f"Could not calculate Stochastic Oscillator: {e}", icon="⚠️")
    try: df['cci'] = ta.momentum.cci(df['High'], df['Low'], df['Close'])
    except AttributeError: st.warning("CCI indicator failed. Your `ta` library might be outdated.", icon="⚠️")
    except Exception as e: st.warning(f"Could not calculate CCI: {e}", icon="⚠️")
    try: df['roc'] = ta.momentum.ROCIndicator(df['Close']).roc()
    except Exception as e: st.warning(f"Could not calculate ROC: {e}", icon="⚠️")
    try: df['obv'] = ta.volume.OnBalanceVolumeIndicator(df['Close'], df['Volume']).on_balance_volume()
    except Exception as e: st.warning(f"Could not calculate OBV: {e}", icon="⚠️")
    if is_intraday:
        try: df['vwap'] = ta.volume.VolumeWeightedAveragePrice(df['High'], df['Low'], df['Close'], df['Volume']).volume_weighted_average_price()
        except Exception as e: st.warning(f"Could not calculate VWAP: {e}", icon="⚠️")
    
    try: df["ATR"]=ta.volatility.AverageTrueRange(df["High"],df["Low"],df["Close"]).average_true_range()
    except Exception as e: st.warning(f"Could not calculate ATR: {e}", icon="⚠️")
    
    try: bb=ta.volatility.BollingerBands(df["Close"]); df["BB_low"]=bb.bollinger_lband(); df["BB_high"]=bb.bollinger_hband()
    except Exception as e: st.warning(f"Could not calculate Bollinger Bands: {e}", icon="⚠️")
    
    try: df["Vol_Avg_50"]=df["Volume"].rolling(50).mean()
    except Exception as e: st.warning(f"Could not calculate Volume Average: {e}", icon="⚠️")
    
    return df

def calculate_pivot_points(df):
    df_pivots = pd.DataFrame(index=df.index)
    df_pivots['Pivot'] = (df['High'] + df['Low'] + df['Close']) / 3
    df_pivots['R1'] = (2 * df_pivots['Pivot']) - df['Low']
    df_pivots['S1'] = (2 * df_pivots['Pivot']) - df['High']
    df_pivots['R2'] = df_pivots['Pivot'] + (df['High'] - df['Low'])
    df_pivots['S2'] = df_pivots['Pivot'] - (df['High'] - df['Low'])
    return df_pivots.shift(1)

def generate_signals_for_row(row_data, selection, full_df=None, is_intraday=False):
    signals = {}
    
    # OBV Rolling Average (requires full_df for lookback)
    if selection.get("OBV") and 'obv' in row_data and full_df is not None and len(full_df) >= 10:
        # Get the rolling mean up to the current row's index (exclusive of current row for proper backtest)
        # Ensure that full_df.index is monotonically increasing and unique.
        try:
            current_index_loc = full_df.index.get_loc(row_data.name)
            prev_data_for_rolling = full_df.iloc[:current_index_loc] # Data strictly before current row
            if not prev_data_for_rolling.empty and 'obv' in prev_data_for_rolling.columns and len(prev_data_for_rolling) >= 10:
                signals["OBV Rising"] = row_data['obv'] > prev_data_for_rolling['obv'].rolling(10).mean().iloc[-1]
            else:
                signals["OBV Rising"] = False # Not enough data for rolling mean
        except KeyError: # Handle cases where row_data.name might not be directly in full_df.index (e.g., if full_df was sliced)
            signals["OBV Rising"] = False
    else:
        signals["OBV Rising"] = False # Default to False if not selected or conditions not met

    if selection.get("EMA Trend") and 'EMA50' in row_data:
        signals["Uptrend (21>50>200 EMA)"] = row_data["EMA50"] > row_data["EMA200"] and row_data["EMA21"] > row_data["EMA50"]
    if selection.get("Ichimoku Cloud") and 'ichimoku_a' in row_data:
        signals["Bullish Ichimoku"] = row_data['Close'] > row_data['ichimoku_a'] and row_data['Close'] > row_data['ichimoku_b']
    if selection.get("Parabolic SAR") and 'psar' in row_data:
        signals["Bullish PSAR"] = row_data['Close'] > row_data['psar']
    if selection.get("ADX") and 'adx' in row_data:
        signals["Strong Trend (ADX > 25)"] = row_data['adx'] > 25
    if selection.get("RSI Momentum") and 'RSI' in row_data:
        signals["Bullish Momentum (RSI > 50)"] = row_data["RSI"] > 50
    if selection.get("Stochastic") and 'stoch_k' in row_data:
        signals["Bullish Stoch Cross"] = row_data['stoch_k'] > row_data['stoch_d']
    if selection.get("CCI") and 'cci' in row_data:
        signals["Bullish CCI (>0)"] = row_data['cci'] > 0
    if selection.get("ROC") and 'roc' in row_data:
        signals["Positive ROC (>0)"] = row_data['roc'] > 0
    if selection.get("Volume Spike") and 'Vol_Avg_50' in row_data:
        signals["Volume Spike (>1.5x Avg)"] = row_data["Volume"] > row_data["Vol_Avg_50"] * 1.5
    if selection.get("VWAP") and is_intraday and 'vwap' in row_data:
        signals["Price > VWAP"] = row_data['Close'] > row_data['vwap']
    return signals

def backtest_strategy(df_historical_calculated, selection, atr_multiplier=1.5, reward_risk_ratio=2.0):
    trades = []
    in_trade = False
    entry_price = 0
    stop_loss = 0
    take_profit = 0

    # Ensure we have enough data for all selected indicators and lookbacks
    # For backtesting, we need at least 200 data points if EMA200 is used, or minimum for other indicators
    min_data_points = 200 # A safe minimum for typical indicators

    if len(df_historical_calculated) < min_data_points:
        return [], 0, 0 # Not enough data for backtesting

    # Start from the first index where all selected indicators have valid data
    # We iterate from the first index that has no NaN for any column that could be used for signals.
    # This might require a more sophisticated check if indicators are selectively used.
    # For simplicity, we assume df_historical_calculated.dropna() has been called.
    
    # We need at least one day's data BEFORE the entry day to generate signals.
    for i in range(1, len(df_historical_calculated)):
        current_day_data = df_historical_calculated.iloc[i]
        prev_day_data = df_historical_calculated.iloc[i-1]

        # Ensure essential columns exist and are not NaN for the previous day's data
        if pd.isna(prev_day_data.get('ATR')) or prev_day_data['ATR'] == 0:
            continue # Cannot set stop loss/take profit without valid ATR

        if in_trade:
            # Check for exit on current day's price action
            # Prioritize stop-loss over take-profit if both hit on the same day (common convention)
            if current_day_data['Low'] <= stop_loss:
                pnl = stop_loss - entry_price
                trades.append({"Date": current_day_data.name.strftime('%Y-%m-%d'), "Type": "Exit (Loss)", "Price": round(stop_loss, 2), "Entry Price": round(entry_price, 2), "PnL": round(pnl, 2)})
                in_trade = False
            elif current_day_data['High'] >= take_profit:
                pnl = take_profit - entry_price
                trades.append({"Date": current_day_data.name.strftime('%Y-%m-%d'), "Type": "Exit (Win)", "Price": round(take_profit, 2), "Entry Price": round(entry_price, 2), "PnL": round(pnl, 2)})
                in_trade = False
            # If no exit, the trade continues.

        if not in_trade:
            # Generate signals based on previous day's *fully calculated* indicators
            # Pass the full DataFrame up to the previous day for rolling calculations
            signals = generate_signals_for_row(prev_day_data, selection, df_historical_calculated.iloc[:i], is_intraday=False)

            # Check if all *selected* signals are true
            selected_and_fired_count = 0
            selected_indicator_count = 0
            for indicator_key, is_selected in selection.items():
                if is_selected and indicator_key in ["EMA Trend", "Ichimoku Cloud", "Parabolic SAR", "ADX", "RSI Momentum",
                                                     "Stochastic", "CCI", "ROC", "Volume Spike", "OBV", "VWAP"]: # Only consider actual signal indicators
                    selected_indicator_count += 1
                    if signals.get(indicator_key.replace(" (Intraday only)", "").replace(" Display", ""), False): # Map selection key to signal key
                        selected_and_fired_count += 1

            if selected_indicator_count > 0 and selected_and_fired_count == selected_indicator_count:
                # Entry on the Open of the current day (i.e., after signals from prev_day's close)
                entry_price = current_day_data['Open']
                
                # Recalculate stop loss and profit target using the current prev_day_data's ATR
                stop_loss = entry_price - (prev_day_data['ATR'] * atr_multiplier)
                take_profit = entry_price + (prev_day_data['ATR'] * atr_multiplier * reward_risk_ratio)

                trades.append({"Date": current_day_data.name.strftime('%Y-%m-%d'), "Type": "Entry", "Price": round(entry_price, 2)})
                in_trade = True

    # Handle any remaining open trades at the end of the backtest period
    if in_trade:
        final_exit_price = df_historical_calculated.iloc[-1]['Close']
        pnl = final_exit_price - entry_price
        trades.append({"Date": df_historical_calculated.index[-1].strftime('%Y-%m-%d'), "Type": "Exit (End of Backtest)", "Price": round(final_exit_price, 2), "Entry Price": round(entry_price, 2), "PnL": round(pnl, 2)})

    wins = len([t for t in trades if t['Type'] == 'Exit (Win)'])
    losses = len([t for t in trades if t['Type'] == 'Exit (Loss)'])
    return trades, wins, losses

# === NEW: Options Strategy Engine ===
def generate_option_trade_plan(ticker, confidence, stock_price, expirations):
    if confidence < 60:
        return {"status": "warning", "message": "Confidence score is too low. No options trade is recommended."}
    
    today = datetime.now()
    target_exp_date = None
    
    # Prioritize expirations between 45 and 365 days out
    suitable_expirations = []
    for exp_str in expirations:
        exp_date = datetime.strptime(exp_str, '%Y-%m-%d')
        days_to_expiry = (exp_date - today).days
        if 45 <= days_to_expiry <= 365: # Changed upper limit to 365 days (approx 1 year)
            suitable_expirations.append((days_to_expiry, exp_str))
            
    if not suitable_expirations:
        return {"status": "warning", "message": "Could not find a suitable expiration date (45-365 days out)."}
    
    # Sort by nearest expiry and pick the first one
    suitable_expirations.sort()
    target_exp_date = suitable_expirations[0][1]

    calls, _ = get_options_chain(ticker, target_exp_date)
    if calls.empty:
        return {"status": "error", "message": f"No call options found for {target_exp_date}."}

    strategy = "Buy Call"
    reason = ""
    target_options = pd.DataFrame() # Initialize empty DataFrame

    if confidence >= 75:
        # Suggest a Bull Call Spread for high confidence, if possible
        # Buy ITM Call, Sell OTM Call (with higher strike)
        itm_calls = calls[(calls['inTheMoney']) & (calls['delta'] > 0.60)].sort_values(by='strike', ascending=False)
        if not itm_calls.empty:
            buy_leg = itm_calls.iloc[0]
            
            # Find a suitable OTM call to sell (e.g., 2-5 strikes higher)
            otm_calls_for_spread = calls[(calls['strike'] > buy_leg['strike']) & (calls['inTheMoney'] == False)].sort_values(by='strike', ascending=True)
            if not otm_calls_for_spread.empty and len(otm_calls_for_spread) > 0:
                sell_leg = otm_calls_for_spread.iloc[min(2, len(otm_calls_for_spread) - 1)] # Sell 3rd OTM option as example
                strategy = "Bull Call Spread"
                reason = f"High confidence ({confidence:.0f}% bullish) suggests a strong directional move with defined risk. A Bull Call Spread limits both upside and downside, reducing premium cost."
                
                spread_cost = buy_leg.get('ask', buy_leg.get('lastPrice',0)) - sell_leg.get('bid', sell_leg.get('lastPrice',0))
                max_profit = (sell_leg['strike'] - buy_leg['strike']) - spread_cost
                max_risk = spread_cost

                if spread_cost > 0 and max_risk > 0: # Ensure valid spread
                    return {"status": "success", "Strategy": strategy, "Reason": reason, "Expiration": target_exp_date,
                            "Buy Strike": f"${buy_leg['strike']:.2f}", "Sell Strike": f"${sell_leg['strike']:.2f}",
                            "Net Debit": f"~${spread_cost:.2f}",
                            "Max Profit": f"~${max_profit:.2f}", "Max Risk": f"~${max_risk:.2f}", "Reward / Risk": f"{max_profit/max_risk:.1f} to 1" if max_risk > 0 else "N/A",
                            "Contracts": {"Buy": buy_leg, "Sell": sell_leg}}
            else:
                strategy = "Buy ITM Call" # Fallback if spread not possible
                reason = f"High confidence ({confidence:.0f}% bullish) suggests a strong directional move. An ITM call (Delta > 0.60) provides good leverage with a higher probability of success. (Bull Call Spread not feasible)."
                target_options = itm_calls
        else: # Fallback if no ITM calls for spread
            strategy = "Buy ITM Call"
            reason = f"High confidence ({confidence:.0f}% bullish) suggests a strong directional move. An ITM call (Delta > 0.60) provides good leverage with a higher probability of success. (No suitable ITM calls for spread)."
            target_options = calls[(calls['inTheMoney'])] # Just any ITM call if delta not avail

                                                        
    elif 60 <= confidence < 75:
        strategy = "Buy ATM Call"
        reason = f"Moderate confidence ({confidence:.0f}% bullish) favors an At-the-Money call to balance cost and potential upside."
        target_options = calls.iloc[[(calls['strike'] - stock_price).abs().idxmin()]]

    if target_options.empty:
        # Fallback if no ideal option is found
        target_options = calls.iloc[[(calls['strike'] - stock_price).abs().idxmin()]]
        reason += " (Fell back to nearest ATM option)."

    if target_options.empty:
        return {"status": "error", "message": "Could not find any suitable options."}

    # For single call strategies
    if strategy == "Buy ITM Call" or strategy == "Buy ATM Call":
        recommended_option = target_options.iloc[0]
        entry_price = recommended_option.get('ask', recommended_option.get('lastPrice'))
        if entry_price is None or entry_price == 0: # Ensure entry_price is valid
            entry_price = recommended_option.get('lastPrice')
            if entry_price is None or entry_price == 0:
                return {"status": "error", "message": "Could not determine a valid entry price for the recommended option."}
        
        risk_per_share = entry_price * 0.50 # 50% stop-loss
        stop_loss = entry_price - risk_per_share
        profit_target = entry_price + (risk_per_share * 2) # 2:1 Reward/Risk

        return {"status": "success", "Strategy": strategy, "Reason": reason, "Expiration": target_exp_date,
                "Strike": f"${recommended_option['strike']:.2f}", "Entry Price": f"~${entry_price:.2f}",
                "Stop-Loss": f"~${stop_loss:.2f} (50% loss)", "Profit Target": f"~${profit_target:.2f} (100% gain)",
                "Max Risk / Share": f"${risk_per_share:.2f}", "Reward / Risk": "2 to 1", "Contract": recommended_option}
    
    return {"status": "error", "message": "No suitable option strategy found."}


def display_dashboard(ticker, hist, info, params, selection):
    is_intraday = params['interval'] in ['5m', '60m']
    df = calculate_indicators(hist.copy(), is_intraday)
    signals = generate_signals_for_row(df.iloc[-1], selection, df, is_intraday) # Pass full df for rolling calculations
    last = df.iloc[-1]
    
    # Ensure signals dictionary is not empty before calculation
    technical_score = (sum(1 for f in signals.values() if f) / len(signals)) * 100 if signals else 0
    
    scores = {"technical": technical_score, "sentiment": sentiment_score, "expert": expert_score}
    final_weights = params['weights'].copy()
    if not use_automation: 
        final_weights = {'technical': 1.0, 'sentiment': 0.0, 'expert': 0.0}
        scores['sentiment'], scores['expert'] = 0, 0 # Explicitly set to 0 when automation is off
    
    overall_confidence = min(round((final_weights["technical"]*scores["technical"] + final_weights["sentiment"]*scores["sentiment"] + final_weights["expert"]*scores["expert"]), 2), 100)
    
    st.header(f"Analysis for {ticker} ({params['interval']} Interval)")
    
    tab_list = ["📊 Main Analysis", "📈 Trade Plan & Options", "🧪 Backtest", "📰 News & Info", "📝 Trade Log"]
    main_tab, trade_tab, backtest_tab, news_tab, log_tab = st.tabs(tab_list)

    with main_tab:
        col1, col2 = st.columns([1, 2])
        with col1:
            st.subheader("💡 Confidence Score")
            st.metric("Overall Confidence", f"{overall_confidence:.0f}/100")
            st.progress(overall_confidence / 100)
            st.markdown(f"- **Technical:** `{scores['technical']:.0f}` (W: `{final_weights['technical']*100:.0f}%`)\n- **Sentiment:** `{scores['sentiment']:.0f}` (W: `{final_weights['sentiment']*100:.0f}%`)\n- **Expert:** `{scores['expert']:.0f}` (W: `{final_weights['expert']*100:.0f}%`)")
            
            st.subheader("🎯 Key Price Levels")
            current_price = last['Close']
            prev_close = df['Close'].iloc[-2] if len(df) >= 2 else current_price # Robust prev_close
            price_delta = current_price - prev_close
            st.metric(label="Current Price", value=f"${current_price:.2f}", delta=f"${price_delta:.2f}")
            
            st.subheader("✅ Technical Analysis Readout") # Categorized display here...
            
            def format_indicator_display(signal_key, current_value, description, ideal_value_desc, selected, signals_dict):
                """
                Formats and displays a single technical indicator's information.
                
                Args:
                    signal_key (str): The key used in the `signals` dictionary (e.g., "Uptrend (21>50>200 EMA)").
                    current_value: The numerical value of the indicator for the latest data point. Can be None if not applicable.
                    description (str): A brief explanation of what the indicator does.
                    ideal_value_desc (str): A description of what constitutes a 'bullish' or 'ideal' value/condition.
                    selected (bool): Whether the indicator is selected by the user.
                    signals_dict (dict): The dictionary of calculated signals.
                """
                if not selected:
                    return "" # Don't display if not selected
                
                is_fired = signals_dict.get(signal_key, False)
                status_icon = '🟢' if is_fired else '🔴'
                display_name = signal_key.split('(')[0].strip() # Clean name for display

                value_str = ""
                if current_value is not None and not pd.isna(current_value):
                    if isinstance(current_value, (int, float)):
                        value_str = f"Current: `{current_value:.2f}`"
                    else:
                        value_str = "Current: N/A"
                else:
                    value_str = "Current: N/A"

                return (
                    f"{status_icon} **{display_name}**\n"
                    f"   - *Description:* {description}\n"
                    f"   - *Ideal (Bullish):* {ideal_value_desc}\n"
                    f"   - *{value_str}*\n"
                )

            with st.expander("📈 Trend Indicators", expanded=True):
                st.markdown(format_indicator_display(
                    "Uptrend (21>50>200 EMA)", None,
                    "Exponential Moving Averages (EMAs) smooth price data to identify trend direction. A bullish trend is indicated when shorter EMAs (e.g., 21-day) are above longer EMAs (e.g., 50-day), and both are above the longest EMA (e.g., 200-day).",
                    "21 EMA > 50 EMA > 200 EMA",
                    selection.get("EMA Trend"), signals
                ))
                st.markdown(format_indicator_display(
                    "Bullish Ichimoku", None,
                    "The Ichimoku Cloud is a comprehensive indicator that defines support and resistance, identifies trend direction, and gauges momentum. A bullish signal occurs when the price is above the cloud, indicating an uptrend.",
                    "Price > Ichimoku Cloud",
                    selection.get("Ichimoku Cloud"), signals
                ))
                st.markdown(format_indicator_display(
                    "Bullish PSAR", last.get('psar'),
                    "Parabolic Stop and Reverse (PSAR) is a time and price based trading system used to identify potential reversals in the price movement of traded assets. Bullish when dots are below price.",
                    "Dots below price",
                    selection.get("Parabolic SAR"), signals
                ))
                st.markdown(format_indicator_display(
                    "Strong Trend (ADX > 25)", last.get('adx'),
                    "The Average Directional Index (ADX) quantifies the strength of a trend. Values above 25 generally indicate a strong trend (either up or down), while values below 20 suggest a weak or non-trending market.",
                    "ADX > 25",
                    selection.get("ADX"), signals
                ))
            
            with st.expander("💨 Momentum & Volume Indicators", expanded=True):
                st.markdown(format_indicator_display(
                    "Bullish Momentum (RSI > 50)", last.get('RSI'),
                    "The Relative Strength Index (RSI) is a momentum oscillator measuring the speed and change of price movements. An RSI above 50 generally suggests bullish momentum, while below 50 indicates bearish momentum.",
                    "RSI > 50",
                    selection.get("RSI Momentum"), signals
                ))
                st.markdown(format_indicator_display(
                    "Bullish Stoch Cross", last.get('stoch_k'),
                    "The Stochastic Oscillator is a momentum indicator comparing a particular closing price of a security to a range of its prices over a certain period. A bullish cross occurs when %K (fast line) crosses above %D (slow line), often below 50.",
                    "%K line crosses above %D line (preferably below 50)",
                    selection.get("Stochastic"), signals
                ))
                st.markdown(format_indicator_display(
                    "Bullish CCI (>0)", last.get('cci'),
                    "The Commodity Channel Index (CCI) measures the current price level relative to an average price level over a given period. A CCI above zero generally indicates the price is above its average, suggesting an uptrend.",
                    "CCI > 0",
                    selection.get("CCI"), signals
                ))
                st.markdown(format_indicator_display(
                    "Positive ROC (>0)", last.get('roc'),
                    "Rate of Change (ROC) is a momentum indicator that measures the percentage change between the current price and the price a certain number of periods ago. A positive ROC indicates upward momentum.",
                    "ROC > 0",
                    selection.get("ROC"), signals
                ))
                st.markdown(format_indicator_display(
                    "Volume Spike (>1.5x Avg)", last.get('Volume'),
                    "A volume spike indicates an unusual increase in trading activity, which often precedes or accompanies significant price movements. A volume greater than 1.5 times the average suggests strong interest.",
                    "Volume > 1.5x 50-day Average Volume",
                    selection.get("Volume Spike"), signals
                ))
                st.markdown(format_indicator_display(
                    "OBV Rising", last.get('obv'), # Current OBV value
                    "On-Balance Volume (OBV) relates volume to price changes. A rising OBV indicates that positive volume pressure is increasing and confirms an uptrend.",
                    "OBV is rising (higher than its recent average)",
                    selection.get("OBV"), signals
                ))
                if is_intraday:
                    st.markdown(format_indicator_display(
                        "Price > VWAP", last.get('vwap'),
                        "Volume Weighted Average Price (VWAP) is a trading benchmark that represents the average price a security has traded at throughout the day, based on both volume and price. Price trading above VWAP is considered bullish.",
                        "Price > VWAP",
                        selection.get("VWAP"), signals
                    ))
        
        with col2:
            st.subheader("📈 Price Chart")
            mav_tuple = (21, 50, 200) if selection.get("EMA Trend") else None
            ap = [mpf.make_addplot(df.tail(120)[['BB_high', 'BB_low']])] if selection.get("Bollinger Bands") else None
            
            fig, axlist = mpf.plot(
                df.tail(120),
                type='candle',
                style='yahoo',
                mav=mav_tuple,
                volume=True,
                addplot=ap,
                title=f"{ticker} - {params['interval']} chart",
                returnfig=True
            )
            st.pyplot(fig, clear_figure=True)
            plt.close(fig)

    with trade_tab:
        st.subheader("📋 Suggested Stock Trade Plan (Bullish Swing)")
        entry_zone_start = last['EMA21'] * 0.99 if 'EMA21' in last else last['Close'] * 0.99
        entry_zone_end = last['EMA21'] * 1.01 if 'EMA21' in last else last['Close'] * 1.01
        
        # Ensure ATR and Low are available for stop loss/profit target
        stop_loss_val = last['Low'] - last['ATR'] if 'Low' in last and 'ATR' in last and not pd.isna(last['ATR']) else last['Close'] * 0.95
        profit_target_val = last['Close'] + (2 * (last['Close'] - stop_loss_val)) if 'Close' in last and stop_loss_val and not pd.isna(stop_loss_val) else last['Close'] * 1.1
        
        st.info(f"**Based on {overall_confidence:.0f}% Overall Confidence:**\n\n"
                f"**Entry Zone:** Between **${entry_zone_start:.2f}** and **${entry_zone_end:.2f}**.\n"
                f"**Stop-Loss:** A close below **${stop_loss_val:.2f}**.\n"
                f"**Profit Target:** Around **${profit_target_val:.2f}** (2:1 Reward/Risk).")
        st.markdown("---")
        
        st.subheader("🎭 Automated Options Strategy")
        stock_obj = yf.Ticker(ticker)
        expirations = stock_obj.options
        if not expirations:
            st.warning("No options data available for this ticker.")
        else:
            trade_plan = generate_option_trade_plan(ticker, overall_confidence, last['Close'], expirations)
            if trade_plan['status'] == 'success':
                st.success(f"**Recommended Strategy: {trade_plan['Strategy']}** (Confidence: {overall_confidence:.0f}%)")
                st.info(trade_plan['Reason'])
                
                if trade_plan['Strategy'] == "Bull Call Spread":
                    col1, col2, col3, col4, col5 = st.columns(5)
                    col1.metric("Buy Strike", trade_plan['Buy Strike'])
                    col2.metric("Sell Strike", trade_plan['Sell Strike'])
                    col3.metric("Expiration", trade_plan['Expiration'])
                    col4.metric("Net Debit", trade_plan['Net Debit'])
                    col5.metric("Max Profit / Max Risk", f"{trade_plan['Max Profit']}/{trade_plan['Max Risk']}")
                    st.write(f"**Reward / Risk:** `{trade_plan['Reward / Risk']}`")
                else: # Single Call strategies
                    col1, col2, col3, col4 = st.columns(4)
                    col1.metric("Strike", trade_plan['Strike'])
                    col2.metric("Expiration", trade_plan['Expiration'])
                    col3.metric("Entry Price", trade_plan['Entry Price'])
                    col4.metric("Reward/Risk", trade_plan['Reward / Risk'])
                    st.write(f"**Stop-Loss:** `{trade_plan['Stop-Loss']}` | **Profit Target:** `{trade_plan['Profit Target']}` | **Max Risk:** `{trade_plan['Max Risk / Share']}` per share")
                
                st.markdown("---")
                st.subheader("🔬 Recommended Option Deep-Dive")
                if trade_plan['Strategy'] == "Bull Call Spread":
                    st.write("**Buy Leg:**")
                    rec_option_buy = trade_plan['Contracts']['Buy']
                    option_metrics_buy = [
                        {"Metric": "Implied Volatility (IV)", "Description": "Market's forecast of volatility. High IV = expensive premium.", "Value": f"{rec_option_buy.get('impliedVolatility', 0):.2%}", "Ideal for Buyers": "Lower is better"},
                        {"Metric": "Delta", "Description": "Option's price change per $1 stock change.", "Value": f"{rec_option_buy.get('delta', 0):.2f}", "Ideal for Buyers": "0.60 to 0.80"},
                        {"Metric": "Theta", "Description": "Time decay. Daily value lost from the premium.", "Value": f"{rec_option_buy.get('theta', 0):.3f}", "Ideal for Buyers": "As low as possible"},
                        {"Metric": "Open Interest", "Description": "Total open contracts. High OI indicates good liquidity.", "Value": f"{rec_option_buy.get('openInterest', 0):,}", "Ideal for Buyers": "> 100s"},
                        {"Metric": "Bid", "Description": "The price buyers are willing to pay.", "Value": f"{rec_option_buy.get('bid', 0):.2f}", "Ideal for Buyers": "Lower to enter"},
                        {"Metric": "Ask", "Description": "The price sellers are willing to accept.", "Value": f"{rec_option_buy.get('ask', 0):.2f}", "Ideal for Buyers": "Lower to enter"},
                        {"Metric": "Volume (Today)", "Description": "Number of contracts traded today.", "Value": f"{rec_option_buy.get('volume', 0):,}", "Ideal for Buyers": "Higher (>100)"},
                    ]
                    st.table(pd.DataFrame(option_metrics_buy).set_index("Metric"))

                    st.write("**Sell Leg:**")
                    rec_option_sell = trade_plan['Contracts']['Sell']
                    option_metrics_sell = [
                        {"Metric": "Implied Volatility (IV)", "Description": "Market's forecast of volatility. High IV = expensive premium.", "Value": f"{rec_option_sell.get('impliedVolatility', 0):.2%}", "Ideal for Sellers": "Higher is better"},
                        {"Metric": "Delta", "Description": "Option's price change per $1 stock change.", "Value": f"{rec_option_sell.get('delta', 0):.2f}", "Ideal for Sellers": "Lower (0.20-0.40)"},
                        {"Metric": "Theta", "Description": "Time decay. Daily value lost from the premium.", "Value": f"{rec_option_sell.get('theta', 0):.3f}", "Ideal for Sellers": "Higher (more decay)"},
                        {"Metric": "Open Interest", "Description": "Total open contracts. High OI indicates good liquidity.", "Value": f"{rec_option_sell.get('openInterest', 0):,}", "Ideal for Sellers": "> 100s"},
                        {"Metric": "Bid", "Description": "The price buyers are willing to pay.", "Value": f"{rec_option_sell.get('bid', 0):.2f}", "Ideal for Sellers": "Higher to exit"},
                        {"Metric": "Ask", "Description": "The price sellers are willing to accept.", "Value": f"{rec_option_sell.get('ask', 0):.2f}", "Ideal for Sellers": "Higher to exit"},
                        {"Metric": "Volume (Today)", "Description": "Number of contracts traded today.", "Value": f"{rec_option_sell.get('volume', 0):,}", "Ideal for Sellers": "Higher (>100)"},
                    ]
                    st.table(pd.DataFrame(option_metrics_sell).set_index("Metric"))

                else: # Single Call strategy
                    rec_option = trade_plan['Contract']
                    option_metrics = [
                        {"Metric": "Implied Volatility (IV)", "Description": "Market's forecast of volatility. High IV = expensive premium.", "Value": f"{rec_option.get('impliedVolatility', 0):.2%}", "Ideal for Buyers": "Lower is better"},
                        {"Metric": "Delta", "Description": "Option's price change per $1 stock change.", "Value": f"{rec_option.get('delta', 0):.2f}", "Ideal for Buyers": "0.60 to 0.80 (for ITM calls)"},
                        {"Metric": "Theta", "Description": "Time decay. Daily value lost from the premium.", "Value": f"{rec_option.get('theta', 0):.3f}", "Ideal for Buyers": "As low as possible"},
                        {"Metric": "Open Interest", "Description": "Total open contracts. High OI indicates good liquidity.", "Value": f"{rec_option.get('openInterest', 0):,}", "Ideal for Buyers": "> 100s"},
                        {"Metric": "Bid", "Description": "The price buyers are willing to pay.", "Value": f"{rec_option.get('bid', 0):.2f}", "Ideal for Buyers": "Lower to enter"},
                        {"Metric": "Ask", "Description": "The price sellers are willing to accept.", "Value": f"{rec_option.get('ask', 0):.2f}", "Ideal for Buyers": "Lower to enter"},
                        {"Metric": "Volume (Today)", "Description": "Number of contracts traded today.", "Value": f"{rec_option.get('volume', 0):,}", "Ideal for Buyers": "Higher (>100)"},
                    ]
                    st.table(pd.DataFrame(option_metrics).set_index("Metric"))
            else:
                st.warning(trade_plan['message'])
            
            st.markdown("---")
            st.subheader("⛓️ Full Option Chain")
            option_type = st.radio("Select Option Type", ["Calls", "Puts"], horizontal=True)
            exp_date_str = st.selectbox("Select Expiration Date to View", expirations)
            if exp_date_str:
                calls, puts = get_options_chain(ticker, exp_date_str)
                # Removed the general get_options_suggestion, as detailed plan is above
                st.markdown(f"[**🔗 Analyze this chain on OptionCharts.io**](https://optioncharts.io/options/{ticker}/chain/{exp_date_str})")
                
                chain_to_display = calls if option_type == "Calls" else puts
                desired_cols = ['strike', 'lastPrice', 'bid', 'ask', 'volume', 'openInterest', 'impliedVolatility', 'inTheMoney', 'delta', 'theta', 'gamma', 'vega', 'rho']
                available_cols = [col for col in desired_cols if col in chain_to_display.columns]
                if available_cols: st.dataframe(chain_to_display[available_cols].set_index('strike'))

    with backtest_tab:
        st.subheader(f"🧪 Historical Backtest for {ticker}")
        st.info(f"Simulating trades based on your **currently selected indicators**. Entry is triggered if ALL selected signals are positive.")
        
        daily_hist, _ = get_data(ticker, "2y", "1d")
        if daily_hist is not None and not daily_hist.empty:
            daily_df_calculated = calculate_indicators(daily_hist.copy(), is_intraday=False) # Ensure not intraday for daily data
            # Drop initial NaNs that result from indicator calculations
            daily_df_calculated = daily_df_calculated.dropna()

            if len(daily_df_calculated) < 200: # Adjust threshold as needed based on your longest indicator period
                st.warning("Not enough complete historical data for robust backtesting after indicator calculation. (Need at least 200 data points after NaN removal).")
            else:
                trades, wins, losses = backtest_strategy(daily_df_calculated, selection)
                total_trades = wins + losses
                win_rate = (wins / total_trades) * 100 if total_trades > 0 else 0
                
                col1, col2, col3 = st.columns(3)
                col1.metric("Trades Simulated", total_trades)
                col2.metric("Wins", wins)
                col3.metric("Win Rate", f"{win_rate:.1f}%")
                
                if trades: st.dataframe(pd.DataFrame(trades).tail(20))
                else: st.info("No trades were executed based on the current strategy and historical data.")
        else:
            st.warning("Could not fetch daily data for backtesting or data is empty.")

    with news_tab:
        st.subheader(f"📰 News & Information for {ticker}")
        col1, col2 = st.columns(2)
        with col1:
            st.markdown("#### ℹ️ Company Info")
            st.write(f"**Name:** {info.get('longName', ticker)}")
            st.write(f"**Sector:** {info.get('sector', 'N/A')}")
            st.markdown("#### 🔗 External Research Links")
            st.markdown(f"- [Yahoo Finance]({info.get('website', 'https://finance.yahoo.com')}) | [Finviz](https://finviz.com/quote.ashx?t={ticker})")
        with col2:
            st.markdown("#### 📅 Company Calendar")
            stock_obj_for_cal = yf.Ticker(ticker)
            if stock_obj_for_cal and hasattr(stock_obj_for_cal, 'calendar') and isinstance(stock_obj_for_cal.calendar, pd.DataFrame) and not stock_obj_for_cal.calendar.empty: 
                st.dataframe(stock_obj_for_cal.calendar.T)
            else: 
                st.info("No upcoming calendar events found.")
        st.markdown("#### 🗞️ Latest Headlines")
        for h in finviz_data['headlines']:
            st.markdown(f"_{h}_")
                        
    with log_tab:
        st.subheader("📝 Log Your Trade Analysis")
        user_notes = st.text_area("Add your personal notes or trade thesis here:")
        
        # === Trade Log Saving Logic (TO BE IMPLEMENTED) ===
        # Example structure for saving:
        # if st.button("Save Trade Log"):
        #     # Ensure LOG_FILE exists
        #     if not os.path.exists(LOG_FILE):
        #         pd.DataFrame(columns=["Date", "Ticker", "Timeframe", "Confidence", "Notes"]).to_csv(LOG_FILE, index=False)
        #     
        #     current_date = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        #     new_log_entry = pd.DataFrame([{
        #         "Date": current_date,
        #         "Ticker": ticker,
        #         "Timeframe": timeframe,
        #         "Confidence": f"{overall_confidence:.0f}",
        #         "Notes": user_notes
        #     }])
        #     new_log_entry.to_csv(LOG_FILE, mode='a', header=False, index=False)
        #     st.success("Trade log saved!")
        #     st.dataframe(pd.read_csv(LOG_FILE)) # Display the updated log
        # else:
        #     if os.path.exists(LOG_FILE):
        #         st.dataframe(pd.read_csv(LOG_FILE))
        #     else:
        #         st.info("No trade logs yet. Save your first entry!")
        st.info("Trade log functionality is pending implementation.")

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
        if hist_data is None or info_data is None:
            st.error(f"Could not fetch data for {ticker} on a {selected_params_main['interval']} interval. Please check the ticker symbol or try again later.")
        else:
            display_dashboard(ticker, hist_data, info_data, selected_params_main, indicator_selection)
    except Exception as e:
        st.error(f"An unexpected error occurred during data processing: {e}", icon="🚫")
        st.exception(e) # Display the full traceback for debugging
else:
    st.info("Please enter a stock ticker in the sidebar to begin analysis.")