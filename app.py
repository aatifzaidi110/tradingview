#===GoogleAIStudio 2 ====
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

@st.cache_data(ttl=300)
def get_options_chain(ticker, expiry_date):
    stock_obj = yf.Ticker(ticker); options = stock_obj.option_chain(expiry_date)
    return options.calls, options.puts

def calculate_indicators(df, is_intraday=False):
    # This function remains the same, with robust error handling
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

def calculate_pivot_points(df):
    df_pivots = pd.DataFrame(index=df.index)
    df_pivots['Pivot'] = (df['High'] + df['Low'] + df['Close']) / 3
    df_pivots['R1'] = (2 * df_pivots['Pivot']) - df['Low']; df_pivots['S1'] = (2 * df_pivots['Pivot']) - df['High']
    df_pivots['R2'] = df_pivots['Pivot'] + (df['High'] - df['Low']); df_pivots['S2'] = df_pivots['Pivot'] - (df['High'] - df['Low'])
    return df_pivots.shift(1)	
	
def generate_signals(df, selection, is_intraday=False):
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

# === NEW: Options Strategy Engine ===
def generate_option_trade_plan(ticker, confidence, stock_price, expirations):
    if confidence < 60:
        return {"status": "warning", "message": "Confidence score is too low. No options trade is recommended."}
    
    today = datetime.now()
    target_exp_date = None
    for exp_str in expirations:
        exp_date = datetime.strptime(exp_str, '%Y-%m-%d')
        if 45 <= (exp_date - today).days <= 90:
            target_exp_date = exp_str
            break
    if not target_exp_date:
        return {"status": "warning", "message": "Could not find a suitable expiration date (45-90 days out)."}

    calls, _ = get_options_chain(ticker, target_exp_date)
    if calls.empty:
        return {"status": "error", "message": f"No call options found for {target_exp_date}."}

    strategy = "Buy Call"; reason = ""
    target_options = pd.DataFrame() # Initialize empty DataFrame
    if confidence >= 75:
        strategy = "Buy ITM Call"; reason = "High confidence suggests a strong directional move. An ITM call (Delta > 0.60) provides good leverage with a higher probability of success."
        target_options = calls[(calls['inTheMoney']) & (calls.get('delta', 0) > 0.60)]
														 
    elif 60 <= confidence < 75:
        strategy = "Buy ATM Call"; reason = "Moderate confidence favors an At-the-Money call to balance cost and potential upside."
        target_options = calls.iloc[[(calls['strike'] - stock_price).abs().idxmin()]]

    if target_options.empty:
        # Fallback if no ideal option is found
        target_options = calls.iloc[[(calls['strike'] - stock_price).abs().idxmin()]]
        reason += " (Fell back to nearest ATM option)."

    recommended_option = target_options.iloc[0]
    entry_price = recommended_option.get('ask', recommended_option.get('lastPrice'))
    if entry_price == 0: entry_price = recommended_option.get('lastPrice')
    
    risk_per_share = entry_price * 0.50 # 50% stop-loss
    stop_loss = entry_price - risk_per_share
    profit_target = entry_price + (risk_per_share * 2) # 2:1 Reward/Risk

    return {"status": "success", "Strategy": strategy, "Reason": reason, "Expiration": target_exp_date,
            "Strike": f"${recommended_option['strike']:.2f}", "Entry Price": f"~${entry_price:.2f}",
            "Stop-Loss": f"~${stop_loss:.2f} (50% loss)", "Profit Target": f"~${profit_target:.2f} (100% gain)",
            "Max Risk / Share": f"${risk_per_share:.2f}", "Reward / Risk": "2 to 1", "Contract": recommended_option}

def display_dashboard(ticker, hist, info, params, selection):
    is_intraday = params['interval'] in ['5m', '60m']
    df = calculate_indicators(hist.copy(), is_intraday); signals = generate_signals(df, selection, is_intraday); last = df.iloc[-1]
    technical_score = (sum(1 for f in signals.values() if f) / len(signals)) * 100 if signals else 0
    scores = {"technical": technical_score, "sentiment": sentiment_score, "expert": expert_score}
    final_weights = params['weights'].copy()
    if not use_automation: final_weights = {'technical': 1.0, 'sentiment': 0.0, 'expert': 0.0}; scores['sentiment'], scores['expert'] = 0, 0
    overall_confidence = min(round((final_weights["technical"]*scores["technical"] + final_weights["sentiment"]*scores["sentiment"] + final_weights["expert"]*scores["expert"]), 2), 100)
    
    st.header(f"Analysis for {ticker} ({params['interval']} Interval)")
    
    tab_list = ["📊 Main Analysis", "📈 Trade Plan & Options", "🧪 Backtest", "📰 News & Info", "📝 Trade Log"]
    main_tab, trade_tab, backtest_tab, news_tab, log_tab = st.tabs(tab_list)

    with main_tab:
        # Main analysis content...
        col1, col2 = st.columns([1, 2])
        with col1:
            st.subheader("💡 Confidence Score"); st.metric("Overall Confidence", f"{overall_confidence:.0f}/100"); st.progress(overall_confidence / 100)
            st.markdown(f"- **Technical:** `{scores['technical']:.0f}` (W: `{final_weights['technical']*100:.0f}%`)\n- **Sentiment:** `{scores['sentiment']:.0f}` (W: `{final_weights['sentiment']*100:.0f}%`)\n- **Expert:** `{scores['expert']:.0f}` (W: `{final_weights['expert']*100:.0f}%`)")
			st.subheader("🎯 Key Price Levels"); current_price = last['Close']; prev_close = df['Close'].iloc[-2]; price_delta = current_price - prev_close
            st.metric(label="Current Price", value=f"${current_price:.2f}", delta=f"${price_delta:.2f}")																																					 																									
            st.subheader("✅ Technical Analysis Readout") # Categorized display here...
			with st.expander("📈 Trend Indicators", expanded=True):
                def format_value(signal_name, value):
                    is_fired = signals.get(signal_name, False); status_icon = '🟢' if is_fired else '🔴'
                    name = signal_name.split('(')[0].strip(); value_str = f"`{value:.2f}`" if isinstance(value, (int, float)) else ""
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
        with col2:
            st.subheader("📈 Price Chart"); chart_path = f"chart_{ticker}.png"
			mav_tuple = (21, 50, 200) if selection.get("EMA Trend") else None
            ap = [mpf.make_addplot(df.tail(120)[['BB_high', 'BB_low']])] if selection.get("Bollinger Bands") else None
            mpf.plot(df.tail(120), type='candle', style='yahoo', mav=mav_tuple, volume=True, addplot=ap, title=f"{ticker} - {params['interval']} chart", savefig=chart_path)
            st.image(chart_path); os.remove(chart_path)																 																												  

    with trade_tab: # === FIX: Restored and Enhanced Options Analysis ===
        st.subheader("🎭 Automated Options Strategy")
		st.subheader("📋 Suggested Stock Trade Plan (Bullish Swing)")
        entry_zone_start = last['EMA21'] * 0.99; entry_zone_end = last['EMA21'] * 1.01
        stop_loss = last['Low'] - last['ATR']; profit_target = last['Close'] + (2 * (last['Close'] - stop_loss))
        st.info(f"**Entry Zone:** Between **${entry_zone_start:.2f}** and **${entry_zone_end:.2f}**.\n"
                f"**Stop-Loss:** A close below **${stop_loss:.2f}**.\n"
                f"**Profit Target:** Around **${profit_target:.2f}** (2:1 Reward/Risk).")
        st.markdown("---")
        st.subheader("🎭 Options Analysis")																			  											 
        stock_obj = yf.Ticker(ticker); expirations = stock_obj.options
        if not expirations: st.warning("No options data available for this ticker.")
        else:
            trade_plan = generate_option_trade_plan(ticker, overall_confidence, last['Close'], expirations)
            if trade_plan['status'] == 'success':
                st.success(f"**Recommended Strategy: {trade_plan['Strategy']}**")
                st.info(trade_plan['Reason'])
                col1, col2, col3, col4 = st.columns(4)
                col1.metric("Strike", trade_plan['Strike'])
                col2.metric("Expiration", trade_plan['Expiration'])
                col3.metric("Entry Price", trade_plan['Entry Price'])
                col4.metric("Reward/Risk", trade_plan['Reward / Risk'])
                st.write(f"**Stop-Loss:** `{trade_plan['Stop-Loss']}` | **Profit Target:** `{trade_plan['Profit Target']}` | **Max Risk:** `{trade_plan['Max Risk / Share']}` per share")
                
                st.markdown("---")
                st.subheader("🔬 Recommended Option Deep-Dive")
                rec_option = trade_plan['Contract']
                option_metrics = [
                    {"Metric": "Implied Volatility (IV)", "Description": "Market's forecast of volatility. High IV = expensive premium.", "Value": f"{rec_option.get('impliedVolatility', 0):.2%}", "Ideal for Buyers": "Lower is better"},
                    {"Metric": "Delta", "Description": "Option's price change per $1 stock change.", "Value": f"{rec_option.get('delta', 0):.2f}", "Ideal for Buyers": "0.60 to 0.80 (for ITM calls)"},
                    {"Metric": "Theta", "Description": "Time decay. Daily value lost from the premium.", "Value": f"{rec_option.get('theta', 0):.3f}", "Ideal for Buyers": "As low as possible"},
                    {"Metric": "Open Interest", "Description": "Total open contracts. High OI indicates good liquidity.", "Value": f"{rec_option.get('openInterest', 0):,}", "Ideal for Buyers": "> 100s"},
                ]
                st.table(pd.DataFrame(option_metrics).set_index("Metric"))
            else: st.warning(trade_plan['message'])
            st.markdown("---")
            st.subheader("⛓️ Full Option Chain")
            option_type = st.radio("Select Option Type", ["Calls", "Puts"], horizontal=True)
            exp_date_str = st.selectbox("Select Expiration Date to View", expirations)
            if exp_date_str:
                calls, puts = get_options_chain(ticker, exp_date_str)
				rec_type, suggestion, reason, target_call = get_options_suggestion(overall_confidence, last['Close'], calls)
                if rec_type == "success": st.success(suggestion)
                elif rec_type == "info": st.info(suggestion)
                else: st.warning(suggestion)
                st.write(reason)
                if target_call is not None: st.write("**Example Target Option:**"); st.json(target_call.to_dict())
                st.markdown(f"[**🔗 Analyze this chain on OptionCharts.io**](https://optioncharts.io/options/{ticker}/chain/{exp_date_str})")
                option_type = st.radio("Select Option Type to View", ["Calls", "Puts"], horizontal=True)																																																					
                chain_to_display = calls if option_type == "Calls" else puts
                desired_cols = ['strike', 'lastPrice', 'volume', 'openInterest', 'impliedVolatility', 'inTheMoney', 'delta', 'theta']
                available_cols = [col for col in desired_cols if col in chain_to_display.columns]
                if available_cols: st.dataframe(chain_to_display[available_cols].set_index('strike'))

    with backtest_tab: # === FIX: Restored Backtest ===
        st.subheader(f"🧪 Historical Backtest for {ticker}"); st.info(f"Simulating trades based on your **currently selected indicators**. Entry is triggered if ALL selected signals are positive.")
        daily_hist, _ = get_data(ticker, "2y", "1d")
        if daily_hist is not None:
            daily_df = calculate_indicators(daily_hist.copy()); trades, wins, losses = backtest_strategy(daily_df, selection)
            total_trades = wins + losses; win_rate = (wins / total_trades) * 100 if total_trades > 0 else 0
            col1, col2, col3 = st.columns(3); col1.metric("Trades Simulated", total_trades); col2.metric("Wins", wins); col3.metric("Win Rate", f"{win_rate:.1f}%")
            if trades: st.dataframe(pd.DataFrame(trades).tail(20))
        else: st.warning("Could not fetch daily data for backtesting.")

    with news_tab:
        st.subheader(f"📰 News & Information for {ticker}"); col1, col2 = st.columns(2)
        with col1:
            st.markdown("#### ℹ️ Company Info"); st.write(f"**Name:** {info.get('longName', ticker)}"); st.write(f"**Sector:** {info.get('sector', 'N/A')}")
            st.markdown("#### 🔗 External Research Links"); st.markdown(f"- [Yahoo Finance]({info.get('website', 'https://finance.yahoo.com')}) | [Finviz](https://finviz.com/quote.ashx?t={ticker})")
        with col2:
            st.markdown("#### 📅 Company Calendar")
            stock_obj_for_cal = yf.Ticker(ticker)
            if stock_obj_for_cal and hasattr(stock_obj_for_cal, 'calendar') and isinstance(stock_obj_for_cal.calendar, pd.DataFrame) and not stock_obj_for_cal.calendar.empty: st.dataframe(stock_obj_for_cal.calendar.T)
            else: st.info("No upcoming calendar events found.")
        st.markdown("#### 🗞️ Latest Headlines"); [st.markdown(f"_{h}_") for h in finviz_data['headlines']]
																
            
    with log_tab:
        st.subheader("📝 Log Your Trade Analysis"); user_notes = st.text_area("Add your personal notes or trade thesis here:")
        # ... log saving logic ...

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