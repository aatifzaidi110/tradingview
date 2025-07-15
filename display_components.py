# display_components.py

import streamlit as st
import pandas as pd
import mplfinance as mpf
import matplotlib.pyplot as plt
import yfinance as yf
import os # For chart saving/removing, though direct plot is preferred

# Import functions from utils.py
from utils import backtest_strategy, calculate_indicators, generate_signals_for_row, generate_option_trade_plan, get_options_chain, get_data, get_finviz_data

# === Helper for Indicator Display ===
def format_indicator_display(signal_key, current_value, description, ideal_value_desc, selected, signals_dict):
    """
    Formats and displays a single technical indicator's information.
    """
    if not selected:
        return ""
    
    is_fired = signals_dict.get(signal_key, False)
    status_icon = '🟢' if is_fired else '🔴'
    display_name = signal_key.split('(')[0].strip()

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

# === Dashboard Tab Display Functions ===
def display_main_analysis_tab(ticker, df, info, params, selection, overall_confidence, scores, final_weights, sentiment_score, expert_score):
    """Displays the main technical analysis and confidence score tab."""
    is_intraday = params['interval'] in ['5m', '60m']
    last = df.iloc[-1]
    signals = generate_signals_for_row(last, selection, df, is_intraday) # Recalculate for display

    col1, col2 = st.columns([1, 2])
    with col1:
        st.subheader("💡 Confidence Score")
        st.metric("Overall Confidence", f"{overall_confidence:.0f}/100")
        st.progress(overall_confidence / 100)
        st.markdown(f"- **Technical:** `{scores['technical']:.0f}` (W: `{final_weights['technical']*100:.0f}%`)\n- **Sentiment:** `{scores['sentiment']:.0f}` (W: `{final_weights['sentiment']*100:.0f}%`)\n- **Expert:** `{scores['expert']:.0f}` (W: `{final_weights['expert']*100:.0f}%`)")
        
        st.subheader("🎯 Key Price Levels")
        current_price = last['Close']
        prev_close = df['Close'].iloc[-2] if len(df) >= 2 else current_price
        price_delta = current_price - prev_close
        st.metric(label="Current Price", value=f"${current_price:.2f}", delta=f"${price_delta:.2f}")
        
        st.subheader("✅ Technical Analysis Readout")
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
                "OBV Rising", last.get('obv'),
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
        
        if not df.empty:
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
        else:
            st.info("Not enough data to generate chart.")

def display_trade_plan_options_tab(ticker, df, overall_confidence):
    """Displays the suggested trade plan and options strategy."""
    last = df.iloc[-1]

    st.subheader("📋 Suggested Stock Trade Plan (Bullish Swing)")
    entry_zone_start = last['EMA21'] * 0.99 if 'EMA21' in last and not pd.isna(last['EMA21']) else last['Close'] * 0.99
    entry_zone_end = last['EMA21'] * 1.01 if 'EMA21' in last and not pd.isna(last['EMA21']) else last['Close'] * 1.01
    
    stop_loss_val = last['Low'] - last['ATR'] if 'Low' in last and 'ATR' in last and not pd.isna(last['ATR']) and last['ATR'] > 0 else last['Close'] * 0.95
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
                col5.metric("Max Profit / Max Risk", f"{trade_plan['Max Profit']:.2f} / {trade_plan['Max Risk']:.2f}")
                st.write(f"**Reward / Risk:** `{trade_plan['Reward / Risk']}`")
                st.markdown("---")
                st.subheader("🔬 Recommended Option Deep-Dive (Spread Legs)")
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

def display_backtest_tab(ticker, selection):
    """Displays the historical backtest results."""
    st.subheader(f"🧪 Historical Backtest for {ticker}")
    st.info(f"Simulating trades based on your **currently selected indicators**. Entry is triggered if ALL selected signals are positive.")
    
    daily_hist, _ = get_data(ticker, "2y", "1d")
    if daily_hist is not None and not daily_hist.empty:
        daily_df_calculated = calculate_indicators(daily_hist.copy(), is_intraday=False)
        daily_df_calculated = daily_df_calculated.dropna() # Ensure no NaNs after indicator calculation

        if len(daily_df_calculated) < 200: # Adjust threshold as needed
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
            else: st.info("No trades were executed based on the current strategy and historical data. Try adjusting indicators or timeframes.")
    else:
        st.warning("Could not fetch daily data for backtesting or data is empty.")

def display_news_info_tab(ticker, info, finviz_data):
    """Displays news and company information."""
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

def display_trade_log_tab(LOG_FILE, ticker, timeframe, overall_confidence):
    """Displays and manages the trade log."""
    st.subheader("📝 Log Your Trade Analysis")
    user_notes = st.text_area("Add your personal notes or trade thesis here:")
    
    # === Trade Log Saving Logic (TO BE IMPLEMENTED) ===
    # This is a placeholder. You'll need to implement the actual saving and loading.
    # Example structure for saving:
    # if st.button("Save Trade Log"):
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