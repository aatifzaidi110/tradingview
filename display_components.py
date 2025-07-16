# display_components.py - Version 1.8

import streamlit as st
import pandas as pd
import mplfinance as mpf
import matplotlib.pyplot as plt
import yfinance as yf # Ensure yfinance is imported here
import os # For chart saving/removing, though direct plot is preferred

# Import functions from utils.py
from utils import backtest_strategy, calculate_indicators, generate_signals_for_row, generate_option_trade_plan, get_options_chain, get_data, get_finviz_data, calculate_pivot_points # Import calculate_pivot_points here

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
        if isinstance(current_value, (int, float)) and not isinstance(current_value, bool): # Exclude boolean from float formatting
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
def display_main_analysis_tab(ticker, df, info, params, selection, overall_confidence, scores, final_weights, sentiment_score, expert_score, df_pivots, show_finviz_link):
    """Displays the main technical analysis and confidence score tab."""
    is_intraday = params['interval'] in ['5m', '60m']
    last = df.iloc[-1]
    signals = generate_signals_for_row(last, selection, df, is_intraday) # Recalculate for display

    col1, col2 = st.columns([1, 2])
    with col1:
        st.subheader("💡 Confidence Score")
        st.metric("Overall Confidence", f"{overall_confidence:.0f}/100")
        st.progress(overall_confidence / 100)
        
        sentiment_display = f"`{sentiment_score:.0f}`" if sentiment_score is not None else "N/A (Excluded)"
        expert_display = f"`{expert_score:.0f}`" if expert_score is not None else "N/A (Excluded)"

        st.markdown(f"- **Technical Score:** `{scores['technical']:.0f}` (Weight: `{final_weights['technical']*100:.0f}%`)\n"
                    f"- **Sentiment Score:** {sentiment_display} (Weight: `{final_weights['sentiment']*100:.0f}%`)\n"
                    f"- **Expert Rating:** {expert_display} (Weight: `{final_weights['expert']*100:.0f}%`)")
        
        if show_finviz_link:
            st.markdown(f"**Source for Sentiment & Expert Scores:** [Finviz.com]({f'https://finviz.com/quote.ashx?t={ticker}'})")

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
            # Ichimoku Cloud is disabled, so no display here
            # st.markdown(format_indicator_display(
            #     "Bullish Ichimoku", None,
            #     "The Ichimoku Cloud is a comprehensive indicator that defines support and resistance, identifies trend direction, and gauges momentum. A bullish signal occurs when the price is above the cloud, indicating an uptrend.",
            #     "Price > Ichimoku Cloud",
            #     selection.get("Ichimoku Cloud"), signals
            # ))
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
                "%K line crosses above %D (preferably below 50)",
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
        
        with st.expander("📊 Display-Only Indicators Status"):
            # Bollinger Bands Status
            if selection.get("Bollinger Bands"):
                if 'BB_high' in last and 'BB_low' in last and not pd.isna(last['BB_high']) and not pd.isna(last['BB_low']):
                    if last['Close'] > last['BB_high']:
                        bb_status = '🔴 **Price Above Upper Band** (Overbought/Strong Uptrend)'
                    elif last['Close'] < last['BB_low']:
                        bb_status = '🟢 **Price Below Lower Band** (Oversold/Strong Downtrend)'
                    else:
                        bb_status = '🟡 **Price Within Bands** (Neutral/Consolidation)'
                    st.markdown(f"**Bollinger Bands:** {bb_status}")
                else:
                    st.info("Bollinger Bands data not available for display.")

            # Pivot Points Status
            if selection.get("Pivot Points") and not is_intraday: # Pivot Points are for daily/weekly
                if not df_pivots.empty and len(df_pivots) > 1:
                    last_pivot = df_pivots.iloc[-1] # This is the pivot for the current day (calculated from previous day's data)
                    if 'Pivot' in last_pivot and not pd.isna(last_pivot['Pivot']):
                        if last['Close'] > last_pivot['R1']:
                            pivot_status = '🟢 **Price Above R1** (Strong Bullish)'
                        elif last['Close'] > last_pivot['Pivot']:
                            pivot_status = '🟡 **Price Above Pivot** (Bullish)'
                        elif last['Close'] < last_pivot['S1']:
                            pivot_status = '🔴 **Price Below S1** (Strong Bearish)'
                        elif last['Close'] < last_pivot['Pivot']:
                            pivot_status = '🟡 **Price Below Pivot** (Bearish)'
                        else:
                            pivot_status = '⚪ **Price Near Pivot** (Neutral/Ranging)'
                        st.markdown(f"**Pivot Points:** {pivot_status}")
                    else:
                        st.info("Pivot Points data not fully available for display.")
                else:
                    st.info("Pivot Points data not available for display or not enough history.")
            elif selection.get("Pivot Points") and is_intraday:
                st.info("Pivot Points are typically used for daily/weekly timeframes, not intraday.")

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
        
        # --- Start of new detailed options display ---
        if trade_plan['status'] == 'success':
            st.success(f"**Recommended Strategy: {trade_plan['Strategy']}** (Confidence: {overall_confidence:.0f}%)")
            st.info(trade_plan['Reason'])
            
            if trade_plan['Strategy'] == "Bull Call Spread":
                col1, col2, col3, col4, col5 = st.columns(5)
                col1.metric("Buy Strike", trade_plan['Buy Strike'])
                col2.metric("Sell Strike", trade_plan['Sell Strike'])
                col3.metric("Expiration", trade_plan['Expiration'])
                col4.metric("Net Debit", trade_plan['Net Debit'])
                col5.metric("Max Profit / Max Risk", f"{trade_plan['Max Profit']} / {trade_plan['Max Risk']}")
                st.write(f"**Reward / Risk:** `{trade_plan['Reward / Risk']}`")
                st.markdown("---")
                st.subheader("🔬 Recommended Option Deep-Dive (Spread Legs)")
                
                st.write("**Buy Leg:**")
                rec_option_buy = trade_plan['Contracts']['Buy']
                # Use .get() with a default value of None, then check for None or pd.isna
                option_metrics_buy = [
                    {"Metric": "Strike", "Value": f"${rec_option_buy.get('strike', 0):.2f}", "Description": "The price at which the option can be exercised.", "Ideal for Buyers": "Lower for calls, higher for puts"},
                    {"Metric": "Expiration", "Value": trade_plan['Expiration'], "Description": "Date the option expires.", "Ideal for Buyers": "Longer term (45-365 days)"},
                    {"Metric": f"Value ({ticker})", "Value": f"${rec_option_buy.get('lastPrice', None):.2f}" if rec_option_buy.get('lastPrice') is not None and not pd.isna(rec_option_buy.get('lastPrice')) else "N/A", "Description": "The last traded price of the option.", "Ideal for Buyers": "Lower to enter"},
                    {"Metric": "Bid", "Value": f"${rec_option_buy.get('bid', None):.2f}" if rec_option_buy.get('bid') is not None and not pd.isna(rec_option_buy.get('bid')) else "N/A", "Description": "Highest price a buyer is willing to pay.", "Ideal for Buyers": "Lower to enter"},
                    {"Metric": "Ask", "Value": f"${rec_option_buy.get('ask', None):.2f}" if rec_option_buy.get('ask') is not None and not pd.isna(rec_option_buy.get('ask')) else "N/A", "Description": "Lowest price a seller is willing to accept.", "Ideal for Buyers": "Lower to enter"},
                    {"Metric": "Implied Volatility (IV)", "Value": f"{rec_option_buy.get('impliedVolatility', None):.2%}" if rec_option_buy.get('impliedVolatility') is not None and not pd.isna(rec_option_buy.get('impliedVolatility')) else "N/A", "Description": "Market's forecast of volatility. High IV = expensive premium.", "Ideal for Buyers": "Lower is better"},
                    {"Metric": "Delta", "Value": f"{rec_option_buy.get('delta', None):.2f}" if rec_option_buy.get('delta') is not None and not pd.isna(rec_option_buy.get('delta')) else "N/A", "Description": "Option's price change per $1 stock change.", "Ideal for Buyers": "0.60 to 0.80 (for ITM calls)"},
                    {"Metric": "Theta", "Value": f"{rec_option_buy.get('theta', None):.3f}" if rec_option_buy.get('theta') is not None and not pd.isna(rec_option_buy.get('theta')) else "N/A", "Description": "Time decay. Daily value lost from the premium.", "Ideal for Buyers": "As low as possible"},
                    {"Metric": "Gamma", "Value": f"{rec_option_buy.get('gamma', None):.3f}" if rec_option_buy.get('gamma') is not None and not pd.isna(rec_option_buy.get('gamma')) else "N/A", "Description": "Rate of change of Delta. High Gamma = faster delta changes.", "Ideal for Buyers": "Higher for directional plays"},
                    {"Metric": "Vega", "Value": f"{rec_option_buy.get('vega', None):.3f}" if rec_option_buy.get('vega') is not None and not pd.isna(rec_option_buy.get('vega')) else "N/A", "Description": "Option's price change per 1% change in interest rates.", "Ideal for Buyers": "Lower if IV expected to fall"},
                    {"Metric": "Rho", "Value": f"{rec_option_buy.get('rho', None):.3f}" if rec_option_buy.get('rho') is not None and not pd.isna(rec_option_buy.get('rho')) else "N/A", "Description": "Option's price change per 1% change in interest rates.", "Ideal for Buyers": "Less significant for short-term"},
                    {"Metric": "Open Interest", "Value": f"{rec_option_buy.get('openInterest', None):,}" if rec_option_buy.get('openInterest') is not None and not pd.isna(rec_option_buy.get('openInterest')) else "N/A", "Description": "Total open contracts. High OI indicates good liquidity.", "Ideal for Buyers": "> 100s"},
                    {"Metric": "Volume (Today)", "Value": f"{rec_option_buy.get('volume', None):,}" if rec_option_buy.get('volume') is not None and not pd.isna(rec_option_buy.get('volume')) else "N/A", "Description": "Number of contracts traded today.", "Ideal for Buyers": "Higher (>100)"},
                ]
                st.table(pd.DataFrame(option_metrics_buy).set_index("Metric"))

                st.write("**Sell Leg:**")
                rec_option_sell = trade_plan['Contracts']['Sell']
                option_metrics_sell = [
                    {"Metric": "Strike", "Value": f"${rec_option_sell.get('strike', 0):.2f}", "Description": "The price at which the option can be exercised.", "Ideal for Sellers": "Higher for calls, lower for puts"},
                    {"Metric": "Expiration", "Value": trade_plan['Expiration'], "Description": "Date the option expires.", "Ideal for Sellers": "Shorter term (to maximize time decay)"},
                    {"Metric": f"Value ({ticker})", "Value": f"${rec_option_sell.get('lastPrice', None):.2f}" if rec_option_sell.get('lastPrice') is not None and not pd.isna(rec_option_sell.get('lastPrice')) else "N/A", "Description": "The last traded price of the option.", "Ideal for Sellers": "Higher to enter"},
                    {"Metric": "Bid", "Value": f"${rec_option_sell.get('bid', None):.2f}" if rec_option_sell.get('bid') is not None and not pd.isna(rec_option_sell.get('bid')) else "N/A", "Description": "Highest price a buyer is willing to pay.", "Ideal for Sellers": "Higher to enter"},
                    {"Metric": "Ask", "Value": f"${rec_option_sell.get('ask', None):.2f}" if rec_option_sell.get('ask') is not None and not pd.isna(rec_option_sell.get('ask')) else "N/A", "Description": "Lowest price a seller is willing to accept.", "Ideal for Sellers": "Higher to enter"},
                    {"Metric": "Implied Volatility (IV)", "Value": f"{rec_option_sell.get('impliedVolatility', None):.2%}" if rec_option_sell.get('impliedVolatility') is not None and not pd.isna(rec_option_sell.get('impliedVolatility')) else "N/A", "Description": "Market's forecast of volatility. High IV = expensive premium.", "Ideal for Sellers": "Higher is better"},
                    {"Metric": "Delta", "Value": f"{rec_option_sell.get('delta', None):.2f}" if rec_option_sell.get('delta') is not None and not pd.isna(rec_option_sell.get('delta')) else "N/A", "Description": "Option's price change per $1 stock change.", "Ideal for Sellers": "Lower (0.20-0.40) for defined risk spreads"},
                    {"Metric": "Theta", "Value": f"{rec_option_sell.get('theta', None):.3f}" if rec_option_sell.get('theta') is not None and not pd.isna(rec_option_sell.get('theta')) else "N/A", "Description": "Time decay. Daily value lost from the premium.", "Ideal for Sellers": "Higher (more decay)"},
                    {"Metric": "Gamma", "Value": f"{rec_option_sell.get('gamma', None):.3f}" if rec_option_sell.get('gamma') is not None and not pd.isna(rec_option_sell.get('gamma')) else "N/A", "Description": "Rate of change of Delta. High Gamma = faster delta changes.", "Ideal for Sellers": "Lower for stability"},
                    {"Metric": "Vega", "Value": f"{rec_option_sell.get('vega', None):.3f}" if rec_option_sell.get('vega') is not None and not pd.isna(rec_option_sell.get('vega')) else "N/A", "Description": "Option's price change per 1% change in interest rates.", "Ideal for Sellers": "Lower if IV expected to fall"},
                    {"Metric": "Rho", "Value": f"{rec_option_sell.get('rho', None):.3f}" if rec_option_sell.get('rho') is not None and not pd.isna(rec_option_sell.get('rho')) else "N/A", "Description": "Option's price change per 1% change in interest rates.", "Ideal for Sellers": "Less significant for short-term"},
                    {"Metric": "Open Interest", "Value": f"{rec_option_sell.get('openInterest', None):,}" if rec_option_sell.get('openInterest') is not None and not pd.isna(rec_option_sell.get('openInterest')) else "N/A", "Description": "Total open contracts. High OI indicates good liquidity.", "Ideal for Sellers": "> 100s"},
                    {"Metric": "Volume (Today)", "Value": f"{rec_option_sell.get('volume', None):,}" if rec_option_sell.get('volume') is not None and not pd.isna(rec_option_sell.get('volume')) else "N/A", "Description": "Number of contracts traded today.", "Ideal for Sellers": "Higher (>100)"},
                ]
                st.table(pd.DataFrame(option_metrics_sell).set_index("Metric"))

            else: # Single Call strategy (Buy ITM Call or Buy ATM Call)
                rec_option = trade_plan['Contract']
                option_metrics = [
                    {"Metric": "Strike", "Value": f"${rec_option.get('strike', 0):.2f}", "Description": "The price at which the option can be exercised.", "Ideal for Buyers": "Lower for calls, higher for puts"},
                    {"Metric": "Expiration", "Value": trade_plan['Expiration'], "Description": "Date the option expires.", "Ideal for Buyers": "Longer term (45-365 days)"},
                    {"Metric": f"Value ({ticker})", "Value": f"${rec_option.get('lastPrice', None):.2f}" if rec_option.get('lastPrice') is not None and not pd.isna(rec_option.get('lastPrice')) else "N/A", "Description": "The last traded price of the option.", "Ideal for Buyers": "Lower to enter"},
                    {"Metric": "Bid", "Value": f"${rec_option.get('bid', None):.2f}" if rec_option.get('bid') is not None and not pd.isna(rec_option.get('bid')) else "N/A", "Description": "Highest price a buyer is willing to pay.", "Ideal for Buyers": "Lower to enter"},
                    {"Metric": "Ask", "Value": f"${rec_option.get('ask', None):.2f}" if rec_option.get('ask') is not None and not pd.isna(rec_option.get('ask')) else "N/A", "Description": "Lowest price a seller is willing to accept.", "Ideal for Buyers": "Lower to enter"},
                    {"Metric": "Implied Volatility (IV)", "Value": f"{rec_option.get('impliedVolatility', None):.2%}" if rec_option.get('impliedVolatility') is not None and not pd.isna(rec_option.get('impliedVolatility')) else "N/A", "Description": "Market's forecast of volatility. High IV = expensive premium.", "Ideal for Buyers": "Lower is better"},
                    {"Metric": "Delta", "Value": f"{rec_option.get('delta', None):.2f}" if rec_option.get('delta') is not None and not pd.isna(rec_option.get('delta')) else "N/A", "Description": "Option's price change per $1 stock change.", "Ideal for Buyers": "0.60 to 0.80 (for ITM calls)"},
                    {"Metric": "Theta", "Value": f"{rec_option.get('theta', None):.3f}" if rec_option.get('theta') is not None and not pd.isna(rec_option.get('theta')) else "N/A", "Description": "Time decay. Daily value lost from the premium.", "Ideal for Buyers": "As low as possible"},
                    {"Metric": "Gamma", "Value": f"{rec_option.get('gamma', None):.3f}" if rec_option.get('gamma') is not None and not pd.isna(rec_option.get('gamma')) else "N/A", "Description": "Rate of change of Delta. High Gamma = faster delta changes.", "Ideal for Buyers": "Higher for directional plays"},
                    {"Metric": "Vega", "Value": f"{rec_option.get('vega', None):.3f}" if rec_option.get('vega') is not None and not pd.isna(rec_option.get('vega')) else "N/A", "Description": "Option's price change per 1% change in interest rates.", "Ideal for Buyers": "Lower if IV expected to fall"},
                    {"Metric": "Rho", "Value": f"{rec_option.get('rho', None):.3f}" if rec_option.get('rho') is not None and not pd.isna(rec_option.get('rho')) else "N/A", "Description": "Option's price change per 1% change in interest rates.", "Ideal for Buyers": "Less significant for short-term"},
                    {"Metric": "Open Interest", "Value": f"{rec_option.get('openInterest', None):,}" if rec_option.get('openInterest') is not None and not pd.isna(rec_option.get('openInterest')) else "N/A", "Description": "Total open contracts. High OI indicates good liquidity.", "Ideal for Buyers": "> 100s"},
                    {"Metric": "Volume (Today)", "Value": f"{rec_option.get('volume', None):,}" if rec_option.get('volume') is not None and not pd.isna(rec_option.get('volume')) else "N/A", "Description": "Number of contracts traded today.", "Ideal for Buyers": "Higher (>100)"},
                ]
                st.table(pd.DataFrame(option_metrics).set_index("Metric"))
        else:
            st.warning(trade_plan['message'])
        # --- End of new detailed options display ---
        
        st.markdown("---")
        st.subheader("⛓️ Full Option Chain")
        option_type = st.radio("Select Option Type", ["Calls", "Puts"], horizontal=True, key=f"option_type_{ticker}") # Added unique key
        exp_date_str = st.selectbox("Select Expiration Date to View", expirations, key=f"exp_date_select_{ticker}") # Added unique key
        if exp_date_str:
            calls, puts = get_options_chain(ticker, exp_date_str)
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
        # Pass is_intraday=False to calculate_indicators for daily data
        daily_df_calculated = calculate_indicators(daily_hist.copy(), is_intraday=False)
        # Removed .dropna() here, relying on backtest_strategy to handle NaNs via first_valid_idx

        # Pass the selection directly to backtest_strategy
        trades, wins, losses = backtest_strategy(daily_df_calculated, selection)
        total_trades = wins + losses
        win_rate = (wins / total_trades) * 100 if total_trades > 0 else 0
        
        col1, col2, col3 = st.columns(3)
        col1.metric("Trades Simulated", total_trades)
        col2.metric("Wins", wins)
        col3.metric("Win Rate", f"{win_rate:.1f}%")
        
        if trades: st.dataframe(pd.DataFrame(trades).tail(20))
        else: st.info("No trades were executed based on the current strategy and historical data. Try adjusting indicators or timeframes, or check if enough historical data is available.")
    else:
        st.warning("Could not fetch daily data for backtesting or data is empty. Ensure the ticker is valid and enough historical data is available for the selected period (e.g., 2 years).")

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
    if finviz_data and finviz_data.get('headlines'):
        for h in finviz_data['headlines']:
            st.markdown(f"_{h}_")
    else:
        st.info("No recent headlines found or automated scoring is disabled.")

def display_trade_log_tab(LOG_FILE, ticker, timeframe, overall_confidence):
    """Displays and manages the trade log."""
    st.subheader("📝 Log Your Trade Analysis")
    # Added a unique key for the text_area based on the ticker
    user_notes = st.text_area("Add your personal notes or trade thesis here:", key=f"trade_notes_{ticker}")
    
    st.info("Trade log functionality is pending implementation.")

