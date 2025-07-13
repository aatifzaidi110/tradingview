##//app.py
import streamlit as st
import random
import urllib.parse

# === App Header ===
st.set_page_config(page_title="Swing Trade Analyzer", layout="centered")
st.title("📊 Aatif's Swing Trade Dashboard")

# === Ticker Input ===
ticker = st.text_input("Enter a Ticker Symbol", value="NVDA")

if ticker:
    st.subheader(f"📈 Analysis for {ticker.upper()}")

    # === Simulated Confidence Score ===
    score = random.randint(40, 100)

    # === Strategy Signal ===
    if score >= 85:
        signal = "✅ Buy Zone"
        strategy = "Swing Entry Recommended"
    elif score >= 65:
        signal = "⏳ Watchlist"
        strategy = "Partial Entry or Wait"
    else:
        signal = "🚫 Avoid"
        strategy = "No Entry Recommended"

    st.write(f"🧠 Confidence Score: **{score}**")
    st.write(f"{signal} — *{strategy}*")

    # === Expert Analysis Links ===
    encoded_ticker = urllib.parse.quote(ticker)
    google_news = f"https://news.google.com/search?q={encoded_ticker}+stock"
    finviz = f"https://finviz.com/quote.ashx?t={ticker}"
    barchart = f"https://www.barchart.com/stocks/quotes/{ticker}/overview"
    tipranks = f"https://www.tipranks.com/stocks/{ticker}/forecast"
    yahoo = f"https://finance.yahoo.com/quote/{ticker}"

    st.subheader("💬 Expert & Sentiment Links")
    st.markdown(f"- [Google News Sentiment]({google_news})")
    st.markdown(f"- [Finviz Overview]({finviz})")
    st.markdown(f"- [Barchart Technical Opinion]({barchart})")
    st.markdown(f"- [TipRanks Analyst Forecast]({tipranks})")
    st.markdown(f"- [Yahoo Finance Summary]({yahoo})")
