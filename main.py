import streamlit as st
import yfinance as yf
import pandas as pd
import plotly.express as px
from transformers import pipeline
from datetime import datetime

st.set_page_config(page_title="AI Market Intelligence Platform", layout="wide")

st.title("AI-Powered Market Intelligence & Investment Research Platform")
st.write("Institutional-style financial intelligence dashboard using market data, NLP sentiment analysis, and automated investment memo generation.")

@st.cache_resource
def load_sentiment_model():
    return pipeline("sentiment-analysis")

sentiment_model = load_sentiment_model()

symbol = st.sidebar.text_input("Enter Stock Ticker", "AAPL").upper()
period = st.sidebar.selectbox("Select Period", ["6mo", "1y", "2y", "5y"], index=1)

stock = yf.Ticker(symbol)
hist = stock.history(period=period)

st.header(f"{symbol} Market Dashboard")

if hist.empty:
    st.error("No market data found. Try another ticker.")
else:
    hist["Daily Return"] = hist["Close"].pct_change()

    col1, col2, col3, col4 = st.columns(4)

    latest_price = hist["Close"].iloc[-1]
    total_return = (hist["Close"].iloc[-1] / hist["Close"].iloc[0] - 1) * 100
    volatility = hist["Daily Return"].std() * (252 ** 0.5) * 100
    max_drawdown = ((hist["Close"] / hist["Close"].cummax()) - 1).min() * 100

    col1.metric("Latest Price", f"${latest_price:,.2f}")
    col2.metric("Total Return", f"{total_return:.2f}%")
    col3.metric("Annualized Volatility", f"{volatility:.2f}%")
    col4.metric("Max Drawdown", f"{max_drawdown:.2f}%")

    fig = px.line(hist, x=hist.index, y="Close", title=f"{symbol} Stock Price")
    st.plotly_chart(fig, use_container_width=True)

    st.subheader("Company Information")
    info = stock.info

    st.write({
        "Company": info.get("longName", "N/A"),
        "Sector": info.get("sector", "N/A"),
        "Industry": info.get("industry", "N/A"),
        "Market Cap": info.get("marketCap", "N/A"),
        "Trailing PE": info.get("trailingPE", "N/A"),
        "Beta": info.get("beta", "N/A")
    })

st.header("Financial Sentiment Engine")

sample_text = st.text_area(
    "Paste financial headline, earnings commentary, or research text:",
    "The company exceeded earnings expectations, raised guidance, and reported strong revenue growth."
)

if st.button("Analyze Sentiment"):
    result = sentiment_model(sample_text[:512])[0]
    st.write("Sentiment Result:", result)

st.header("AI Investment Memo Generator")

def generate_memo(symbol, total_return, volatility, max_drawdown, sentiment_text):
    sentiment = sentiment_model(sentiment_text[:512])[0]

    memo = f"""
# Investment Research Memo: {symbol}

Date: {datetime.today().strftime('%Y-%m-%d')}

## Executive Summary
{symbol} was analyzed using market performance, volatility, downside risk, and NLP-based sentiment analysis.

## Market Performance
- Total Return: {total_return:.2f}%
- Annualized Volatility: {volatility:.2f}%
- Maximum Drawdown: {max_drawdown:.2f}%

## Sentiment Analysis
- Sentiment Label: {sentiment['label']}
- Sentiment Confidence: {sentiment['score']:.2f}

## Risk Assessment
The asset shows a volatility profile of {volatility:.2f}% and a maximum drawdown of {max_drawdown:.2f}%. 
Higher volatility and deeper drawdowns may indicate elevated downside risk.

## Investment View
This memo provides a structured research summary combining quantitative market data and AI-powered sentiment analysis. 
Further research should include SEC filings, earnings transcripts, competitive analysis, and macroeconomic conditions.

## Recommendation
Use this output as a first-pass research screen, not a final investment recommendation.
"""
    return memo

if not hist.empty:
    memo = generate_memo(symbol, total_return, volatility, max_drawdown, sample_text)
    st.markdown(memo)

    st.download_button(
        label="Download Investment Memo",
        data=memo,
        file_name=f"{symbol}_investment_memo.md",
        mime="text/markdown"
    )