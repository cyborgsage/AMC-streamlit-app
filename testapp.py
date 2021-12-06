import yfinance as yf
import streamlit as st
import pandas as pd

st.write("""
# AMC Sentiment Analysis App

AMC daily closing prices in line chart

""")

tickerData = yf.Ticker('AMC')
tickerDf = tickerData.history(period='1d', start='2021-01-01', end='2021-11-20')

st.line_chart(tickerDf.Close)