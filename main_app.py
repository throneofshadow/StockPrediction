import pdb

import streamlit as st
from datetime import date
import numpy as np
import yfinance as yf
from prophet import Prophet
from prophet.plot import plot_plotly
from plotly import graph_objs as go
from pytickersymbols import PyTickerSymbols as PTS

START = "2015-01-01"  # Limit of historical data from yfinance
TODAY = date.today().strftime("%Y-%m-%d")

st.title('Long-Term Stock Price Forecasting For Major Indices')
stock_data = PTS()
indices = stock_data.get_all_indices()
selected_indices = st.selectbox(' Select indices the stock is traded in. ', indices)
stocks = stock_data.get_stocks_by_index(selected_indices)
selected_stock = st.selectbox('Select a company ', stocks)['symbol']

n_years = st.slider('Desired length of forecast in years:', 1, 4)  # Users choose what time period to predict over
period = n_years * 365


@st.cache_data
def load_data(ticker):
    data_f = yf.download(ticker, START, TODAY)
    data_f.reset_index(inplace=True)
    return data_f


data_load_state = st.text('Consulting database..')
data = load_data(selected_stock)
st.subheader('Historical Price Action at Open and Close (2015-current) of ' + str(selected_stock))


def plot_daily_trading_volume():
    fig = go.Figure(data=[go.Candlestick(x=data.index,
                                         open=data['Open'],
                                         high=data['High'],
                                         low=data['Low'],
                                         close=data['Close'])])
    fig.layout.update(title_text='Daily Candlestick Chart of ' + str(selected_stock),
                      xaxis_rangeslider_visible=True)
    st.plotly_chart(fig)


def plot_raw_data():
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=data['Date'], y=data['Open'], name=str(selected_stock) + ' Open Price'))
    fig.add_trace(go.Scatter(x=data['Date'], y=data['Close'], name=str(selected_stock) + 'Closing Price'))
    fig.layout.update(title_text='Historical Price of ' + str(selected_stock) + ' at open and close.',
                      xaxis_rangeslider_visible=True)
    st.plotly_chart(fig)


plot_daily_trading_volume()
plot_raw_data()

# Feature selection


# Model selection, training
# Predict forecast with Prophet.
df_train = data[['Date', 'Close']]
df_train = df_train.rename(columns={"Date": "ds", "Close": "y"})
# Log-Scale our data (to prevent negative forecasting),
# this performs a box-cox transformation or conventionally sets lambda = 0
m = Prophet(growth='logistic')
m.fit(df_train)
future = m.make_future_dataframe(periods=period)
future['cap'] = df_train['y'][-1] + period * (0.05 * df_train['y'][-1])
# Set highest value achievable to 5% daily climb
future['floor'] = 0.5  # Set saturating minimum
forecast = m.predict(future)
# Show and plot forecast
st.subheader('Forecast data')
st.write(forecast.tail())

st.write(f'Forecast plot for {n_years} years')
fig1 = plot_plotly(m, forecast)
st.plotly_chart(fig1)

st.write("Forecast components")
fig2 = m.plot_components(forecast)
st.write(fig2)
