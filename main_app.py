import pdb
import streamlit as st
from datetime import date
import numpy as np
import yfinance as yf
from prophet import Prophet
from prophet.plot import plot_plotly, plot_components_plotly, add_changepoints_to_plot
from plotly import graph_objs as go
from pytickersymbols import PyTickerSymbols as PTS


def plot_daily_trading_volume(data, selected_stock):
    fig = go.Figure(data=[go.Candlestick(x=data['Date'],
                                         open=data['Open'],
                                         high=data['High'],
                                         low=data['Low'],
                                         close=data['Close'])])
    fig.layout.update(title_text='Daily Candlestick Chart of ' + str(selected_stock),
                      xaxis_rangeslider_visible=True)
    st.plotly_chart(fig)


@st.cache_data
def load_data(ticker, START, TODAY):
    data_f = yf.download(ticker, START, TODAY)
    data_f.reset_index(inplace=True)
    return data_f


def plot_raw_data(data, selected_stock):
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=data['Date'], y=data['Open'], name=str(selected_stock) + ' Open Price'))
    fig.add_trace(go.Scatter(x=data['Date'], y=data['Close'], name=str(selected_stock) + 'Closing Price'))
    fig.layout.update(title_text='Historical Price of ' + str(selected_stock) + ' at open and close.',
                      xaxis_rangeslider_visible=True)
    st.plotly_chart(fig)


def main():
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
    data = load_data(selected_stock, START, TODAY)
    st.subheader('Historical Price Action at Open and Close (2015-current) of ' + str(selected_stock))

    plot_daily_trading_volume(data, selected_stock)
    plot_raw_data(data, selected_stock)

    # Feature selection

    # Model selection, training
    # Predict forecast with Prophet.
    df_train = data[['Date', 'Close']]
    df_train = df_train.rename(columns={"Date": "ds", "Close": "y"})
    log_df_train = df_train
    log_df_train['cap'] = 3 * df_train['y']  # avoid large swings in prediction
    log_df_train['floor'] = np.zeros(df_train.shape[0]) + 0.5  # Set saturating minimum
    # Log-Scale our data (to prevent negative forecasting),
    # this performs a box-cox transformation or conventionally sets lambda = 0
    m = Prophet()
    m.fit(df_train)
    future = m.make_future_dataframe(periods=period)

    forecast = m.predict(future)
    # Visualizations of prophet forecasts
    st.write(f'Forecast plot for {n_years} years using prophet')
    fig1 = plot_plotly(m, forecast)
    st.plotly_chart(fig1)
    st.write("Model uncertainty and seasonality")
    fig2 = plot_components_plotly(m, forecast)
    st.write(fig2)
