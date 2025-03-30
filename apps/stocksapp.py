
import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.tsa.stattools import adfuller
import itertools
import warnings

warnings.filterwarnings("ignore")

def load_data(stock_code, period="10y"):
    stock = yf.Ticker(stock_code)
    df = stock.history(period=period)
    df = df[['Close']].rename(columns={'Close': 'close'})
    df.index.name = 'date'
    return df

def preprocess_data(df):
    df_week = df.resample('W').mean()
    df_week['weekly_ret'] = np.log(df_week['close']).diff()
    df_week.dropna(inplace=True)
    return df_week

def test_stationarity(series):
    result = adfuller(series)
    return result[1] < 0.05

def make_stationary(series):
    d = 0
    while not test_stationarity(series) and d < 3:
        series = series.diff().dropna()
        d += 1
    return series, d

def find_best_arima(series, d, max_p=3, max_q=3):
    best_aic = float("inf")
    best_order = None
    for p, q in itertools.product(range(max_p+1), range(max_q+1)):
        try:
            model = ARIMA(series, order=(p, d, q)).fit()
            if model.aic < best_aic:
                best_aic = model.aic
                best_order = (p, d, q)
        except:
            continue
    return best_order

def forecast_prices(df_week, years, stock_code):
    series = df_week['weekly_ret']
    stationary_series, d = make_stationary(series)
    best_order = find_best_arima(series, d)
    model = ARIMA(series, order=best_order).fit()
    steps = years * 52
    forecast = model.forecast(steps=steps)
    last_price = df_week['close'].iloc[-1]
    forecast_price = last_price * np.exp(np.cumsum(forecast))
    forecast_index = pd.date_range(df_week.index[-1] + pd.Timedelta(weeks=1), periods=steps, freq='W')
    forecast_df = pd.DataFrame({'forecast': forecast_price}, index=forecast_index)
    return forecast_df

def main():
    st.title("📈 Multi-Stock ARIMA Forecasting App")
    
    stock_input = st.text_input("Enter stock codes (comma-separated, e.g., AAPL,MSFT,GOOGL):", "AAPL,MSFT")
    period = st.selectbox("Select historical data period:", ["5y", "10y", "max"], index=1)
    years = st.slider("Select number of years to forecast:", 1, 5, 2)

    if st.button("Run Forecast"):
        stock_codes = [s.strip().upper() for s in stock_input.split(",")]
        forecast_results = {}
        
        plt.figure(figsize=(14, 7))
        for stock_code in stock_codes:
            try:
                df = load_data(stock_code, period)
                df_week = preprocess_data(df)
                forecast_df = forecast_prices(df_week, years, stock_code)

                # Plot historical and forecasted prices
                plt.plot(df_week['close'], label=f'{stock_code} (History)')
                plt.plot(forecast_df['forecast'], linestyle='--', label=f'{stock_code} (Forecast)')

                forecast_results[stock_code] = forecast_df

            except Exception as e:
                st.error(f"Error processing {stock_code}: {e}")
        
        plt.title("Stock Price Forecast (Historical + Future)")
        plt.xlabel("Date")
        plt.ylabel("Price")
        plt.legend()
        plt.grid(True)
        st.pyplot(plt)

if __name__ == "__main__":
    main()
