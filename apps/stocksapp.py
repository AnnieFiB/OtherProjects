
import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import arima_forecasting_utils as afu

st.set_page_config(page_title="ARIMA Forecast App", layout="wide")

st.title("📈 ARIMA Stock Forecasting App")
st.markdown("Compare and forecast up to 4 stock tickers using ARIMA models.")

query = st.text_input("Enter stock tickers or company names (comma-separated)" )
tickers = [afu.resolve_ticker(q.strip()) for q in query.split(",") if q.strip()]
tickers = tickers[:4]

years = st.number_input("Forecast years", min_value=1, value=3)
fast_mode = st.checkbox("🚀 Enable Fast Mode (Limit forecast to 1 year)", value=True)
forecast_years = 1 if fast_mode else years

run = st.button("Run Forecast")

if run and tickers:
    st.subheader("🔍 1. Loading Raw Stock Data from Yahoo Finance")
    display_data, raw_data = afu.load_all_data(tickers)
    for t in tickers:
        df_sample = display_data[t].copy()
        df_sample["ticker"] = t
        st.write(f"**Raw data for `{t}`**")
        st.dataframe(df_sample.head())

    st.subheader("⚙️ 2. Visualising Logged Weekly Returns")
    _, weekly_raw = afu.preprocess_all_data(raw_data)
    for t in tickers:
        st.write(f"**Weekly Returns for `{t}`**")
        weekly_ret_series = weekly_raw[t]["weekly_ret"]
        fig, ax = plt.subplots()
        ax.plot(weekly_ret_series.index, weekly_ret_series, label=f"{t} Weekly Returns")
        ax.set_title(f"{t} - Weekly Returns")
        ax.set_ylabel("Return")
        ax.grid(True)
        st.pyplot(fig)

    st.subheader("📉 3. Making Data Stationary if Needed")
    stationary_series, diff_orders = afu.make_all_stationary_auto({k: v["weekly_ret"] for k, v in weekly_raw.items()})
    for t in tickers:
        d = diff_orders.get(t, 0)
        st.write(f"`{t}` → Final differencing applied: d = {d}")

    st.subheader("📊 4. Plotting ACF/PACF Plots for Initial Observation")
    for t in tickers:
        st.write(f"**ACF/PACF for `{t}`**")
        afu.plot_all_acf_pacf({t: stationary_series[t]})

    st.subheader("🤖 5. Suggesting ARIMA Parameters")
    suggestions = afu.suggest_arima_params(stationary_series, diff_orders)
    st.json(suggestions)

    st.subheader("🧠 6. Selecting Best ARIMA Model for Forecasting")
    best_orders = {}
    for t in tickers:
        best_order = afu.find_best_arima(stationary_series[t], **suggestions[t], ticker=t)
        best_orders[t] = best_order
        st.write(f"{t} → Best ARIMA Order: {best_order}")

    st.subheader("✅ 7. Running Residual Diagnostics for Model Ealuation")
    res_df = afu.evaluate_arima_residuals({k: v["weekly_ret"] for k, v in weekly_raw.items()}, best_orders, plot=True)
    st.dataframe(res_df)

    st.subheader("📈 8. Forecasting & Visualising Forecasted Stock Prices")
    forecast_df = afu.forecast_prices(
        {k: v["weekly_ret"] for k, v in weekly_raw.items()},
        weekly_raw,
        best_orders,
        forecast_years
    )
    st.dataframe(forecast_df)

    st.subheader("📉 9. Visualising Respective Ticker's Rolling Averages")
    afu.plot_combined_rolling_average(forecast_df)


else:
    st.info("Enter up to 4 tickers and click 'Run Forecast' to begin.")
