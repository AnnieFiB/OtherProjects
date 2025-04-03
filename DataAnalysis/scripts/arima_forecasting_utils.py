
import yfinance as yf
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import warnings
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.tsa.stattools import adfuller
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
from statsmodels.tsa.stattools import acf, pacf
from sklearn.metrics import mean_squared_error
from statsmodels.stats.diagnostic import acorr_ljungbox
from scipy.stats import kurtosis
from textwrap import dedent
from scipy.signal import find_peaks
import itertools
import requests


warnings.filterwarnings("ignore")

def resolve_ticker(query):
    try:
        url = f"https://query2.finance.yahoo.com/v1/finance/search?q={query}"
        headers = {"User-Agent": "Mozilla/5.0"}
        response = requests.get(url, headers=headers, timeout=10)
        response.raise_for_status()
        data = response.json()
        if 'quotes' in data and len(data['quotes']) > 0 and 'symbol' in data['quotes'][0]:
            return data['quotes'][0]['symbol']
        else:
            print(f"No ticker found for '{query}', using input as-is.")
            return query
    except Exception as e:
        print(f"Error resolving '{query}': {e}")
        return query

def get_stock_input():
    query = input("Enter stock ticker(s) or company name(s) (comma-separated): ").strip()
    stock_queries = [q.strip() for q in query.split(",")]
    resolved = [resolve_ticker(q) for q in stock_queries]
    
    while True:
        try:
            years = int(input("Enter number of years to forecast: "))
            break
        except ValueError:
            print("Please enter a valid number.")

    print(f"Resolved stock codes: {resolved}")
    return resolved, years

# Single-stock loader
def load_data(stock_code, period="10y"):
    stock = yf.Ticker(stock_code)
    df = stock.history(period=period)
    df.reset_index(inplace=True)
    df.columns = [col.lower() for col in df.columns]
    df.set_index("date", inplace=True)
    return df

# Multi-stock loader
def load_all_data(stock_codes, period="10y"):
    raw_data = {}
    display_data = {}

    for code in stock_codes:
        df = load_data(code, period)
        raw_data[code] = df
        display_df = df.copy().reset_index()
        display_df['ticker'] = code
        display_data[code] = display_df

    return display_data, raw_data


# Single-stock preprocessing
def preprocess_data(df):
    df_week = df.resample('W').mean()
    df_week['weekly_ret'] = np.log(df_week['close']).diff()
    df_week.dropna(inplace=True)
    return df_week

# Multi-stock preprocessing
def preprocess_all_data(data_dict):
    preprocessed_raw = {}
    display_data = {}

    for stock_code, df in data_dict.items():
        df_week = df[['close']].resample('W').mean()
        df_week['weekly_ret'] = np.log(df_week['close']).diff()
        df_week.dropna(inplace=True)

        preprocessed_raw[stock_code] = df_week

        display_df = df_week.reset_index()
        display_df['ticker'] = stock_code
        display_data[stock_code] = display_df

    return display_data, preprocessed_raw

def plot_weekly_returns(data_dict):
    plt.figure(figsize=(14, 6))
    for stock_code, df in data_dict.items():
        plt.plot(df.index, df['weekly_ret'], label=stock_code)
    plt.title("Weekly Log Returns")
    plt.xlabel("Date")
    plt.ylabel("Log Return")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()

def test_stationarity(series, window=12, plot=False, title=None):
    # Handle single series
    if isinstance(series, pd.Series):
        roll_mean = series.rolling(window=window).mean()
        roll_std = series.rolling(window=window).std()

        if plot:
            plt.figure(figsize=(10, 5))
            plt.plot(series, label='Original')
            plt.plot(roll_mean, label='Rolling Mean', linestyle='--')
            plt.plot(roll_std, label='Rolling Std Dev', linestyle=':')
            plt.title(f'Stationarity Check: {title or "Time Series"}')
            plt.legend()
            plt.grid(True)
            plt.show()

        result = adfuller(series.dropna())
        print(f"{title or 'Series'} - ADF Statistic: {result[0]:.2f}, p-value: {result[1]}")
        return result[1] < 0.05

    # Handle dictionary of series
    elif isinstance(series, dict):
        results = {}
        for code, s in series.items():
            result = test_stationarity(s, window=window, plot=plot, title=code)
            results[code] = result
        return results

    else:
        raise TypeError("Input must be a pandas Series or a dictionary of Series.")

def summarize_adf_tests(series_dict, window=12, plot=False):
    summary = []

    for code, series in series_dict.items():
        roll_mean = series.rolling(window=window).mean()
        roll_std = series.rolling(window=window).std()

        if plot:
            plt.figure(figsize=(10, 4))
            plt.plot(series, label='Original')
            plt.plot(roll_mean, label='Rolling Mean', linestyle='--')
            plt.plot(roll_std, label='Rolling Std Dev', linestyle=':')
            plt.title(f'Stationarity Check: {code}')
            plt.legend()
            plt.grid(True)
            plt.show()

        result = adfuller(series.dropna())
        summary.append({
            "Ticker": code,
            "ADF Statistic": round(result[0], 4),
            "p-value": result[1],
            "Conclusion": "Stationary" if result[1] < 0.05 else "Not Stationary"
        })

    return pd.DataFrame(summary)

def make_all_stationary_auto(series_dict):
    stationary_series = {}
    diff_orders = {}

    for code, series in series_dict.items():
        is_stationary = test_stationarity(series, plot=False, title=code)
        d = 0
        while not is_stationary and d < 3:
            series = series.diff().dropna()
            d += 1
            is_stationary = test_stationarity(series, plot=False, title=f"{code} (d={d})")

        print(f"{code}: Final differencing applied d = {d}")
        stationary_series[code] = series
        diff_orders[code] = d

    return stationary_series, diff_orders


def plot_all_acf_pacf(series_dict, max_lags=20):

    for ticker, series in series_dict.items():
        cleaned = series.dropna()
        max_allowed = int(len(cleaned) * 0.5) - 1
        lags = min(max_lags, max_allowed)

        fig, axes = plt.subplots(1, 2, figsize=(12, 4))
        fig.suptitle(f"{ticker} - ACF and PACF (lags={lags})")
        plot_acf(cleaned, ax=axes[0], lags=lags)
        plot_pacf(cleaned, ax=axes[1], lags=lags)
        axes[0].set_title("ACF")
        axes[1].set_title("PACF")
        plt.tight_layout()
        plt.show()
def suggest_arima_params(series_dict, diff_orders=None, alpha=0.1):
    from statsmodels.tsa.stattools import acf, pacf

    def get_strongest_significant_lag(values, conf_int):
        significant = []
        for i in range(1, len(values)):
            lower, upper = conf_int[i]
            if values[i] < lower or values[i] > upper:
                strength = abs(values[i])
                significant.append((i, strength))
        return max(significant, key=lambda x: x[1])[0] if significant else None

    suggestions = {}

    for ticker, series in series_dict.items():
        cleaned = series.dropna()
        max_lags = min(20, len(cleaned) // 2 - 1)

        if max_lags < 1 or len(cleaned) < 10:
            print(f"⚠️ Not enough data for {ticker}. Using fallback ARIMA(1, {diff_orders.get(ticker, 0)}, 1)")
            suggestions[ticker] = {"p": 1, "d": diff_orders.get(ticker, 0), "q": 1}
            continue

        try:
            pacf_vals, pacf_conf = pacf(cleaned, nlags=max_lags, alpha=alpha)
            acf_vals, acf_conf = acf(cleaned, nlags=max_lags, alpha=alpha)

            p = get_strongest_significant_lag(pacf_vals, pacf_conf)
            q = get_strongest_significant_lag(acf_vals, acf_conf)
            d = diff_orders.get(ticker, 0) if diff_orders else 0

            # Fallback if no spikes
            if p is None and q is None:
                print(f"⚠️ No significant spikes for {ticker}. Using fallback ARIMA(1, {d}, 1)")
                p, q = 1, 1

            suggestions[ticker] = {"p": p or 0, "d": d, "q": q or 0}
            print(f"{ticker} → Suggested ARIMA(p,d,q): ({p or 0}, {d}, {q or 0})")

        except Exception as e:
            print(f"⚠️ Error analyzing {ticker}: {e}. Using fallback ARIMA(1, {diff_orders.get(ticker, 0)}, 1)")
            suggestions[ticker] = {"p": 1, "d": diff_orders.get(ticker, 0), "q": 1}

    return suggestions



def find_best_arima(series, p=1, q=1, d=0, max_offset=2, ticker=None):
    best_aic = float("inf")
    best_order = None
    best_model = None

    p_range = range(max(0, p - max_offset), p + max_offset + 1)
    q_range = range(max(0, q - max_offset), q + max_offset + 1)

    for pi, qi in itertools.product(p_range, q_range):
        if pi == 0 and qi == 0:
            continue
        try:
            model = ARIMA(series, order=(pi, d, qi)).fit()
            if model.aic < best_aic:
                best_aic = model.aic
                best_order = (pi, d, qi)
                best_model = model
        except:
            continue

    if best_model:
        forecast = best_model.fittedvalues
        actual = series[d:]
        mse = mean_squared_error(actual, forecast[d:])
    else:
        mse = None

    label = f"{ticker} " if ticker else ""
    print(f"{label}→ Best ARIMA near ({p}, {d}, {q}): {best_order} | AIC: {best_aic:.2f} | MSE: {mse:.6f}")
    return best_order


def evaluate_arima_residuals(series_dict, best_orders_dict, plot=True):
    results = []

    for ticker, series in series_dict.items():
        order = best_orders_dict[ticker]
        model = ARIMA(series, order=order).fit()
        residuals = model.resid

        # Check residual properties
        mean_resid = residuals.mean()
        std_resid = residuals.std()
        kurt = kurtosis(residuals)

        # Ljung-Box test
        lb_test = acorr_ljungbox(residuals, lags=[10], return_df=True)
        p_value = lb_test["lb_pvalue"].iloc[0]
        passed = p_value > 0.05

        print(f"{ticker} | ARIMA{order} | Residual Mean: {mean_resid:.4f}, Std: {std_resid:.4f}, Kurtosis: {kurt:.2f}, Ljung-Box p-value: {p_value:.4f} → {'Good fit' if passed else 'Poor fit'}")

        if plot:
            fig, axes = plt.subplots(1, 2, figsize=(14, 4))
            axes[0].plot(residuals)
            axes[0].set_title(f"{ticker} Residuals")
            axes[0].axhline(0, linestyle='--', color='gray')

            sns.histplot(residuals, kde=True, ax=axes[1])
            axes[1].set_title(f"{ticker} Residual Density")
            plt.tight_layout()
            plt.show()

        results.append({
            "Ticker": ticker,
            "ARIMA Order": order,
            "Residual Mean": mean_resid,
            "Residual Std": std_resid,
            "Kurtosis": kurt,
            "Ljung-Box p-value": p_value,
            "Good Fit": passed
        })

    return pd.DataFrame(results)


def forecast_pricessep(series_dict, close_dict, best_orders_dict, years):
    save_csv = input("Do you want to save the forecast to CSV file? (y/n): ").strip().lower() == "y"

    forecast_steps = years * 52
    all_forecasts = {}

    for ticker, series in series_dict.items():
        order = best_orders_dict[ticker]
        model = ARIMA(series, order=order).fit()

        forecast_returns = model.forecast(steps=forecast_steps)
        last_price = close_dict[ticker]['close'].iloc[-1]
        forecast_prices = last_price * np.exp(np.cumsum(forecast_returns))
        forecast_index = pd.date_range(start=series.index[-1] + pd.Timedelta(weeks=1), periods=forecast_steps, freq='W')

        forecast_df = pd.DataFrame({
            "date": forecast_index,
            f"{ticker}_price": forecast_prices,
        })
        forecast_df.set_index("date", inplace=True)

        forecast_df[f"{ticker}_% growth"] = ((forecast_df[f"{ticker}_price"] - last_price) / last_price) * 100

        # Format values
        forecast_df[f"{ticker}_price"] = forecast_df[f"{ticker}_price"].apply(lambda x: f"${x:,.2f}")
        forecast_df[f"{ticker}_% growth"] = forecast_df[f"{ticker}_% growth"].apply(lambda x: f"{x:.2f}%")

        # Plot (keep numerical for plotting)
        plt.figure(figsize=(12, 5))
        plt.plot(close_dict[ticker]['close'], label='Historical Price')
        plt.plot(pd.Series(forecast_prices, index=forecast_index), label='Forecasted Price', linestyle='--')
        plt.title(f"{ticker} Forecasted Prices and % Growth (ARIMA{order})")
        plt.xlabel("Date")
        plt.ylabel("Price")
        plt.legend()
        plt.grid(True)
        plt.tight_layout()
        plt.show()

        all_forecasts[ticker] = forecast_df

    combined = pd.concat(all_forecasts.values(), axis=1)

    if save_csv:
        filename = f"forecast_combined_{years}y.csv"
        combined.to_csv(filename)
        print(f"Combined forecast table saved to {filename}")

    return combined

def forecast_prices(series_dict, close_dict, best_orders_dict, years):
    save_csv = input("Do you want to save the forecast to CSV file? (y/n): ").strip().lower() == "y"

    forecast_steps = years * 52
    all_forecasts = {}
    all_forecast_lines = {}
    all_growth_annotations = {}
    all_hist_lines = {}
    ticker_titles = []

    for ticker, series in series_dict.items():
        order = best_orders_dict[ticker]
        model = ARIMA(series, order=order).fit()

        forecast_returns = model.forecast(steps=forecast_steps)
        last_price = close_dict[ticker]['close'].iloc[-1]

        forecast_index = pd.date_range(start=series.index[-1] + pd.Timedelta(weeks=1), periods=forecast_steps, freq='W')
        forecast_prices = last_price * np.exp(np.cumsum(forecast_returns))
        pct_growth_series = ((forecast_prices - last_price) / last_price) * 100
        rolling_avg = forecast_prices.rolling(window=4).mean()

        forecast_df = pd.DataFrame({
            f"{ticker}_price": forecast_prices,
            f"{ticker}_pct_growth": pct_growth_series,
            f"{ticker}_rolling_avg": rolling_avg
        }, index=forecast_index)

        all_forecasts[ticker] = forecast_df
        all_forecast_lines[ticker] = (forecast_index, forecast_prices)
        all_growth_annotations[ticker] = pct_growth_series.iloc[-1]
        ticker_titles.append(f"{ticker} (ARIMA{order})")

        # Include historical from 2024
        hist_data = close_dict[ticker]['close']
        all_hist_lines[ticker] = hist_data[hist_data.index >= "2024-01-01"]

    # Combined plot
    plt.figure(figsize=(14, 6))

    for ticker, (dates, prices) in all_forecast_lines.items():
        plt.plot(dates, prices, label=f"{ticker} Forecast", linestyle='--')

    for ticker, hist_series in all_hist_lines.items():
        plt.plot(hist_series.index, hist_series.values, label=f"{ticker} Historical", linewidth=1.5)

    forecast_start = next(iter(all_forecast_lines.values()))[0][0]
    plt.axvline(x=forecast_start, color='red', linestyle=':', lw=2, label='Forecast Start')

    # Add annotations
    for ticker, (dates, prices) in all_forecast_lines.items():
        final_growth = all_growth_annotations[ticker]
        plt.annotate(f"{ticker}: {final_growth:.2f}%",
                     xy=(dates[-1], prices.iloc[-1]),
                     xytext=(dates[-1], prices.iloc[-1] * 1.05),
                     arrowprops=dict(facecolor='green' if final_growth > 0 else 'red', arrowstyle="->"),
                     fontsize=10, color='green' if final_growth > 0 else 'red')

    # Title with ARIMA per ticker
    plt.title("Forecasted Prices and Growth: " + " | ".join(ticker_titles))
    plt.xlabel("Date")
    plt.ylabel("Price")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()

    combined = pd.concat(all_forecasts.values(), axis=1).fillna(0)

    formatted_combined = combined.copy()
    for col in formatted_combined.columns:
        if "price" in col or "rolling_avg" in col:
            formatted_combined[col] = formatted_combined[col].apply(lambda x: f"${x:,.2f}")
        elif "pct_growth" in col:
            formatted_combined[col] = formatted_combined[col].apply(lambda x: f"{x:.2f}%")

    if save_csv:
        filename = f"forecast_combined_{years}y.csv"
        formatted_combined.to_csv(filename)
        print(f"📁 Combined forecast table saved to {filename}")

    return formatted_combined


def plot_rolling_average(forecast_table):
    
    # Extract all ticker prefixes based on column names
    tickers = set(col.split("_")[0] for col in forecast_table.columns if col.endswith("rolling_avg"))

    for ticker in tickers:
        col = f"{ticker}_rolling_avg"
        if col in forecast_table.columns:
            # Clean formatting if already formatted as strings
            series = forecast_table[col].replace('[\$,]', '', regex=True).replace('%', '', regex=True).replace('', '0').astype(float)

            plt.figure(figsize=(12, 5))
            plt.plot(forecast_table.index, series, label=f'{ticker} Rolling Avg (4-week)', color='darkgreen', linewidth=2)
            plt.title(f"{ticker} 4-Week Rolling Average of Forecasted Prices")
            plt.xlabel("Date")
            plt.ylabel("Rolling Avg Price ($)")
            plt.grid(True)
            plt.legend()
            plt.tight_layout()
            plt.show()
        else:
            print(f"⚠️ Rolling average column not found for {ticker}")

def plot_combined_rolling_average(forecast_table):

    tickers = set(col.split("_")[0] for col in forecast_table.columns if col.endswith("rolling_avg"))

    plt.figure(figsize=(14, 6))
    for ticker in tickers:
        col = f"{ticker}_rolling_avg"
        if col in forecast_table.columns:
            series = forecast_table[col].replace('[\$,]', '', regex=True).replace('%', '', regex=True).replace('', '0').astype(float)
            plt.plot(forecast_table.index, series, label=f"{ticker} Rolling Avg", linewidth=2)
        else:
            print(f"⚠️ Rolling average column not found for {ticker}")

    plt.title("4-Week Rolling Average of Forecasted Prices (All Tickers)")
    plt.xlabel("Date")
    plt.ylabel("Rolling Avg Price ($)")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()