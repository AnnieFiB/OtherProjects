import os
from pathlib import Path
from dotenv import load_dotenv
import requests
from datetime import datetime, timedelta
from collections import Counter
import matplotlib.pyplot as plt

def load_env():
    # Get the root directory (parent of current file's parent)
    root_dir = Path(__file__).resolve().parent.parent
    env_path = root_dir / ".env"
    
    if not env_path.exists():
        raise FileNotFoundError(f".env file not found at {env_path}")
    
    load_dotenv(env_path)
    api_key = os.getenv("OPENWEATHER_API_KEY")
    
    if not api_key:
        raise ValueError("OPENWEATHER_API_KEY not found in .env file")
    
    return api_key

API_KEY = load_env()


# API URLs
GEO_URL = "http://api.openweathermap.org/geo/1.0/direct"
WEATHER_URL = "https://api.openweathermap.org/data/2.5/weather"
FORECAST_URL = "https://api.openweathermap.org/data/2.5/forecast"
AIR_POLLUTION_URL = "http://api.openweathermap.org/data/2.5/air_pollution"
AIR_POLLUTION_FORECAST_URL = "http://api.openweathermap.org/data/2.5/air_pollution/forecast"

# Data processing functions
def get_aqi_text(aqi):
    return {
        1: "Good", 2: "Fair", 3: "Moderate",
        4: "Poor", 5: "Very Poor"
    }.get(aqi, "Unknown")

def get_city_coordinates(city_name):
    params = {"q": city_name, "limit": 1, "appid": API_KEY}
    response = requests.get(GEO_URL, params=params)
    if response.status_code != 200 or not response.json():
        raise ValueError(f"City '{city_name}' not found")
    return response.json()[0]["lat"], response.json()[0]["lon"]

def fetch_weather_data(lat, lon):
    params = {"lat": lat, "lon": lon, "appid": API_KEY, "units": "metric"}
    return _handle_api_request(WEATHER_URL, params, "weather data")

def fetch_forecast_data(lat, lon):
    params = {"lat": lat, "lon": lon, "appid": API_KEY, "units": "metric"}
    return _handle_api_request(FORECAST_URL, params, "forecast data")

def fetch_air_pollution_data(lat, lon):
    params = {"lat": lat, "lon": lon, "appid": API_KEY}
    return _handle_api_request(AIR_POLLUTION_URL, params, "air pollution data")

def fetch_air_pollution_forecast(lat, lon):
    params = {"lat": lat, "lon": lon, "appid": API_KEY}
    return _handle_api_request(AIR_POLLUTION_FORECAST_URL, params, "air pollution forecast")

def _handle_api_request(url, params, error_context):
    response = requests.get(url, params=params)
    if response.status_code != 200:
        raise Exception(f"Error fetching {error_context}: {response.status_code}")
    return response.json()

# Data formatting functions
def prepare_weather_table(data, air_pollution_data=None):
    main_data = data.get("main", {})
    weather_data = data.get("weather", [{}])[0]
    sys_data = data.get("sys", {})
    
    table_data = [
        ["City", f"{data.get('name')}, {sys_data.get('country')}"],
        ["Temperature", f"{main_data.get('temp', 'N/A')}°C"],
        ["Feels Like", f"{main_data.get('feels_like', 'N/A')}°C"],
        ["Weather", weather_data.get('description', 'N/A')],
        ["Pressure", f"{main_data.get('pressure', 'N/A')} hPa"],
        ["Humidity", f"{main_data.get('humidity', 'N/A')}%"],
        ["Wind Speed", f"{data.get('wind', {}).get('speed', 'N/A')} m/s"],
        ["Rain (1h)", f"{data.get('rain', {}).get('1h', 0)} mm"]
    ]
    
    if air_pollution_data:
        aqi = air_pollution_data.get("list", [{}])[0].get("main", {}).get("aqi")
        table_data.append(["Air Quality", get_aqi_text(aqi)])
    
    return table_data

def prepare_forecast_chart_data(forecast_data):
    chart_data = {"dates": [], "temps": [], "humidity": [], "rain": []}
    for item in forecast_data.get("list", []):
        chart_data["dates"].append(datetime.fromtimestamp(item.get("dt")))
        chart_data["temps"].append(item.get("main", {}).get("temp"))
        chart_data["humidity"].append(item.get("main", {}).get("humidity"))
        chart_data["rain"].append(item.get("rain", {}).get("3h", 0))
    return chart_data

def create_forecast_figure(chart_data, city_name):
    fig, ax1 = plt.subplots(figsize=(10, 5))
    
    # Temperature and Humidity
    ax1.plot(chart_data["dates"], chart_data["temps"], marker='o', color='red', label='Temperature (°C)')
    ax1.plot(chart_data["dates"], chart_data["humidity"], marker='o', color='green', label='Humidity (%)')
    ax1.set_ylabel("Temperature/Humidity")
    ax1.legend(loc='upper left')
    ax1.grid()
    
    # Rain
    ax2 = ax1.twinx()
    ax2.bar(chart_data["dates"], chart_data["rain"], color='blue', alpha=0.3, label='Rain (mm)')
    ax2.set_ylabel("Rainfall")
    ax2.legend(loc='upper right')
    
    plt.title(f"5-Day Forecast for {city_name}")
    plt.xticks(rotation=45)
    plt.tight_layout()
    return fig

# CLI-specific functions
def cli_display_table(data, headers, title):
    print(f"\n--- {title} ---")
    print(tabulate(data, headers=headers, tablefmt="grid"))
    print("-----------------------")

def cli_main():
    # CLI implementation using the functions above
    pass  # (Original CLI implementation goes here)