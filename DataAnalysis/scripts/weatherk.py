# === Weather Comparison Tool with OpenWeather API Integration ===

import os
import requests
from dotenv import load_dotenv
from datetime import datetime, timedelta
from tabulate import tabulate
import matplotlib.pyplot as plt
import pandas as pd
from collections import Counter
import kaggle_secrets

# Step 1: Load API key from secret
from kaggle_secrets import UserSecretsClient
user_secrets = UserSecretsClient()
API_KEY  = user_secrets.get_secret("OPENWEATHER_API_KEY")
if not API_KEY:
    raise ValueError("Please set the OPENWEATHER_API_KEY in the .env file.")

# ==================================================================================================
# Step 2: Define API URLs
GEO_URL = "http://api.openweathermap.org/geo/1.0/direct"
WEATHER_URL = "https://api.openweathermap.org/data/2.5/weather"
FORECAST_URL = "https://api.openweathermap.org/data/2.5/forecast"
AIR_POLLUTION_URL = "http://api.openweathermap.org/data/2.5/air_pollution"
AIR_POLLUTION_FORECAST_URL = "http://api.openweathermap.org/data/2.5/air_pollution/forecast"
HISTORICAL_URL = "https://api.openweathermap.org/data/2.5/onecall/timemachine"

# ====================================================================================================
# Helper function to convert AQI to text
def get_aqi_text(aqi):
    if aqi == 1:
        return "Good"
    elif aqi == 2:
        return "Fair"
    elif aqi == 3:
        return "Moderate"
    elif aqi == 4:
        return "Poor"
    elif aqi == 5:
        return "Very Poor"
    else:
        return "Unknown"

# Step 3: Get user input for city name
def get_user_input():
    city_name = input("Enter the city name: ").strip()
    if not city_name:
        raise ValueError("City name cannot be empty.")
    return city_name

# Step 4: Fetch latitude and longitude for the city
def get_city_coordinates(city_name):
    params = {
        "q": city_name,
        "limit": 1,
        "appid": API_KEY
    }
    response = requests.get(GEO_URL, params=params)
    if response.status_code == 200:
        data = response.json()
        if data:
            return data[0]["lat"], data[0]["lon"]
        else:
            raise ValueError(f"City '{city_name}' not found.")
    else:
        raise Exception(f"Error fetching coordinates: {response.status_code} - {response.json().get('message', 'Unknown error')}")

# ====================================================================================================
# Step 5: Fetch weather data
def fetch_weather_data(lat, lon):
    params = {
        "lat": lat,
        "lon": lon,
        "appid": API_KEY,
        "units": "metric"
    }
    response = requests.get(WEATHER_URL, params=params)
    if response.status_code == 200:
        return response.json()
    else:
        raise Exception(f"Error fetching weather data: {response.status_code} - {response.json().get('message', 'Unknown error')}")

# Step 6: Fetch forecast data
def fetch_forecast_data(lat, lon):
    params = {
        "lat": lat,
        "lon": lon,
        "appid": API_KEY,
        "units": "metric"
    }
    response = requests.get(FORECAST_URL, params=params)
    if response.status_code == 200:
        return response.json()
    else:
        raise Exception(f"Error fetching forecast data: {response.status_code} - {response.json().get('message', 'Unknown error')}")

# Step 7: Fetch air pollution data
def fetch_air_pollution_data(lat, lon):
    params = {
        "lat": lat,
        "lon": lon,
        "appid": API_KEY
    }
    response = requests.get(AIR_POLLUTION_URL, params=params)
    if response.status_code == 200:
        return response.json()
    else:
        raise Exception(f"Error fetching air pollution data: {response.status_code} - {response.json().get('message', 'Unknown error')}")

# Step 8: Fetch air pollution forecast data
def fetch_air_pollution_forecast(lat, lon):
    params = {
        "lat": lat,
        "lon": lon,
        "appid": API_KEY
    }
    response = requests.get(AIR_POLLUTION_FORECAST_URL, params=params)
    if response.status_code == 200:
        return response.json()
    else:
        raise Exception(f"Error fetching air pollution forecast data: {response.status_code} - {response.json().get('message', 'Unknown error')}")

# ====================================================================================================
# Step 9: Display weather data
def display_weather_data(data, air_pollution_data=None):
    if data.get("cod") != 200:
        print(f"Error: {data.get('message', 'Unknown error')}")
        return

    main_data = data.get("main", {})
    weather_data = data.get("weather", [{}])[0]
    wind_data = data.get("wind", {})
    sys_data = data.get("sys", {})
    rain_data = data.get("rain", {})
    dt = data.get("dt")
    timezone = data.get("timezone", 0)

    # Convert timestamps to readable format
    sunrise = datetime.utcfromtimestamp(sys_data.get("sunrise", 0)) + timedelta(seconds=timezone)
    sunset = datetime.utcfromtimestamp(sys_data.get("sunset", 0)) + timedelta(seconds=timezone)
    current_time = datetime.utcfromtimestamp(dt) + timedelta(seconds=timezone)

    # Prepare weather data for the table
    weather_table = [
        ["City", f"{data.get('name')}, {sys_data.get('country')}"],
        ["Current Time", current_time.strftime('%Y-%m-%d %H:%M:%S')],
        ["Temperature", f"{main_data.get('temp')}°C"],
        ["Feels Like", f"{main_data.get('feels_like')}°C"],
        ["Weather", weather_data.get('description')],
        ["Pressure", f"{main_data.get('pressure')} hPa"],
        ["Humidity", f"{main_data.get('humidity')}%"],
        ["Wind Speed", f"{wind_data.get('speed')} m/s"],
        ["Rain (1h)", f"{rain_data.get('1h', 0)} mm"],
        ["Sunrise", sunrise.strftime('%Y-%m-%d %H:%M:%S')],
        ["Sunset", sunset.strftime('%Y-%m-%d %H:%M:%S')]
    ]

    # Add AQI if available
    if air_pollution_data:
        aqi = air_pollution_data.get("list", [{}])[0].get("main", {}).get("aqi", "N/A")
        weather_table.append(["Air Quality Index (AQI)", get_aqi_text(aqi)])

    # Display the weather table (non-transposed)
    headers = ["Attribute", "Value"]
    print("\n--- Weather Details ---")
    print(tabulate(weather_table, headers=headers, tablefmt="grid"))
    
    print("-----------------------")

# ====================================================================================================
# Step 10: Display forecast data
def display_forecast_data(data, air_pollution_forecast_data, city_name):
    if data.get("cod") != "200":
        print(f"Error: {data.get('message', 'Unknown error')}")
        return

    # Extract data for plotting
    dates = []
    temps = []
    humidity = []
    rain = []
    for item in data.get("list", []):
        dates.append(datetime.fromtimestamp(item.get("dt")))
        temps.append(item.get("main", {}).get("temp"))
        humidity.append(item.get("main", {}).get("humidity"))
        rain.append(item.get("rain", {}).get("3h", 0))  # Rain in the last 3 hours

    # Create a single figure
    fig, ax1 = plt.subplots(figsize=(12, 6))

    # Plot temperature and humidity as lines
    ax1.plot(dates, temps, marker='o', color='red', label='Temperature (°C)')
    ax1.plot(dates, humidity, marker='o', color='green', label='Humidity (%)')
    ax1.set_ylabel("Temperature (°C) / Humidity (%)")
    ax1.legend(loc='upper left')
    ax1.grid()

    # Plot rain as bars
    ax2 = ax1.twinx()  # Create a second y-axis
    ax2.bar(dates, rain, color='blue', alpha=0.3, label='Rain (mm)')
    ax2.set_ylabel("Rain (mm)")
    ax2.legend(loc='upper right')

    # Set title and labels
    plt.title(f"5-Day Forecast for {city_name}: Temperature, Humidity, and Rain")
    plt.xlabel("Date")
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.show()

    # Process air pollution forecast data for AQI
    aqi_daily = {}
    if air_pollution_forecast_data and air_pollution_forecast_data.get("list"):
        for item in air_pollution_forecast_data["list"]:
            dt = datetime.fromtimestamp(item["dt"])
            date = dt.strftime('%Y-%m-%d')
            aqi = item.get("main", {}).get("aqi")
            if aqi is None:
                continue  # Skip if no AQI data
            if date not in aqi_daily:
                aqi_daily[date] = []
            aqi_daily[date].append(aqi)
        # Compute most common AQI category per day
        for date in aqi_daily:
            aqi_numbers = aqi_daily[date]
            aqi_texts = [get_aqi_text(num) for num in aqi_numbers]
            counts = Counter(aqi_texts)
            if counts:
                most_common = counts.most_common(1)[0][0]
            else:
                most_common = "Unknown"
            aqi_daily[date] = most_common

    # Prepare daily summary data for the table
    daily_data = {}
    for item in data.get("list", []):
        date = datetime.fromtimestamp(item.get("dt")).strftime('%Y-%m-%d')
        if date not in daily_data:
            daily_data[date] = {
                "temps": [],
                "humidity": [],
                "rain": [],
                "weather": []
            }
        daily_data[date]["temps"].append(item.get("main", {}).get("temp"))
        daily_data[date]["humidity"].append(item.get("main", {}).get("humidity"))
        daily_data[date]["rain"].append(item.get("rain", {}).get("3h", 0))
        daily_data[date]["weather"].append(item.get("weather", [{}])[0].get("description"))

    # Create a list of rows for the table
    summary_table = []
    for date, values in daily_data.items():
        avg_temp = sum(values["temps"]) / len(values["temps"])
        avg_humidity = sum(values["humidity"]) / len(values["humidity"])
        total_rain = sum(values["rain"])
        common_weather = max(set(values["weather"]), key=values["weather"].count)
        aqi_str = aqi_daily.get(date, "Unknown")
        summary_table.append([date, f"{avg_temp:.1f}°C", f"{avg_humidity:.1f}%", f"{total_rain:.1f} mm", common_weather, aqi_str])

    # Display the daily summary as a table
    print("\n--- Daily Forecast Summary ---")
    print(tabulate(summary_table, headers=["Date", "Avg Temp", "Avg Humidity", "Total Rain", "Weather", "AQI"], tablefmt="grid"))

# ====================================================================================================
# Step 11: Compare two cities
def compare_cities(city1_data, city2_data, city1_forecast, city2_forecast):
    # Prepare comparison table for current weather
    comparison_table = [
        ["City", city1_data.get("name"), city2_data.get("name")],
        ["Temperature (°C)", f"{city1_data.get('main', {}).get('temp')}", f"{city2_data.get('main', {}).get('temp')}"],
        ["Feels Like (°C)", f"{city1_data.get('main', {}).get('feels_like')}", f"{city2_data.get('main', {}).get('feels_like')}"],
        ["Weather", city1_data.get("weather", [{}])[0].get("description"), city2_data.get("weather", [{}])[0].get("description")],
        ["Pressure (hPa)", f"{city1_data.get('main', {}).get('pressure')}", f"{city2_data.get('main', {}).get('pressure')}"],
        ["Humidity (%)", f"{city1_data.get('main', {}).get('humidity')}", f"{city2_data.get('main', {}).get('humidity')}"],
        ["Wind Speed (m/s)", f"{city1_data.get('wind', {}).get('speed')}", f"{city2_data.get('wind', {}).get('speed')}"],
        ["Rain (1h) (mm)", f"{city1_data.get('rain', {}).get('1h', 0)}", f"{city2_data.get('rain', {}).get('1h', 0)}"]
    ]

    print("\n--- Current Weather Comparison ---")
    print(tabulate(comparison_table, headers=["Attribute", "City 1", "City 2"], tablefmt="grid"))

    # Plot forecast comparison
    dates1 = [datetime.fromtimestamp(item.get("dt")) for item in city1_forecast.get("list", [])]
    temps1 = [item.get("main", {}).get("temp") for item in city1_forecast.get("list", [])]
    humidity1 = [item.get("main", {}).get("humidity") for item in city1_forecast.get("list", [])]
    rain1 = [item.get("rain", {}).get("3h", 0) for item in city1_forecast.get("list", [])]

    dates2 = [datetime.fromtimestamp(item.get("dt")) for item in city2_forecast.get("list", [])]
    temps2 = [item.get("main", {}).get("temp") for item in city2_forecast.get("list", [])]
    humidity2 = [item.get("main", {}).get("humidity") for item in city2_forecast.get("list", [])]
    rain2 = [item.get("rain", {}).get("3h", 0) for item in city2_forecast.get("list", [])]

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 12))

    # Plot temperature and humidity for both cities on ax1
    ax1.plot(dates1, temps1, marker='o', color='red', label='Temperature (°C) - City 1')
    ax1.plot(dates1, humidity1, marker='o', color='green', label='Humidity (%) - City 1')
    ax1.plot(dates2, temps2, marker='o', color='orange', label='Temperature (°C) - City 2')
    ax1.plot(dates2, humidity2, marker='o', color='purple', label='Humidity (%) - City 2')
    ax1.set_ylabel("Temperature (°C) / Humidity (%)")
    ax1.legend(loc='upper left')
    ax1.grid()

    # Plot rain as line graphs for both cities on ax2
    ax2.plot(dates1, rain1, marker='o', color='blue', linestyle='-', label='Rain (mm) - City 1')
    ax2.plot(dates2, rain2, marker='o', color='cyan', linestyle='-', label='Rain (mm) - City 2')
    ax2.set_ylabel("Rain (mm)")
    ax2.legend(loc='upper right')
    ax2.grid()

    # Set title and labels
    plt.suptitle("5-Day Forecast Comparison: Temperature, Humidity, and Rain")
    plt.xlabel("Date")
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.show()

# ====================================================================================================
# Step 12: Main function
def main():
    try:
        print("What would you like to do?")
        print("1. View Weather and Forecast for a City")
        print("2. Compare Weather and Forecast for 2 Cities")
        choice = input("Enter your choice (1/2): ").strip()

        if choice == "1":
            city_name = get_user_input()
            lat, lon = get_city_coordinates(city_name)
            print(f"Coordinates for '{city_name}': Latitude = {lat}, Longitude = {lon}")

            weather_data = fetch_weather_data(lat, lon)
            air_pollution_data = fetch_air_pollution_data(lat, lon)
            display_weather_data(weather_data, air_pollution_data)

            forecast_data = fetch_forecast_data(lat, lon)
            air_pollution_forecast_data = fetch_air_pollution_forecast(lat, lon)
            display_forecast_data(forecast_data, air_pollution_forecast_data, city_name)

        elif choice == "2":
            city1_name = input("Enter the first city name: ").strip()
            city2_name = input("Enter the second city name: ").strip()

            lat1, lon1 = get_city_coordinates(city1_name)
            lat2, lon2 = get_city_coordinates(city2_name)

            city1_data = fetch_weather_data(lat1, lon1)
            city2_data = fetch_weather_data(lat2, lon2)
            city1_forecast = fetch_forecast_data(lat1, lon1)
            city2_forecast = fetch_forecast_data(lat2, lon2)

            compare_cities(city1_data, city2_data, city1_forecast, city2_forecast)

        else:
            print("Invalid choice. Please try again.")

    except Exception as e:
        print(f"An error occurred: {e}")

