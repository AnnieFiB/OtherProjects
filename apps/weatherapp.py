import streamlit as st
import sys
from pathlib import Path
import matplotlib.pyplot as plt
from tabulate import tabulate
from datetime import datetime, timedelta
from collections import Counter

# Add the path to weather.py to the system path
script_path = Path(__file__).resolve().parent / "DataAnalysis" / "scripts"
sys.path.append(str(script_path))

import weather  # Import the weather module from its subdirectory

# App Title and Description
st.title("Weather Explorer")
st.markdown("Explore weather data and compare forecasts interactively!")

# Helper function to display weather data as a table
def display_weather_table(data, air_pollution_data=None):
    if data.get("cod") != 200:
        st.error(f"Error: {data.get('message', 'Unknown error')}")
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
        weather_table.append(["Air Quality Index (AQI)", weather.get_aqi_text(aqi)])

    # Display the weather table
    st.table(weather_table)

# Helper function to display forecast data
def display_forecast_chart_and_table(data, air_pollution_forecast_data, city_name):
    if data.get("cod") != "200":
        st.error(f"Error: {data.get('message', 'Unknown error')}")
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
    fig, ax1 = plt.subplots(figsize=(10, 5))

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
    
    # Display the chart in Streamlit
    st.pyplot(fig)

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
            aqi_texts = [weather.get_aqi_text(num) for num in aqi_numbers]
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
    st.subheader("Daily Forecast Summary")
    st.table(summary_table)

# Helper function to compare two cities
def display_city_comparison(city1_data, city2_data, city1_forecast, city2_forecast):
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

    st.subheader("Current Weather Comparison")
    st.table(comparison_table)

    # Plot forecast comparison
    dates1 = [datetime.fromtimestamp(item.get("dt")) for item in city1_forecast.get("list", [])]
    temps1 = [item.get("main", {}).get("temp") for item in city1_forecast.get("list", [])]
    humidity1 = [item.get("main", {}).get("humidity") for item in city1_forecast.get("list", [])]
    rain1 = [item.get("rain", {}).get("3h", 0) for item in city1_forecast.get("list", [])]

    dates2 = [datetime.fromtimestamp(item.get("dt")) for item in city2_forecast.get("list", [])]
    temps2 = [item.get("main", {}).get("temp") for item in city2_forecast.get("list", [])]
    humidity2 = [item.get("main", {}).get("humidity") for item in city2_forecast.get("list", [])]
    rain2 = [item.get("rain", {}).get("3h", 0) for item in city2_forecast.get("list", [])]

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 10))

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
    
    # Display the chart in Streamlit
    st.pyplot(fig)

# Options for User Choice
st.markdown("### What would you like to do?")
option = st.radio(
    "Choose an action:",
    ("View Weather and Forecast for a City", "Compare Weather for Two Cities")
)

# Option 1: View Weather and Forecast for a City
if option == "View Weather and Forecast for a City":
    city_name = st.text_input("Enter the city name:")
    if city_name:
        try:
            # Fetch Coordinates
            lat, lon = weather.get_city_coordinates(city_name)
            st.write(f"Coordinates for **{city_name}**: Latitude = {lat}, Longitude = {lon}")

            # Fetch Weather Data
            weather_data = weather.fetch_weather_data(lat, lon)
            air_pollution_data = weather.fetch_air_pollution_data(lat, lon)

            # Display Current Weather
            st.subheader("Current Weather")
            display_weather_table(weather_data, air_pollution_data)

            # Fetch and Display Forecast Data
            forecast_data = weather.fetch_forecast_data(lat, lon)
            air_pollution_forecast_data = weather.fetch_air_pollution_forecast(lat, lon)
            
            st.subheader("5-Day Forecast")
            display_forecast_chart_and_table(forecast_data, air_pollution_forecast_data, city_name)
            
        except Exception as e:
            st.error(f"An error occurred: {e}")

# Option 2: Compare Weather for Two Cities
elif option == "Compare Weather for Two Cities":
    col1, col2 = st.columns(2)
    with col1:
        city1_name = st.text_input("Enter the first city name:")
    with col2:
        city2_name = st.text_input("Enter the second city name:")

    if city1_name and city2_name:
        try:
            # Fetch Coordinates for Both Cities
            lat1, lon1 = weather.get_city_coordinates(city1_name)
            lat2, lon2 = weather.get_city_coordinates(city2_name)

            # Fetch Weather Data
            city1_data = weather.fetch_weather_data(lat1, lon1)
            city2_data = weather.fetch_weather_data(lat2, lon2)

            # Fetch Forecast Data
            city1_forecast = weather.fetch_forecast_data(lat1, lon1)
            city2_forecast = weather.fetch_forecast_data(lat2, lon2)

            # Display Comparison
            st.subheader(f"Weather Comparison: **{city1_name}** vs **{city2_name}**")
            display_city_comparison(city1_data, city2_data, city1_forecast, city2_forecast)
            
        except Exception as e:
            st.error(f"An error occurred: {e}")