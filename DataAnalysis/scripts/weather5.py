# === Weather Comparison Tool with OpenWeather API Integration ===

import os
import requests
from dotenv import load_dotenv
from datetime import datetime, timedelta
from tabulate import tabulate
import matplotlib.pyplot as plt
from collections import Counter

# Load environment variables
load_dotenv()
API_KEY = os.getenv("OPENWEATHER_API_KEY")
if not API_KEY:
    raise ValueError("Please set the OPENWEATHER_API_KEY in the .env file.")

# API endpoints
GEO_URL = "http://api.openweathermap.org/geo/1.0/direct"
WEATHER_URL = "https://api.openweathermap.org/data/2.5/weather"
FORECAST_URL = "https://api.openweathermap.org/data/2.5/forecast"
AIR_POLLUTION_URL = "http://api.openweathermap.org/data/2.5/air_pollution"

def get_aqi_text(aqi):
    """Convert AQI numerical value to text description"""
    return {
        1: "Good",
        2: "Fair",
        3: "Moderate",
        4: "Poor",
        5: "Very Poor"
    }.get(aqi, "Unknown")

def get_user_input():
    """Get validated city name from user"""
    city = input("Enter city name: ").strip()
    return city or get_user_input()

def get_city_coordinates(city_name):
    """Fetch geographic coordinates for a city"""
    params = {"q": city_name, "limit": 1, "appid": API_KEY}
    response = requests.get(GEO_URL, params=params)
    if response.status_code == 200 and response.json():
        data = response.json()[0]
        return data["lat"], data["lon"]
    raise ValueError(f"City '{city_name}' not found")

def fetch_weather_data(lat, lon):
    """Fetch current weather data"""
    params = {"lat": lat, "lon": lon, "appid": API_KEY, "units": "metric"}
    response = requests.get(WEATHER_URL, params=params)
    response.raise_for_status()
    return response.json()

def fetch_forecast_data(lat, lon):
    """Fetch weather forecast data"""
    params = {"lat": lat, "lon": lon, "appid": API_KEY, "units": "metric"}
    response = requests.get(FORECAST_URL, params=params)
    response.raise_for_status()
    return response.json()

def fetch_air_pollution_data(lat, lon):
    """Fetch air pollution data"""
    params = {"lat": lat, "lon": lon, "appid": API_KEY}
    response = requests.get(AIR_POLLUTION_URL, params=params)
    response.raise_for_status()
    return response.json()

def display_weather_data(data, air_pollution):
    """Display current weather information"""
    weather_table = [
        ["City", f"{data['name']}, {data['sys']['country']}"],
        ["Temperature", f"{data['main']['temp']}°C"],
        ["Feels Like", f"{data['main']['feels_like']}°C"],
        ["Weather", data['weather'][0]['description']],
        ["Humidity", f"{data['main']['humidity']}%"],
        ["Wind Speed", f"{data['wind']['speed']} m/s"],
        ["AQI", get_aqi_text(air_pollution['list'][0]['main']['aqi'])]
    ]
    print("\nCurrent Weather:")
    print(tabulate(weather_table, tablefmt="grid"))

def plot_forecast(forecast, city_name):
    """Generate forecast visualization"""
    dates = [datetime.fromtimestamp(item['dt']) for item in forecast['list']]
    temps = [item['main']['temp'] for item in forecast['list']]
    rain = [item.get('rain', {}).get('3h', 0) for item in forecast['list']]

    fig, ax = plt.subplots(figsize=(12, 6))
    ax.plot(dates, temps, label='Temperature', color='red')
    ax.set_xlabel('Date')
    ax.set_ylabel('Temperature (°C)')
    ax2 = ax.twinx()
    ax2.bar(dates, rain, alpha=0.3, color='blue', label='Rainfall')
    ax2.set_ylabel('Rainfall (mm)')
    plt.title(f"5-Day Forecast for {city_name}")
    fig.legend(loc='upper left')
    plt.show()

def compare_cities(city1_data, city2_data, city1_forecast, city2_forecast, aqi1, aqi2):
    """Compare two cities' weather data"""
    # Comparison table
    comparison = [
        ["City", city1_data['name'], city2_data['name']],
        ["Temperature", f"{city1_data['main']['temp']}°C", f"{city2_data['main']['temp']}°C"],
        ["Humidity", f"{city1_data['main']['humidity']}%", f"{city2_data['main']['humidity']}%"],
        ["AQI", get_aqi_text(aqi1['list'][0]['main']['aqi']), 
               get_aqi_text(aqi2['list'][0]['main']['aqi'])]
    ]
    print("\nCity Comparison:")
    print(tabulate(comparison, headers=["Metric", "City 1", "City 2"], tablefmt="grid"))

    # Forecast visualization
    dates = [datetime.fromtimestamp(item['dt']) for item in city1_forecast['list']]
    plt.figure(figsize=(12, 6))
    plt.plot(dates, [item['main']['temp'] for item in city1_forecast['list']], label=city1_data['name'])
    plt.plot(dates, [item['main']['temp'] for item in city2_forecast['list']], label=city2_data['name'])
    plt.title("Temperature Forecast Comparison")
    plt.xlabel("Date")
    plt.ylabel("Temperature (°C)")
    plt.legend()
    plt.xticks(rotation=45)
    plt.show()

def save_data(data, comparison=False):
    """Save weather data to file"""
    filename = f"weather_data_{datetime.now().strftime('%Y%m%d_%H%M')}.txt"
    with open(filename, 'w') as f:
        f.write("Weather Data Report\n")
        f.write(tabulate(data, tablefmt="grid"))
    print(f"Data saved to {filename}")

def main():
    """Main application flow"""
    try:
        print("1. Single City\n2. Compare Cities")
        choice = input("Choose option (1/2): ").strip()

        if choice == '1':
            city = get_user_input()
            lat, lon = get_city_coordinates(city)
            weather = fetch_weather_data(lat, lon)
            forecast = fetch_forecast_data(lat, lon)
            aqi = fetch_air_pollution_data(lat, lon)
            
            display_weather_data(weather, aqi)
            plot_forecast(forecast, city)
            
            if input("Save data? (y/n): ").lower() == 'y':
                save_data([
                    ["City", city],
                    ["Temperature", f"{weather['main']['temp']}°C"],
                    ["Humidity", f"{weather['main']['humidity']}%"],
                    ["AQI", get_aqi_text(aqi['list'][0]['main']['aqi'])]
                ])

        elif choice == '2':
            city1 = get_user_input()
            city2 = get_user_input()
            
            lat1, lon1 = get_city_coordinates(city1)
            lat2, lon2 = get_city_coordinates(city2)
            
            data1 = fetch_weather_data(lat1, lon1)
            data2 = fetch_weather_data(lat2, lon2)
            forecast1 = fetch_forecast_data(lat1, lon1)
            forecast2 = fetch_forecast_data(lat2, lon2)
            aqi1 = fetch_air_pollution_data(lat1, lon1)
            aqi2 = fetch_air_pollution_data(lat2, lon2)
            
            compare_cities(data1, data2, forecast1, forecast2, aqi1, aqi2)
            
            if input("Save comparison? (y/n): ").lower() == 'y':
                save_data([
                    ["Metric", "City 1", "City 2"],
                    ["Temperature", f"{data1['main']['temp']}°C", f"{data2['main']['temp']}°C"],
                    ["Humidity", f"{data1['main']['humidity']}%", f"{data2['main']['humidity']}%"],
                    ["AQI", get_aqi_text(aqi1['list'][0]['main']['aqi']), 
                           get_aqi_text(aqi2['list'][0]['main']['aqi'])]
                ], comparison=True)

        else:
            print("Invalid choice")

    except Exception as e:
        print(f"Error: {str(e)}")
