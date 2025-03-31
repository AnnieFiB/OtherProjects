import os
from pathlib import Path
#from dotenv import load_dotenv
import requests
from datetime import datetime, timedelta
from collections import Counter
import matplotlib.pyplot as plt
import pandas as pd
import streamlit as st

# Step 1: Load API key from .env file
API_KEY = st.secrets.get("OPENWEATHER_API_KEY")

# API endpoints
GEO_URL = "http://api.openweathermap.org/geo/1.0/direct"
WEATHER_URL = "https://api.openweathermap.org/data/2.5/weather"
FORECAST_URL = "https://api.openweathermap.org/data/2.5/forecast"
AIR_POLLUTION_URL = "http://api.openweathermap.org/data/2.5/air_pollution"
AIR_POLLUTION_FORECAST_URL = "http://api.openweathermap.org/data/2.5/air_pollution/forecast"

def get_aqi_text(aqi):
    """Convert AQI value to text description"""
    aqi_map = {
        1: "Good", 2: "Fair", 3: "Moderate",
        4: "Poor", 5: "Very Poor"
    }
    return aqi_map.get(aqi, "Unknown")

def get_user_input():
    """Get city and country separated by last space"""
    while True:
        location = input("Enter city and country (e.g., 'London UK'): ").strip()
        if location.count(" ") >= 1:
            parts = location.rsplit(" ", 1)
            return parts[0], parts[1]
        print("Invalid format. Use 'City Country'")

def get_coordinates(city, country):
    """Get latitude and longitude for location"""
    params = {"q": f"{city},{country}", "limit": 1, "appid": API_KEY}
    response = requests.get(GEO_URL, params=params)
    if response.status_code == 200 and response.json():
        data = response.json()[0]
        return data["lat"], data["lon"]
    raise ValueError(f"Location not found: {city}, {country}")

def fetch_data(url, lat, lon, params=None):
    """Generic API request handler"""
    base_params = {"lat": lat, "lon": lon, "appid": API_KEY, "units": "metric"}
    if params: base_params.update(params)
    try:
        response = requests.get(url, params=base_params, timeout=15)
        return response.json() if response.status_code == 200 else None
    except Exception as e:
        print(f"API Error: {str(e)}")
        return None

def prepare_current_weather(data, aqi_data=None):
    """Process current weather data"""
    if not data or data.get("cod") != 200:
        return None
    
    main = data.get("main", {})
    weather = data.get("weather", [{}])[0]
    wind = data.get("wind", {})
    sys = data.get("sys", {})
    rain = data.get("rain", {}).get("1h", 0)
    tz_offset = data.get("timezone", 0)
    
    current_time = datetime.utcfromtimestamp(data["dt"]) + timedelta(seconds=tz_offset)
    sunrise = datetime.utcfromtimestamp(sys.get("sunrise", 0)) + timedelta(seconds=tz_offset)
    sunset = datetime.utcfromtimestamp(sys.get("sunset", 0)) + timedelta(seconds=tz_offset)

    weather_info = {
        "City": f"{data['name']}, {sys.get('country')}",
        "Temperature": f"{main.get('temp', 'N/A')}°C",
        "Feels Like": f"{main.get('feels_like', 'N/A')}°C",
        "Pressure": f"{main.get('pressure', 'N/A')} hPa",
        "Humidity": f"{main.get('humidity', 'N/A')}%",
        "Weather": weather.get('description', 'N/A'),
        "Wind Speed": f"{wind.get('speed', 'N/A')} m/s",
        "Rain (1h)": f"{rain} mm",
        "Sunrise": sunrise.strftime('%H:%M:%S'),
        "Sunset": sunset.strftime('%H:%M:%S'),
        "Updated": current_time.strftime('%Y-%m-%d %H:%M:%S')
    }

    if aqi_data and aqi_data.get("list"):
        aqi = aqi_data["list"][0].get("main", {}).get("aqi", "N/A")
        weather_info["AQI"] = f"{get_aqi_text(aqi)} ({aqi})" if aqi != "N/A" else "N/A"
    
    return weather_info

def prepare_forecast_summary(forecast, air_forecast=None, city_name=""):
    """Create daily forecast statistics with city-specific columns"""
    if not forecast or not forecast.get("list"):
        return None

    try:
        # Process forecast data
        forecast_items = []
        for item in forecast["list"]:
            dt = datetime.fromtimestamp(item["dt"])
            forecast_items.append({
                "date": dt.strftime('%Y-%m-%d'),
                "temp": item["main"].get("temp", 0),
                "humidity": item["main"].get("humidity", 0),
                "rain": item.get("rain", {}).get("3h", 0),
                "pressure": item["main"].get("pressure", 0),
                "wind": item["wind"].get("speed", 0)
            })
        
        df = pd.DataFrame(forecast_items)
        
        # Add AQI data
        if air_forecast and air_forecast.get("list"):
            aqi_dict = {}
            for item in air_forecast["list"]:
                dt = datetime.fromtimestamp(item["dt"]).strftime('%Y-%m-%d')
                aqi = item["main"].get("aqi")
                if dt not in aqi_dict:
                    aqi_dict[dt] = []
                aqi_dict[dt].append(aqi)
            
            df["aqi"] = df["date"].map(lambda d: 
                max(aqi_dict.get(d, []), key=aqi_dict.get(d, []).count) 
                if aqi_dict.get(d) else None)

        # Daily aggregation and city prefix
        daily = df.groupby("date").agg({
            "temp": "mean",
            "humidity": "mean",
            "rain": "sum",
            "pressure": "mean",
            "wind": "mean",
            "aqi": lambda x: x.mode()[0] if not x.isnull().all() else None
        }).reset_index()
        
        # Format values and add units
        daily["temp"] = daily["temp"].round(1).astype(str) + '°C'
        daily["humidity"] = daily["humidity"].round(1).astype(str) + '%'
        daily["rain"] = daily["rain"].round(1).astype(str) + ' mm'
        daily["pressure"] = daily["pressure"].round(1).astype(str) + ' hPa'
        daily["wind"] = daily["wind"].round(1).astype(str) + ' m/s'
        
        # Fixed AQI formatting (text only)
        daily["aqi"] = daily["aqi"].apply(
            lambda x: get_aqi_text(x) if pd.notnull(x) else "N/A"
        )
        
        daily.columns = ["Date", 
                        f"Avg Temp ({city_name})",
                        f"Avg Humidity ({city_name})",
                        f"Total Rain ({city_name})",
                        f"Avg Pressure ({city_name})",
                        f"Avg Wind ({city_name})",
                        f"AQI ({city_name})"]
        
        return daily.to_dict('records')
    
    except Exception as e:
        print(f"Forecast processing error: {str(e)}")
        return None

def plot_forecast(forecast, city_name):
    """Display single city forecast plots"""
    if not forecast or not forecast.get("list"):
        return None

    try:
        df = pd.DataFrame([{
            "datetime": datetime.fromtimestamp(item["dt"]),
            "temp": item["main"].get("temp", 0),
            "humidity": item["main"].get("humidity", 0),
            "rain": item.get("rain", {}).get("3h", 0)
        } for item in forecast["list"]])

        fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(12, 8))
        
        ax1.plot(df["datetime"], df["temp"], 'r-')
        ax1.set_ylabel('Temperature (°C)')
        ax1.set_title(f'{city_name} Forecast')
        ax1.grid(True)

        ax2.plot(df["datetime"], df["humidity"], 'b-')
        ax2.set_ylabel('Humidity (%)')
        ax2.grid(True)

        ax3.bar(df["datetime"], df["rain"], width=0.05, color='blue', alpha=0.5)
        ax3.set_ylabel('Rain (mm)')
        ax3.grid(True)

        plt.tight_layout()
        return fig  # Return figure instead of showing
    
    except Exception as e:
        print(f"Plotting error: {str(e)}")
        return None

def plot_comparison(forecast1, forecast2, city1_name, city2_name):
    """Compare two cities' forecasts visually"""
    def prepare_df(forecast):
        return pd.DataFrame([{
            "datetime": datetime.fromtimestamp(item["dt"]),
            "temp": item["main"]["temp"],
            "humidity": item["main"]["humidity"],
            "rain": item.get("rain", {}).get("3h", 0)
        } for item in forecast["list"]])
    
    df1 = prepare_df(forecast1)
    df2 = prepare_df(forecast2)

    fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(12, 10))
    
    # Temperature comparison
    ax1.plot(df1["datetime"], df1["temp"], label=city1_name, marker='o')
    ax1.plot(df2["datetime"], df2["temp"], label=city2_name, marker='o')
    ax1.set_ylabel('Temperature (°C)')
    ax1.legend()
    ax1.grid(True)
    ax1.set_title(f'Temperature Comparison: {city1_name} vs {city2_name}')

    # Humidity comparison
    ax2.plot(df1["datetime"], df1["humidity"], label=city1_name, marker='o')
    ax2.plot(df2["datetime"], df2["humidity"], label=city2_name, marker='o')
    ax2.set_ylabel('Humidity (%)')
    ax2.legend()
    ax2.grid(True)
    ax2.set_title('Humidity Comparison')

    # Rain comparison
    ax3.bar(df1["datetime"], df1["rain"], width=0.03, label=city1_name, alpha=0.5)
    ax3.bar(df2["datetime"], df2["rain"], width=0.03, label=city2_name, alpha=0.5)
    ax3.set_ylabel('Rain (mm)')
    ax3.legend()
    ax3.grid(True)
    ax3.set_title('Rainfall Comparison')

    plt.tight_layout()
    return fig  # Return figure instead of showing


def main():
    """Main application flow"""
    try:
        print("1. View city weather\n2. Compare cities")
        choice = input("Choose option (1/2): ").strip()

        if choice == "1":
            city, country = get_user_input()
            lat, lon = get_coordinates(city, country)
            
            # Fetch data
            weather = fetch_data(WEATHER_URL, lat, lon)
            aqi_current = fetch_data(AIR_POLLUTION_URL, lat, lon)
            forecast = fetch_data(FORECAST_URL, lat, lon)
            aqi_forecast = fetch_data(AIR_POLLUTION_FORECAST_URL, lat, lon)
            
            # Current weather
            current = prepare_current_weather(weather, aqi_current)
            if current:
                print("\n=== Current Weather ===")
                print(tabulate([(k, v) for k, v in current.items()], tablefmt="grid"))
            
            # Forecast visualization
            if forecast:
                plot_forecast(forecast, city)
            
            # Forecast summary
            summary = prepare_forecast_summary(forecast, aqi_forecast)
            if summary:
                print("\n=== Daily Forecast Summary ===")
                print(tabulate(summary, headers="keys", tablefmt="grid"))

        elif choice == "2":
            print("\nFirst city:")
            city1, country1 = get_user_input()
            lat1, lon1 = get_coordinates(city1, country1)
            
            print("\nSecond city:")
            city2, country2 = get_user_input()
            lat2, lon2 = get_coordinates(city2, country2)
            
            # Fetch data
            data1 = {
                "weather": fetch_data(WEATHER_URL, lat1, lon1),
                "aqi": fetch_data(AIR_POLLUTION_URL, lat1, lon1),
                "forecast": fetch_data(FORECAST_URL, lat1, lon1),
                "aqi_forecast": fetch_data(AIR_POLLUTION_FORECAST_URL, lat1, lon1)
            }
            data2 = {
                "weather": fetch_data(WEATHER_URL, lat2, lon2),
                "aqi": fetch_data(AIR_POLLUTION_URL, lat2, lon2),
                "forecast": fetch_data(FORECAST_URL, lat2, lon2),
                "aqi_forecast": fetch_data(AIR_POLLUTION_FORECAST_URL, lat2, lon2)
            }
            
            # Current comparison
            current1 = prepare_current_weather(data1["weather"], data1["aqi"])
            current2 = prepare_current_weather(data2["weather"], data2["aqi"])
            if current1 and current2:
                print("\n=== Current Weather Comparison ===")
                comparison = []
                for key in current1:
                    comparison.append([key, current1[key], current2[key]])
                print(tabulate(comparison, headers=["Attribute", "City 1", "City 2"], tablefmt="grid"))
            
            # Forecast comparison
            if data1["forecast"] and data2["forecast"]:
                # Visual comparison
                plot_comparison(data1["forecast"], data2["forecast"], city1, city2)
                
                # Prepare summaries
                summary1 = prepare_forecast_summary(data1["forecast"], data1["aqi_forecast"])
                summary2 = prepare_forecast_summary(data2["forecast"], data2["aqi_forecast"])
                
                if summary1 and summary2:
                    # Add city identifiers
                    for entry in summary1:
                        entry["City"] = city1
                    for entry in summary2:
                        entry["City"] = city2
                    
                    combined = summary1 + summary2
                    print("\n=== Combined Forecast Summary ===")
                    print(tabulate(combined, headers="keys", tablefmt="grid"))

        else:
            print("Invalid choice")

    except Exception as e:
        print(f"Application error: {str(e)}")

if __name__ == "__main__":
    main()