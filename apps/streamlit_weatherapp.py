# app.py (Streamlit interface)
import streamlit as st
import sys
from pathlib import Path
from weather import (
    get_city_coordinates, fetch_weather_data, fetch_forecast_data,
    fetch_air_pollution_data, fetch_air_pollution_forecast,
    prepare_weather_table, prepare_forecast_chart_data, create_forecast_figure
)

# Configure system path
script_path = Path(__file__).resolve().parent / "DataAnalysis" / "scripts"
sys.path.append(str(script_path))

# Streamlit app
st.title("Weather Explorer")
st.markdown("Explore weather data and compare forecasts interactively!")

option = st.radio(
    "Choose an action:",
    ("View City Forecast", "Compare Cities")
)

if option == "View City Forecast":
    city = st.text_input("Enter city name:")
    if city:
        try:
            lat, lon = get_city_coordinates(city)
            st.success(f"Coordinates found: {lat:.2f}, {lon:.2f}")
            
            # Current weather
            weather_data = fetch_weather_data(lat, lon)
            pollution_data = fetch_air_pollution_data(lat, lon)
            weather_table = prepare_weather_table(weather_data, pollution_data)
            
            st.subheader("Current Conditions")
            st.table(weather_table)
            
            # Forecast
            forecast_data = fetch_forecast_data(lat, lon)
            chart_data = prepare_forecast_chart_data(forecast_data)
            fig = create_forecast_figure(chart_data, city)
            
            st.subheader("5-Day Forecast")
            st.pyplot(fig)
            
        except Exception as e:
            st.error(f"Error: {str(e)}")

elif option == "Compare Cities":
    col1, col2 = st.columns(2)
    with col1:
        city1 = st.text_input("City 1:")
    with col2:
        city2 = st.text_input("City 2:")
    
    if city1 and city2:
        try:
            # City 1 Data
            lat1, lon1 = get_city_coordinates(city1)
            weather1 = fetch_weather_data(lat1, lon1)
            forecast1 = fetch_forecast_data(lat1, lon1)
            
            # City 2 Data
            lat2, lon2 = get_city_coordinates(city2)
            weather2 = fetch_weather_data(lat2, lon2)
            forecast2 = fetch_forecast_data(lat2, lon2)
            
            # Comparison Table
            comparison_data = [
                ["Temperature", f"{weather1['main']['temp']}°C", f"{weather2['main']['temp']}°C"],
                ["Humidity", f"{weather1['main']['humidity']}%", f"{weather2['main']['humidity']}%"],
                ["Wind Speed", f"{weather1['wind']['speed']} m/s", f"{weather2['wind']['speed']} m/s"]
            ]
            
            st.subheader("Current Comparison")
            st.table(comparison_data)
            
            # Forecast Comparison Chart
            fig, ax = plt.subplots(figsize=(10, 5))
            chart1 = prepare_forecast_chart_data(forecast1)
            chart2 = prepare_forecast_chart_data(forecast2)
            
            ax.plot(chart1["dates"], chart1["temps"], label=f"{city1} Temp")
            ax.plot(chart2["dates"], chart2["temps"], label=f"{city2} Temp")
            ax.set_ylabel("Temperature (°C)")
            ax.legend()
            ax.grid()
            
            st.subheader("Temperature Forecast Comparison")
            st.pyplot(fig)
            
        except Exception as e:
            st.error(f"Comparison error: {str(e)}")