import streamlit as st
import matplotlib.pyplot as plt
import pandas as pd
from weather import (
    get_coordinates, fetch_data, prepare_current_weather,
    prepare_forecast_summary, plot_forecast, plot_comparison,
    WEATHER_URL, AIR_POLLUTION_URL, FORECAST_URL, AIR_POLLUTION_FORECAST_URL
)

st.set_page_config(page_title="Weather Comparison App", layout="wide")
st.title("🌤️ Weather & Air Quality Explorer")
st.markdown("Explore weather data and compare forecasts interactively!")

option = st.radio("Choose an option:", ["View Single City", "Compare Two Cities"])

def display_current_weather(city, country):
    lat, lon = get_coordinates(city, country)
    weather = fetch_data(WEATHER_URL, lat, lon)
    aqi_data = fetch_data(AIR_POLLUTION_URL, lat, lon)
    current = prepare_current_weather(weather, aqi_data)
    if current:
        st.subheader(f"📍 Current Weather: {city}, {country}")
        for k, v in current.items():
            if k in ["Temperature", "Feels Like"]:
                st.markdown(f"**{k}: {v}**")
            else:
                st.markdown(f"{k}: {v}")
    return lat, lon

def display_forecast(city, lat, lon):
    forecast = fetch_data(FORECAST_URL, lat, lon)
    air_forecast = fetch_data(AIR_POLLUTION_FORECAST_URL, lat, lon)

    if forecast:
        st.subheader(f"📊 5-Day Forecast for {city}")
        fig = plot_forecast(forecast, city)
        if fig:
            st.pyplot(fig)
        else:
            st.error("Failed to generate forecast plot.")

        summary = prepare_forecast_summary(forecast, air_forecast)
        if summary:
            st.subheader("🧾 Forecast Summary")
            st.dataframe(pd.DataFrame(summary))
        else:
            st.warning("No forecast summary available.")

if option == "View Single City":
    with st.form("single_city_form"):
        city = st.text_input("City", "London")
        country = st.text_input("Country", "UK")
        submitted = st.form_submit_button("Get Weather")
        if submitted:
            lat, lon = display_current_weather(city, country)
            display_forecast(city, lat, lon)

elif option == "Compare Two Cities":
    with st.form("compare_form"):
        col1, col2 = st.columns(2)
        with col1:
            city1 = st.text_input("City 1", "London")
            country1 = st.text_input("Country 1", "UK")
        with col2:
            city2 = st.text_input("City 2", "New York")
            country2 = st.text_input("Country 2", "US")
        submitted = st.form_submit_button("Compare")
        if submitted:
            lat1, lon1 = get_coordinates(city1, country1)
            lat2, lon2 = get_coordinates(city2, country2)

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

            st.subheader("🌍 Current Weather Comparison")
            current1 = prepare_current_weather(data1["weather"], data1["aqi"])
            current2 = prepare_current_weather(data2["weather"], data2["aqi"])
            if current1 and current2:
                rows = list(current1.keys())
                st.dataframe({
                    "Attribute": rows,
                    city1: [current1[k] for k in rows],
                    city2: [current2[k] for k in rows]
                })

            if data1["forecast"] and data2["forecast"]:
                st.subheader("📈 Forecast Comparison")
                fig = plot_comparison(data1["forecast"], data2["forecast"], city1, city2)
                if fig:
                    st.pyplot(fig)
                else:
                    st.error("Failed to generate comparison plot.")

                summary1 = prepare_forecast_summary(data1["forecast"], data1["aqi_forecast"])
                summary2 = prepare_forecast_summary(data2["forecast"], data2["aqi_forecast"])
                if summary1 and summary2:
                    for row in summary1:
                        row["City"] = city1
                    for row in summary2:
                        row["City"] = city2
                    combined = summary1 + summary2
                    st.subheader("🧾 Combined Forecast Summary")
                    st.dataframe(pd.DataFrame(combined))