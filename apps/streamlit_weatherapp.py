import streamlit as st
import matplotlib.pyplot as plt
import pandas as pd
from weather import (
    get_coordinates, fetch_data, prepare_current_weather,
    prepare_forecast_summary, plot_forecast, plot_comparison,
    WEATHER_URL, AIR_POLLUTION_URL, FORECAST_URL, AIR_POLLUTION_FORECAST_URL
)

# Initialize session state
if 'history' not in st.session_state:
    st.session_state.history = []

st.set_page_config(page_title="Weather Comparison App", layout="wide")
st.title("🌤️ Weather & Air Quality Explorer")
st.markdown("Explore weather data and compare forecasts interactively!")

# Style configuration
st.markdown("""
<style>
.big-font {
    font-size:24px !important;
    font-weight: bold !important;
    color: #1f77b4 !important;
}
</style>
""", unsafe_allow_html=True)

option = st.radio("Choose an option:", ["View Single City", "Compare Two Cities"])

def display_current_weather(city, country):
    try:
        lat, lon = get_coordinates(city, country)
        weather = fetch_data(WEATHER_URL, lat, lon)
        aqi_data = fetch_data(AIR_POLLUTION_URL, lat, lon)
        current = prepare_current_weather(weather, aqi_data)
        if current:
            st.subheader(f"📍 Current Weather: {city}, {country}")
            # Arrange weather details horizontally
            cols = st.columns(len(current))
            for idx, (k, v) in enumerate(current.items()):
                if k in ["Temperature", "Feels Like"]:
                    cols[idx].markdown(f'<p style="font-size:24px; font-weight:bold; color:#1f77b4;">{k}: {v}</p>', unsafe_allow_html=True)
                else:
                    cols[idx].markdown(f"**{k}**: {v}")
            return lat, lon
        return None, None
    except Exception as e:
        st.error(f"Error fetching current weather: {str(e)}")
        return None, None

def display_forecast(city, lat, lon):
    try:
        if lat is None or lon is None:
            return
            
        forecast = fetch_data(FORECAST_URL, lat, lon)
        air_forecast = fetch_data(AIR_POLLUTION_FORECAST_URL, lat, lon)

        if forecast and forecast.get('cod') == '200':
            st.subheader(f"📊 5-Day Forecast for {city}")
            fig = plot_forecast(forecast, city)
            if fig:
                st.pyplot(fig)
            else:
                st.warning("Forecast data not available for visualization")

            summary = prepare_forecast_summary(forecast, air_forecast)
            if summary:
                st.subheader("🧾 Forecast Summary")
                st.dataframe(pd.DataFrame(summary))
            else:
                st.warning("No forecast summary available")
        else:
            st.warning("No forecast data available for this location")
    except Exception as e:
        st.error(f"Error processing forecast: {str(e)}")

def show_search_history():
    if st.session_state.history:
        st.sidebar.subheader("🔍 Search History")
        for idx, (h_city, h_country) in enumerate(st.session_state.history[:5]):
            if st.sidebar.button(f"{h_city}, {h_country}", key=f"hist_{idx}"):
                st.session_state.selected_city = h_city
                st.session_state.selected_country = h_country
                st.experimental_rerun()

if option == "View Single City":
    show_search_history()
    with st.form("single_city_form"):
        city = st.text_input("City", value=st.session_state.get("selected_city", "London"))
        country = st.text_input("Country", value=st.session_state.get("selected_country", "UK"))
        submitted = st.form_submit_button("Get Weather")
        if submitted:
            lat, lon = display_current_weather(city, country)
            display_forecast(city, lat, lon)
            # Update history
            new_entry = (city, country)
            if new_entry not in st.session_state.history:
                st.session_state.history.insert(0, new_entry)
                st.session_state.history = st.session_state.history[:5]

elif option == "Compare Two Cities":
    show_search_history()
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
            try:
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

                # Current weather comparison
                if data1["weather"] and data2["weather"]:
                    st.subheader("🌍 Current Weather Comparison")
                    current1 = prepare_current_weather(data1["weather"], data1["aqi"])
                    current2 = prepare_current_weather(data2["weather"], data2["aqi"])
                    if current1 and current2:
                        comparison_df = pd.DataFrame({
                            "Attribute": current1.keys(),
                            city1: current1.values(),
                            city2: current2.values()
                        })
                        st.dataframe(comparison_df)

                # Forecast comparison
                if data1["forecast"] and data2["forecast"]:
                    st.subheader("📈 Forecast Comparison")
                    fig = plot_comparison(data1["forecast"], data2["forecast"], city1, city2)
                    if fig:
                        st.pyplot(fig)
                    
                    # Combined summary
                    summary1 = prepare_forecast_summary(data1["forecast"], data1["aqi_forecast"])
                    summary2 = prepare_forecast_summary(data2["forecast"], data2["aqi_forecast"])
                    if summary1 and summary2:
                        combined = pd.DataFrame(summary1 + summary2)
                        combined["City"] = [city1]*len(summary1) + [city2]*len(summary2)
                        st.subheader("🧾 Combined Forecast Summary")
                        st.dataframe(combined)

                # Update history
                for entry in [(city1, country1), (city2, country2)]:
                    if entry not in st.session_state.history:
                        st.session_state.history.insert(0, entry)
                        st.session_state.history = st.session_state.history[:5]

            except Exception as e:
                st.error(f"Comparison failed: {str(e)}")




