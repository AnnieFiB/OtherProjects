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
if 'selected_city' not in st.session_state:
    st.session_state.selected_city = "London"
if 'selected_country' not in st.session_state:
    st.session_state.selected_country = "England"

st.set_page_config(page_title="Weather Comparison App", layout="wide")
st.title("Weather & Air Quality Explorer")
st.markdown("Explore weather data and compare forecasts interactively!")

option = st.radio("Choose an option:", ["View Single City", "Compare Two Cities"], horizontal=True)

def update_history(city, country):
    """Update search history with new entry"""
    entry = (city.strip().title(), country.strip().upper())
    history = [e for e in st.session_state.history if e != entry]
    history.insert(0, entry)
    st.session_state.history = history[:5]


def show_search_history():
    """Display clickable search history in sidebar"""
    if st.session_state.history:
        st.sidebar.subheader("Recent Searches")
        for idx, (h_city, h_country) in enumerate(st.session_state.history):
            if st.sidebar.button(
                label=f"{h_city}, {h_country}",
                key=f"hist_{idx}",
                help="Click to search again",
                use_container_width=True
            ):
                st.session_state.selected_city = h_city
                st.session_state.selected_country = h_country
                st.rerun()


def display_current_weather(city, country):
    try:
        lat, lon = get_coordinates(city, country)
        weather = fetch_data(WEATHER_URL, lat, lon)
        aqi_data = fetch_data(AIR_POLLUTION_URL, lat, lon)
        current = prepare_current_weather(weather, aqi_data)
        
        if current:
            st.subheader(f" Current Weather: {city}, {country}")
            st.write(f"Current Date: {current['Updated']}")
            
            # Main metrics row
            col1, col2, col3, col4 = st.columns(4)
            col1.markdown(f'<p style="font-size:24px; font-weight:1000; color:#1a73e8;">🌡️Temperature: {current["Temperature"]}</p>', unsafe_allow_html=True)
            col2.markdown(f'<p style="font-size:24px; font-weight:1000; color:#1a73e8;">💨Feels Like: {current["Feels Like"]}</p>', unsafe_allow_html=True)
            col3.markdown(f'<p style="font-size:24px; font-weight:1000; color:#1a73e8;">🌫️Air Quality: {current["AQI"]}</p>', unsafe_allow_html=True)

            # Secondary metrics row
            col5, col6, col7, col8 = st.columns(4)
            
            # Column 5
            col5.markdown(f'<p style="font-size:20px; font-weight:600; margin:12px 0;">🌤️{current["Weather"]}</p>', unsafe_allow_html=True)
            col5.markdown(f'<p style="font-size:20px; font-weight:600; margin:12px 0;">💧Humidity: {current["Humidity"]}</p>', unsafe_allow_html=True)
            
            # Column 6
            col6.markdown(f'<p style="font-size:20px; font-weight:600; margin:12px 0;">🎐Wind Speed: {current["Wind Speed"]}</p>', unsafe_allow_html=True)
            col6.markdown(f'<p style="font-size:20px; font-weight:600; margin:12px 0;">🌧️Rain: {current["Rain (1h)"]}</p>', unsafe_allow_html=True)
            
            # Column 7
            col7.markdown(f'<p style="font-size:20px; font-weight:600; margin:12px 0;">🌅Sunrise: {current["Sunrise"]}</p>', unsafe_allow_html=True)
            col7.markdown(f'<p style="font-size:20px; font-weight:600; margin:12px 0;">🌇Sunset: {current["Sunset"]}</p>', unsafe_allow_html=True)
            
            # Column 8
            col8.markdown(f'<p style="font-size:20px; font-weight:600; margin:12px 0;">📊Pressure: {current["Pressure"]}</p>', unsafe_allow_html=True)

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
            st.subheader(f" 5-Day Forecast for {city}")
            fig = plot_forecast(forecast, city)
            if fig:
                st.pyplot(fig)
            else:
                st.warning("Forecast data not available for visualization")

            summary = prepare_forecast_summary(forecast, air_forecast, city)
            if summary:
                st.subheader("Forecast Summary")
                st.dataframe(pd.DataFrame(summary), use_container_width=True)
            else:
                st.warning("No forecast summary available")
        else:
            st.warning("No forecast data available for this location")
    except Exception as e:
        st.error(f"Error processing forecast: {str(e)}")

if option == "View Single City":
    show_search_history()
    with st.form("single_city_form"):
        city = st.text_input("City", value=st.session_state.selected_city)
        country = st.text_input("Country", value=st.session_state.selected_country)
        submitted = st.form_submit_button("Get Weather")
        
        if submitted:
            update_history(city, country)
            lat, lon = display_current_weather(city, country)
            display_forecast(city, lat, lon)

elif option == "Compare Two Cities":
    show_search_history()
    with st.form("compare_form"):
        col1, col2 = st.columns(2)
        with col1:
            city1 = st.text_input("City 1", "London")
            country1 = st.text_input("Country 1", "England")
        with col2:
            city2 = st.text_input("City 2", "New York")
            country2 = st.text_input("Country 2", "US")
        submitted = st.form_submit_button("Compare")
        
        if submitted:
            try:
                update_history(city1, country1)
                update_history(city2, country2)

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
                    st.subheader("Current Weather Comparison")
                    current1 = prepare_current_weather(data1["weather"], data1["aqi"])
                    current2 = prepare_current_weather(data2["weather"], data2["aqi"])
                    st.write(f"Current Date: {current1['Updated']}")
            
                    if current1 and current2:
                        # City 1
                        st.markdown(f"### {city1}")
                        # Main metrics
                        col1, col2, col3 = st.columns(3)
                        col1.markdown(f'<p style="font-size:24px; font-weight:1000; color:#1a73e8;">🌡️ {current1["Temperature"]}</p>', unsafe_allow_html=True)
                        col2.markdown(f'<p style="font-size:24px; font-weight:1000; color:#1a73e8;">💨 {current1["Feels Like"]}</p>', unsafe_allow_html=True)
                        col3.markdown(f'<p style="font-size:24px; font-weight:1000; color:#1a73e8;">🌫️ {current1["AQI"]}</p>', unsafe_allow_html=True)
                        
                        # Secondary metrics
                        col4, col5, col6, col7 = st.columns(4)
                        col4.markdown(f'<p style="font-size:20px; font-weight:600; margin:12px 0;">🌤️ {current1["Weather"]}</p>', unsafe_allow_html=True)
                        col4.markdown(f'<p style="font-size:20px; font-weight:600; margin:12px 0;">💧 {current1["Humidity"]}</p>', unsafe_allow_html=True)
                        col5.markdown(f'<p style="font-size:20px; font-weight:600; margin:12px 0;">🎐 {current1["Wind Speed"]}</p>', unsafe_allow_html=True)
                        col5.markdown(f'<p style="font-size:20px; font-weight:600; margin:12px 0;">🌧️ {current1["Rain (1h)"]}</p>', unsafe_allow_html=True)
                        col6.markdown(f'<p style="font-size:20px; font-weight:600; margin:12px 0;">🌅 {current1["Sunrise"]}</p>', unsafe_allow_html=True)
                        col6.markdown(f'<p style="font-size:20px; font-weight:600; margin:12px 0;">🌇 {current1["Sunset"]}</p>', unsafe_allow_html=True)
                        col7.markdown(f'<p style="font-size:20px; font-weight:600; margin:12px 0;">📊 {current1["Pressure"]}</p>', unsafe_allow_html=True)
                        
                        st.markdown("---")
                        
                        # City 2
                        st.markdown(f"### {city2}")
                        # Main metrics
                        col8, col9, col10 = st.columns(3)
                        col8.markdown(f'<p style="font-size:24px; font-weight:1000; color:#1a73e8;">🌡️ {current2["Temperature"]}</p>', unsafe_allow_html=True)
                        col9.markdown(f'<p style="font-size:24px; font-weight:1000; color:#1a73e8;">💨 {current2["Feels Like"]}</p>', unsafe_allow_html=True)
                        col10.markdown(f'<p style="font-size:24px; font-weight:1000; color:#1a73e8;">🌫️ {current2["AQI"]}</p>', unsafe_allow_html=True)
                        
                        # Secondary metrics
                        col11, col12, col13, col14 = st.columns(4)
                        col11.markdown(f'<p style="font-size:20px; font-weight:600; margin:12px 0;">🌤️ {current2["Weather"]}</p>', unsafe_allow_html=True)
                        col11.markdown(f'<p style="font-size:20px; font-weight:600; margin:12px 0;">💧 {current2["Humidity"]}</p>', unsafe_allow_html=True)
                        col12.markdown(f'<p style="font-size:20px; font-weight:600; margin:12px 0;">🎐 {current2["Wind Speed"]}</p>', unsafe_allow_html=True)
                        col12.markdown(f'<p style="font-size:20px; font-weight:600; margin:12px 0;">🌧️ {current2["Rain (1h)"]}</p>', unsafe_allow_html=True)
                        col13.markdown(f'<p style="font-size:20px; font-weight:600; margin:12px 0;">🌅 {current2["Sunrise"]}</p>', unsafe_allow_html=True)
                        col13.markdown(f'<p style="font-size:20px; font-weight:600; margin:12px 0;">🌇 {current2["Sunset"]}</p>', unsafe_allow_html=True)
                        col14.markdown(f'<p style="font-size:20px; font-weight:600; margin:12px 0;">📊 {current2["Pressure"]}</p>', unsafe_allow_html=True)
                                   
                 # Forecast comparison
                if data1["forecast"] and data2["forecast"]:
                    st.subheader("Forecast Comparison")
                    fig = plot_comparison(data1["forecast"], data2["forecast"], city1, city2)
                    if fig:
                        st.pyplot(fig)
                    
                    # Get formatted summaries
                    summary1 = prepare_forecast_summary(data1["forecast"], data1["aqi_forecast"], city1)
                    summary2 = prepare_forecast_summary(data2["forecast"], data2["aqi_forecast"], city2)
                    
                    if summary1 and summary2:
                        # Convert to DataFrames
                        df1 = pd.DataFrame(summary1)
                        df2 = pd.DataFrame(summary2)
                        
                        # Merge on Date column
                        combined = pd.merge(df1, df2, on="Date", how="outer")
                        
                        # Create logical column order
                        column_order = ["Date"]
                        metrics = ["Avg Temp", "Avg Humidity", "Total Rain", 
                                  "Avg Pressure", "Avg Wind", "AQI"]
                        
                        for metric in metrics:
                            column_order.extend([f"{metric} ({city1})", f"{metric} ({city2})"])
                        
                        # Display formatted comparison
                        st.subheader("Combined Forecast Summary")
                        st.dataframe(combined[column_order], use_container_width=True)

            except Exception as e:
                st.error(f"Comparison failed: {str(e)}")