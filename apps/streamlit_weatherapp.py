# ===== streamlit_weatherapp.py =====
import streamlit as st
from weather import (
    get_city_coordinates, fetch_weather_data, fetch_forecast_data,
    fetch_air_pollution_data, prepare_weather_table,
    prepare_forecast_chart_data, create_forecast_figure
)

# Initialize session state for search history
if 'search_history' not in st.session_state:
    st.session_state.search_history = []

# Configure page
st.title("Weather Explorer")
st.markdown("Explore weather data and compare forecasts interactively!")

# Search history sidebar
with st.sidebar:
    st.subheader("Search History")
    if st.session_state.search_history:
        selected_history = st.selectbox(
            "Recent searches:",
            reversed(st.session_state.search_history),
            format_func=lambda x: x['query']
        )
        if selected_history:
            st.experimental_set_query_params(search=selected_history['query'])

# Main app functionality
option = st.radio(
    "Choose an action:",
    ("View City Forecast", "Compare Cities")
)

@st.cache_data(show_spinner=False, max_entries=5)
def cached_weather_call(lat, lon):
    return (
        fetch_weather_data(lat, lon),
        fetch_forecast_data(lat, lon),
        fetch_air_pollution_data(lat, lon)
    )

def update_search_history(query):
    # Keep only unique entries and limit to 5
    history = [item for item in st.session_state.search_history if item['query'] != query]
    history.insert(0, {'query': query, 'timestamp': datetime.now()})
    st.session_state.search_history = history[:5]

def display_city_forecast():
    query_params = st.experimental_get_query_params()
    default_city = query_params.get("search", [""])[0]
    
    city = st.text_input(
        "Enter location (City, Country/State):",
        value=default_city,
        help="Format: 'City' or 'City, Country' or 'City, State, Country'"
    )
    
    if city:
        try:
            lat, lon = get_city_coordinates(city)
            st.success(f"Coordinates found: {lat:.2f}, {lon:.2f}")
            
            # Update search history
            update_search_history(city)
            
            # Fetch data with caching
            weather_data, forecast_data, pollution_data = cached_weather_call(lat, lon)
            
            # Display current conditions
            weather_table = prepare_weather_table(weather_data, pollution_data)
            st.subheader("Current Conditions")
            st.table(weather_table)
            
            # Display forecast
            chart_data = prepare_forecast_chart_data(forecast_data)
            fig = create_forecast_figure(chart_data, city)
            st.subheader("5-Day Forecast")
            st.pyplot(fig)
            
        except Exception as e:
            st.error(f"Error: {str(e)}")

def display_city_comparison():
    col1, col2 = st.columns(2)
    with col1:
        city1 = st.text_input("City 1:", help="Format: City, Country/State")
    with col2:
        city2 = st.text_input("City 2:", help="Format: City, Country/State")
    
    if city1 and city2:
        try:
            lat1, lon1 = get_city_coordinates(city1)
            lat2, lon2 = get_city_coordinates(city2)
            
            # Update search history for both cities
            update_search_history(city1)
            update_search_history(city2)
            
            # Fetch data with caching
            weather1, forecast1, _ = cached_weather_call(lat1, lon1)
            weather2, forecast2, _ = cached_weather_call(lat2, lon2)
            
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

if option == "View City Forecast":
    display_city_forecast()
elif option == "Compare Cities":
    display_city_comparison()