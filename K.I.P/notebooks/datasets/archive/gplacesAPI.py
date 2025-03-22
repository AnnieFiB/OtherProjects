import requests
import pandas as pd
import os
from dotenv import load_dotenv
import time

# Load API key
load_dotenv()
API_KEY = os.getenv("GOOGLE_API_KEY")

# Ensure datasets directory exists
DATASET_DIR = "datasets"
os.makedirs(DATASET_DIR, exist_ok=True)

def get_coordinates(location):
    """
    Get latitude and longitude for a given location using the Geocoding API.
    
    Args:
        location (str): The location to geocode.
    
    Returns:
        tuple: (latitude, longitude) or None if geocoding fails.
    """
    geocode_url = "https://maps.googleapis.com/maps/api/geocode/json"
    geocode_params = {"address": location, "key": API_KEY}
    geocode_response = requests.get(geocode_url, params=geocode_params).json()
    
    if not geocode_response.get("results"):
        print(f"❌ Geocoding failed for {location}.")
        return None
    
    lat = geocode_response["results"][0]["geometry"]["location"]["lat"]
    lng = geocode_response["results"][0]["geometry"]["location"]["lng"]
    return lat, lng

def fetch_places(lat, lng, search_type, radius, max_results, location):
    """
    Fetch places using the Google Places API with pagination.
    
    Args:
        lat (float): Latitude of the search center.
        lng (float): Longitude of the search center.
        search_type (str): The type of place to search for (e.g., "beauty_salon").
        radius (int): Search radius in meters.
        max_results (int): Maximum number of results to fetch.
        location (str): The location being searched (for error messages).
    
    Returns:
        list: A list of places matching the search criteria.
    """
    places_url = "https://places.googleapis.com/v1/places:searchNearby"
    headers = {
        "Content-Type": "application/json",
        "X-Goog-Api-Key": API_KEY,
        "X-Goog-FieldMask": "places.displayName,places.formattedAddress,places.rating,places.id,places.location,places.businessStatus,places.internationalPhoneNumber,places.websiteUri,places.priceLevel,places.userRatingCount,places.currentOpeningHours,places.takeout,places.delivery,places.curbsidePickup,places.reviews,places.types"
    }

    places = []
    next_page_token = None
    total_fetched = 0

    while True:
        # Calculate the number of results to fetch in this iteration
        results_to_fetch = min(20, max_results - total_fetched)

        payload = {
            "includedTypes": [search_type],
            "maxResultCount": results_to_fetch,  # Fetch up to 20 results per request
            "locationRestriction": {
                "circle": {
                    "center": {"latitude": lat, "longitude": lng},
                    "radius": float(radius)
                }
            }
        }

        # Add next_page_token if available
        if next_page_token:
            payload["pageToken"] = next_page_token

        response = requests.post(places_url, headers=headers, json=payload)
        if response.status_code != 200:
            print(f"❌ API Error for {search_type} in {location}: {response.text}")
            break

        data = response.json()
        places.extend(data.get("places", []))

        total_fetched += len(data.get("places", []))
        if total_fetched >= max_results or "nextPageToken" not in data:
            break

        # Wait for a short time before making the next request (required by the API)
        next_page_token = data["nextPageToken"]
        time.sleep(2)  # Wait 2 seconds to avoid rate limiting

    return places

def get_beauty_stores(location, search_type="beauty_salon", radius=5000, max_results=100):
    """
    Fetch beauty stores dynamically based on user input with pagination.
    
    Args:
        location (str): The location to search.
        search_type (str): The type of place to search for.
        radius (int): Search radius in meters.
        max_results (int): Maximum number of results to fetch.
    """
    # Step 1: Get coordinates using Geocoding API
    coordinates = get_coordinates(location)
    if not coordinates:
        return
    
    lat, lng = coordinates

    # Step 2: Fetch places using the Places API
    places = fetch_places(lat, lng, search_type, radius, max_results, location)

    # Step 3: Filter and format results
    stores = []
    for place in places:
        # Filter out businesses with ratings below 4.0
        if place.get("rating", 0) < 4.0:
            continue

        stores.append({
            "Category": search_type,
            "Location": location,
            "Name": place.get("displayName", {}).get("text", "N/A"),
            "Business Status": place.get("businessStatus", "N/A"),
            "Formatted Address": place.get("formattedAddress", "N/A"),
            "Place ID": place.get("id", "N/A"),
            "Phone Number": place.get("formattedPhoneNumber", "N/A"),
            "International Phone Number": place.get("internationalPhoneNumber", "N/A"),
            "Opening Hours": place.get("currentOpeningHours", {}).get("weekdayDescriptions", ["N/A"]),
            "Website": place.get("websiteUri", "N/A"),
            "Delivery": place.get("delivery", "N/A"),
            "Curbside Pickup": place.get("curbsidePickup", "N/A"),
            "Price Level": place.get("priceLevel", "N/A"),
            "Rating": place.get("rating", "N/A"),
            "Reviews": place.get("reviews", []),
            "User Ratings Total": place.get("userRatingCount", "N/A")
        })

    # Step 4: Save results to CSV
    if stores:
        file_path = os.path.join(DATASET_DIR, "beauty_stores_dynamic.csv")
        df = pd.DataFrame(stores)
        if not os.path.exists(file_path):
            df.to_csv(file_path, index=False)
        else:
            df.to_csv(file_path, index=False, mode='a', header=False)
        print(f"✅ Successfully saved {len(stores)} records to {file_path}.")
    else:
        print("⚠️ No stores found for the given location and search type.")

def get_initial_user_input():
    """
    Get initial search parameters from user input.
    
    Returns:
        tuple: A tuple containing location, radius, and max_results.
    """
    location = input("Enter the location (e.g., 'Mexico City, Mexico'): ")
    radius = int(input("Enter the search radius in meters (e.g., 5000): "))
    max_results = int(input("Enter the maximum number of results (e.g., 100): "))
    return location, radius, max_results

def get_search_type():
    """
    Get search type for subsequent searches.
    
    Returns:
        str: The search type (e.g., "beauty_salon,spa,pharmacy,department_store,store").
    """
    return input("Enter the search type (e.g., 'beauty_salon,spa,pharmacy,department_store,store'): ")

def main():
    """
    Main function to execute the Google Places search and save results to a CSV file.
    """
    # Get initial parameters
    location, radius, max_results = get_initial_user_input()

    while True:
        print("\n=== New Search ===")
        search_type = get_search_type()
        get_beauty_stores(location, search_type, radius, max_results)

        # Ask the user if they want to perform another search
        another_search = input("Do you want to perform another search? (y/n): ").strip().lower()
        if another_search != "y":
            print("Exiting the program. Goodbye!")
            break
