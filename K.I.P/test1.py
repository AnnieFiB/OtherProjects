import requests
import pandas as pd
import time

API_KEY = GOOGLE_API_KEY  # Use your key

def get_beauty_stores(location="Mexico City"):
    # Step 1: Get coordinates using Geocoding API
    geocode_url = "https://maps.googleapis.com/maps/api/geocode/json"
    geocode_params = {"address": location, "key": API_KEY}
    geocode_response = requests.get(geocode_url, params=geocode_params).json()
    
    if not geocode_response.get("results"):
        print("❌ Geocoding failed.")
        return
    
    lat = geocode_response["results"][0]["geometry"]["location"]["lat"]
    lng = geocode_response["results"][0]["geometry"]["location"]["lng"]

    # Step 2: Configure API request parameters
    places_url = "https://places.googleapis.com/v1/places:searchNearby"
    headers = {
        "Content-Type": "application/json",
        "X-Goog-Api-Key": API_KEY,
        "X-Goog-FieldMask": "places.displayName,places.formattedAddress,places.rating,places.id,places.location,nextPageToken"
    }
    
    payload = {
        "includedTypes": ["beauty_salon"],
        "maxResultCount": 20,
        "locationRestriction": {
            "circle": {
                "center": {"latitude": lat, "longitude": lng},
                "radius": 10000.0  # Increased radius to 10km
            }
        }
    }

    stores = []
    page_count = 0
    max_pages = 3  # Google's maximum allowed pages
    
    while page_count < max_pages:
        response = requests.post(places_url, headers=headers, json=payload)
        
        if response.status_code != 200:
            print(f"❌ API Error: {response.text}")
            break

        data = response.json()
        page_count += 1
        
        # Process current page results
        for place in data.get("places", []):
            stores.append({
                "Name": place.get("displayName", {}).get("text", "N/A"),
                "Address": place.get("formattedAddress", "N/A"),
                "Latitude": place.get("location", {}).get("latitude", "N/A"),
                "Longitude": place.get("location", {}).get("longitude", "N/A"),
                "Place ID": place.get("id", "N/A")
            })

        # Check for next page token
        next_token = data.get("nextPageToken")
        if not next_token:
            break
            
        # Set up next page request
        payload["pageToken"] = next_token
        time.sleep(2)  # Required delay between page requests

    # Save results
    if stores:
        pd.DataFrame(stores).to_csv("beauty_stores_new_api.csv", index=False)
        print(f"✅ Saved {len(stores)} stores to CSV.")
    else:
        print("⚠️ No stores found.")
