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

    # Step 2: Use Places API (New) Nearby Search
    places_url = "https://places.googleapis.com/v1/places:searchNearby"
    headers = {
        "Content-Type": "application/json",
        "X-Goog-Api-Key": API_KEY,
        "X-Goog-FieldMask": "places.displayName,places.formattedAddress,places.rating,places.id,places.location"
    }
    payload = {
        "includedTypes": ["department_store"],
        "maxResultCount": 20,
        "locationRestriction": {
            "circle": {
                "center": {"latitude": lat, "longitude": lng},
                "radius": 5000.0
            }
        }
    }

    response = requests.post(places_url, headers=headers, json=payload)
    if response.status_code != 200:
        print(f"❌ API Error: {response.text}")
        return

    stores = []
    for place in response.json().get("places", []):
        stores.append({
            "Name": place.get("displayName", {}).get("text", "N/A"),
            "Address": place.get("formattedAddress", "N/A"),
            "Latitude": place.get("location", {}).get("latitude", "N/A"),
            "Longitude": place.get("location", {}).get("longitude", "N/A"),
            "rating": place.get("rating"),
            "user_ratings_total": place.get("userRatingCount"),
                         # Contact info
            "phone": place.get("internationalPhoneNumber"),
            "website": place.get("websiteUri"),
                        # Business details
            "price_level": place.get("priceLevel"),
            "Place ID": place.get("id", "N/A")
        })

    if stores:
        pd.DataFrame(stores).to_csv("beauty_stores_new_api.csv", index=False)
        print(f"✅ Saved {len(stores)} stores to CSV.")
    else:
        print("⚠️ No stores found.")

# Run the function
get_beauty_stores()
