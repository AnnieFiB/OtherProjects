import requests
import pandas as pd
import os
from dotenv import load_dotenv
import time

load_dotenv()
API_KEY = os.getenv("GOOGLE_API_KEY")
DATASET_DIR = "datasets"
os.makedirs(DATASET_DIR, exist_ok=True)
OUTPUT_FILE = os.path.join(DATASET_DIR, "places_results.csv")

VALID_TYPES = [
    'beauty_salon', 'spa', 'pharmacy', 'department_store', 
    'store', 'cosmetic_store', 'shopping_mall', 'hair_care'
]

def get_valid_search_types():
    print("\n✅ Valid Google Place Types:")
    print(", ".join(VALID_TYPES))
    while True:
        types = input("Enter comma-separated types: ").strip().split(',')
        cleaned = [t.strip() for t in types if t.strip() in VALID_TYPES]
        if cleaned:
            return cleaned
        print("❌ Invalid types. Try again.")

def fetch_places(lat, lng, search_type, radius=5000, max_results=100):
    url = "https://places.googleapis.com/v1/places:searchNearby"
    headers = {
        "Content-Type": "application/json",
        "X-Goog-Api-Key": API_KEY,
        "X-Goog-FieldMask": "places.id,places.displayName,places.formattedAddress,"
        "places.internationalPhoneNumber,places.nationalPhoneNumber,places.rating,"
        "places.userRatingCount,places.types,places.websiteUri,places.location,"
        "places.regularOpeningHours,places.currentOpeningHours,places.utcOffsetMinutes,"
        "places.priceLevel,places.businessStatus,places.shortFormattedAddress,"
        "places.reviews,"
        "places.accessibilityOptions,places.paymentOptions,"
        "places.delivery"
    }

    places = []
    next_page_token = None
    attempts = 0
    
    while True:
        if max_results > 0 and len(places) >= max_results:
            break
            
        try:
            payload = {
                "includedTypes": [search_type],
                "maxResultCount": 20,
                "locationRestriction": {
                    "circle": {
                        "center": {"latitude": lat, "longitude": lng},
                        "radius": float(radius)
                    }
                }
            }

            if next_page_token:
                payload["pageToken"] = next_page_token
                time.sleep(2)

            response = requests.post(url, headers=headers, json=payload)
            response.raise_for_status()
            
            data = response.json()
            new_places = data.get('places', [])
            
            if max_results > 0:
                remaining = max_results - len(places)
                new_places = new_places[:remaining]
                
            places.extend(new_places)
            next_page_token = data.get('nextPageToken')
            attempts = 0

            print(f"✓ {search_type}: Found {len(new_places)} (Total: {len(places)})")

            if not next_page_token:
                break

        except requests.exceptions.HTTPError as e:
            print(f"⚠️ API Error: {e.response.text[:200]}...")
            attempts += 1
            if attempts >= 3:
                break
            time.sleep(2)
        except Exception as e:
            print(f"🚨 Unexpected error: {str(e)}")
            break
            
    return places

def main():
    location = input("📍 Enter location (e.g., 'Mexico City, Mexico'): ").strip()
    search_types = get_valid_search_types()
    radius = int(input("📏 Search radius meters (max 50000): ") or 5000)
    max_results = int(input("🎯 Max results per type (0 for all): ") or 0)

    coords = get_coordinates(location)
    if not coords:
        return

    all_places = []
    for st in search_types:
        print(f"\n🔍 Searching for: {st}")
        places = fetch_places(*coords, st, radius, max_results)
        all_places.extend(places)
        print(f"✅ Finished {st}: {len(places)} results")

    if all_places:
        df = pd.json_normalize(all_places, sep='_')
        df.to_csv(OUTPUT_FILE, mode='a' if os.path.exists(OUTPUT_FILE) else 'w', 
                 header=not os.path.exists(OUTPUT_FILE), index=False)
        print(f"\n💾 Saved {len(df)} total records to {OUTPUT_FILE}")
    else:
        print("\n⚠️ No results found")

def get_coordinates(location):
    try:
        response = requests.get(
            "https://maps.googleapis.com/maps/api/geocode/json",
            params={"address": location, "key": API_KEY}
        )
        data = response.json()
        
        if data['status'] != 'OK':
            print(f"❌ Geocoding failed: {data.get('error_message', 'Unknown error')}")
            return None
            
        loc = data['results'][0]['geometry']['location']
        print(f"🌍 Coordinates: {loc['lat']:.6f}, {loc['lng']:.6f}")
        return loc['lat'], loc['lng']
        
    except Exception as e:
        print(f"🚨 Geocoding error: {str(e)}")
        return None
