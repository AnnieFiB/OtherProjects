import requests
import pandas as pd
import re
from bs4 import BeautifulSoup
import json

GOOGLE_API_KEY = GOOGLE_API_KEY # Replace with valid API key
SEARCH_QUERY = "beauty store in Mexico City"

def get_place_details(place_id):
    """Fetch detailed information using Place Details API"""
    details_url = "https://maps.googleapis.com/maps/api/place/details/json"
    params = {
        "place_id": place_id,
        "key": GOOGLE_API_KEY,
        "fields": "name,formatted_address,rating,formatted_phone_number,website",
        "language": "en"
    }
    
    try:
        response = requests.get(details_url, params=params)
        response.raise_for_status()
        return response.json().get('result', {})
    except Exception as e:
        print(f"Details API Error: {str(e)}")
        return {}

def get_google_places_data():
    """Fetch basic store data from Google Places API"""
    places_url = "https://maps.googleapis.com/maps/api/place/textsearch/json"
    params = {
        "query": SEARCH_QUERY,
        "key": GOOGLE_API_KEY,
        "region": "mx"
    }
    
    try:
        response = requests.get(places_url, params=params)
        response.raise_for_status()
        data = response.json()
        
        if data['status'] != 'OK':
            print(f"API Error: {data.get('error_message', 'Unknown error')}")
            return []
            
        print(f"Found {len(data['results'])} initial results")
        return data['results']
    except Exception as e:
        print(f"Search API Error: {str(e)}")
        return []

def extract_website_data(url):
    """Scrape website for brand and email information"""
    try:
        response = requests.get(url, timeout=10, headers={
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'
        })
        if response.status_code != 200:
            return {}, []

        soup = BeautifulSoup(response.text, 'html.parser')
        
        # Brand extraction
        brand_selectors = [
            {'class': 'brand'},
            {'itemprop': 'brand'},
            {'class': 'marcas'},
            {'class': 'brands'}
        ]
        brands = []
        for selector in brand_selectors:
            brands.extend([b.get_text(strip=True) for b in soup.find_all(attrs=selector)])
            if brands: break
        
        # Email extraction
        emails = re.findall(r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b', response.text)
        
        return {
            'brands': list(set(brands))[:5],  # Max 5 unique brands
            'emails': list(set(emails))[:3]    # Max 3 unique emails
        }
        
    except Exception as e:
        print(f"Website Error ({url}): {str(e)}")
        return {}, []

def main():
    stores = get_google_places_data()
    if not stores:
        print("No stores found. Check API key and network connection.")
        return

    results = []
    
    for idx, place in enumerate(stores[:5]):  # Process first 5 for testing
        print(f"\nProcessing store {idx+1}/{len(stores)}: {place.get('name')}")
        
        # Get detailed information
        details = get_place_details(place['place_id'])
        if not details:
            print("Skipping store - no details available")
            continue
            
        website = details.get('website')
        extra_data = {}
        
        if website:
            print(f"Scraping website: {website}")
            extra_data = extract_website_data(website)
        
        store_record = {
            'name': details.get('name'),
            'address': details.get('formatted_address'),
            'rating': details.get('rating'),
            'phone': details.get('formatted_phone_number'),
            'website': website,
            **extra_data
        }
        
        results.append(store_record)
        print(f"Collected data: {store_record}")

    if results:
        df = pd.DataFrame(results)
        df.to_csv('beauty_stores_data.csv', index=False)
        print(f"\n✅ Successfully saved {len(df)} records to beauty_stores_data.csv")
    else:
        print("\n❌ No data collected. Check error messages above.")

if __name__ == "__main__":
    main()