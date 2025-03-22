from yelpapi import YelpAPI
import pandas as pd
import time
import os
from dotenv import load_dotenv

load_dotenv()

# Initialize Yelp API
yelp_api = YelpAPI(os.getenv('YELP_API_KEY'))  # Ensure key is in .env file

# User Configuration
LOCATION = input("Enter location (e.g., 'Mexico City, MX'): ").strip()
SEARCH_TERMS = [term.strip() for term in input("Enter search terms (comma-separated): ").split(',')]
RECORDS_PER_TERM = int(input("Records to fetch per term (max 1000): ") or 200)
OUTPUT_FILE = 'datasets/yelp_combined_results.csv'

seen_ids = set()

# Load existing data
if os.path.exists(OUTPUT_FILE):
    existing_df = pd.read_csv(OUTPUT_FILE)
    seen_ids.update(existing_df['id'].tolist())
    print(f"Loaded {len(existing_df)} existing records")

all_businesses = []

for term in SEARCH_TERMS:
    print(f"\n=== Searching for: {term} ===")
    
    for offset in range(0, RECORDS_PER_TERM, 50):
        try:
            # API call with parameter validation
            response = yelp_api.search_query(
                term=term,
                location=LOCATION,
                limit=50,
                offset=offset,
                sort_by='best_match'  # Required for some parameter combinations
            )
            
            new_businesses = [biz for biz in response.get('businesses', []) if biz['id'] not in seen_ids]
            
            if not new_businesses:
                print(f"No new results at offset {offset}")
                break
                
            all_businesses.extend(new_businesses)
            seen_ids.update(biz['id'] for biz in new_businesses)
            print(f"✓ Added {len(new_businesses)} records (Total: {len(all_businesses)})")
            time.sleep(1)  # Increased delay for rate limits
            
        except Exception as e:
            print(f"⚠️ Failed to fetch '{term}' at offset {offset}: {str(e)}")
            if "400 Client Error" in str(e):
                print("Tip: Try different search terms or verify location format")
            break  # Continue to next search term

# Save results
if all_businesses:
    df = pd.json_normalize(all_businesses, sep='_')
    df.to_csv(
        OUTPUT_FILE,
        mode='a' if os.path.exists(OUTPUT_FILE) else 'w',
        header=not os.path.exists(OUTPUT_FILE),
        index=False
    )
    print(f"\n✅ Saved {len(df)} new records to {OUTPUT_FILE}")
    print(f"Total unique records: {len(pd.read_csv(OUTPUT_FILE)) if os.path.exists(OUTPUT_FILE) else 0}")
else:
    print("\n⛔ No new records added")