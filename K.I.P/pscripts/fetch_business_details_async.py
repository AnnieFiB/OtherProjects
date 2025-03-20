import aiohttp
import asyncio
import pandas as pd
import os
from dotenv import load_dotenv
import re

# Load API key
load_dotenv()
API_KEY = os.getenv("GOOGLE_API_KEY")

# Ensure datasets directory exists
DATASET_DIR = "datasets"
os.makedirs(DATASET_DIR, exist_ok=True)

def normalize_text(text):
    """
    Normalize text by removing special characters and extra spaces.
    """
    if pd.isna(text):
        return ""
    text = re.sub(r"[^a-zA-Z0-9\s]", "", text)  # Remove special characters
    text = re.sub(r"\s+", " ", text)  # Replace multiple spaces with a single space
    return text.strip()

async def get_business_details(session, business_name, latitude, longitude, retries=3):
    """
    Fetch business details using the Google Places API asynchronously.
    
    Args:
        session (aiohttp.ClientSession): The aiohttp session.
        business_name (str): The name of the business.
        latitude (float): Latitude of the search center.
        longitude (float): Longitude of the search center.
        retries (int): Number of retries for failed requests.
    
    Returns:
        dict: Business details or None if not found.
    """
    # Normalize business name
    business_name = normalize_text(business_name)

    for attempt in range(retries):
        try:
            # Step 1: Use Places API Text Search to find the business
            places_url = "https://maps.googleapis.com/maps/api/place/textsearch/json"
            params = {
                "query": business_name,  # Search by business name
                "location": f"{latitude},{longitude}",  # Use latitude and longitude
                "radius": 5000,  # Search within a 5000-meter radius
                "key": API_KEY
            }

            async with session.get(places_url, params=params) as response:
                response.raise_for_status()  # Raise an error for bad status codes
                data = await response.json()

                if not data.get("results"):
                    print(f"❌ No results found for business: {business_name} near ({latitude}, {longitude}) (Attempt {attempt + 1})")
                    continue  # Retry with a modified query

                # Step 2: Fetch detailed information for the first result
                place_id = data["results"][0]["place_id"]
                details_url = "https://maps.googleapis.com/maps/api/place/details/json"
                details_params = {
                    "place_id": place_id,
                    "fields": "name,formatted_address,rating,user_ratings_total,formatted_phone_number,website,opening_hours,types",
                    "key": API_KEY
                }

                async with session.get(details_url, params=details_params) as details_response:
                    details_response.raise_for_status()
                    details_data = (await details_response.json()).get("result", {})

                    # Step 3: Extract relevant details
                    return {
                        "Name": details_data.get("name", "N/A"),
                        "Address": details_data.get("formatted_address", "N/A"),
                        "Rating": details_data.get("rating", "N/A"),
                        "User Ratings Total": details_data.get("user_ratings_total", "N/A"),
                        "Phone Number": details_data.get("formatted_phone_number", "N/A"),
                        "Website": details_data.get("website", "N/A"),
                        "Opening Hours": details_data.get("opening_hours", {}).get("weekday_text", ["N/A"]),
                        "Types": ", ".join(details_data.get("types", [])),
                        "Latitude": latitude,
                        "Longitude": longitude
                    }

        except Exception as e:
            print(f"❌ Error fetching details for {business_name} near ({latitude}, {longitude}): {e}")
            if attempt == retries - 1:  # Log failure after final attempt
                print(f"❌ Failed to fetch details for {business_name} near ({latitude}, {longitude}) after {retries} attempts.")
            continue

    return None

async def fetch_businesses_from_df(df):
    """
    Fetch details for businesses listed in a DataFrame asynchronously.
    
    Args:
        df (pd.DataFrame): DataFrame containing columns 'Name', 'Latitude', and 'Longitude'.
    
    Returns:
        tuple: A tuple containing:
            - A list of successfully fetched business details.
            - A list of business names that failed to fetch.
    """
    async with aiohttp.ClientSession() as session:
        tasks = []
        for index, row in df.iterrows():
            business_name = row["Name"]
            latitude = row["Latitude"]
            longitude = row["Longitude"]
            print(f"Fetching details for: {business_name} near ({latitude}, {longitude})")
            tasks.append(get_business_details(session, business_name, latitude, longitude))

        # Run all tasks concurrently
        results = await asyncio.gather(*tasks)

    # Separate successful and failed results
    successful_results = [result for result in results if result is not None]
    failed_businesses = [df.iloc[i]["Name"] for i, result in enumerate(results) if result is None]

    return successful_results, failed_businesses

def save_to_csv(data, file_path):
    """
    Save business data to a CSV file.
    
    Args:
        data (list): List of business data dictionaries.
        file_path (str): Path to the CSV file.
    
    Returns:
        bool: True if the file was saved successfully, False otherwise.
    """
    try:
        df = pd.DataFrame(data)
        if not os.path.exists(file_path):
            df.to_csv(file_path, index=False)
        else:
            df.to_csv(file_path, mode='a', header=False, index=False)
        return True
    except Exception as e:
        print(f"❌ Error saving to CSV: {e}")
        return False

async def main(df):
    """
    Main function to execute the business details fetching process.
    
    Args:
        df (pd.DataFrame): DataFrame containing 'Name', 'Latitude', and 'Longitude' columns.
    """
    # Step 1: Fetch details for each business
    successful_results, failed_businesses = await fetch_businesses_from_df(df)

    # Step 2: Print totals
    print(f"✅ Total successful fetches: {len(successful_results)}")
    print(f"❌ Total unsuccessful fetches: {len(failed_businesses)}")

    # Step 3: Save the results to a CSV file
    if successful_results:
        output_file = os.path.join(DATASET_DIR, "business_details.csv")
        if save_to_csv(successful_results, output_file):
            print(f"✅ Successfully saved {len(successful_results)} records to {output_file}.")
        else:
            print("⚠️ Failed to save records to CSV.")
    else:
        print("⚠️ No business details were fetched.")

