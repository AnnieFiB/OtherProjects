import requests
import os
import pandas as pd
from dotenv import load_dotenv
from bs4 import BeautifulSoup
import time

# Load environment variables from .env file
load_dotenv()

def search_yelp_businesses(latitude, longitude, term, radius, limit, offset=0):
    """
    Search for businesses on Yelp using latitude, longitude, and search parameters.
    
    Args:
        latitude (float): Latitude of the search center.
        longitude (float): Longitude of the search center.
        term (str): The search term (e.g., "beauty salon").
        radius (int): Search radius in meters (max 40000).
        limit (int): Maximum number of results to return (max 50).
        offset (int): Offset for pagination.
    
    Returns:
        list: A list of businesses matching the search criteria.
    """
    # Load Yelp API key from .env
    YELP_API_KEY = os.getenv("YELP_API_KEY")
    if not YELP_API_KEY:
        print("Yelp API key not found. Please check your .env file.")
        return None

    # Yelp Fusion API endpoint
    url = "https://api.yelp.com/v3/businesses/search"
    headers = {
        "Authorization": f"Bearer {YELP_API_KEY}"
    }
    params = {
        "latitude": latitude,
        "longitude": longitude,
        "term": term,
        "radius": radius,
        "limit": limit,
        "offset": offset
    }

    try:
        response = requests.get(url, headers=headers, params=params, timeout=10)
        response.raise_for_status()  # Raise an error for bad status codes
        return response.json().get("businesses", [])
    except requests.exceptions.RequestException as e:
        print(f"Error fetching data from Yelp API: {e}")
        return None

def get_yelp_business_details(business_id):
    """
    Fetch detailed business information from Yelp using the Yelp Fusion API.
    
    Args:
        business_id (str): The Yelp business ID.
    
    Returns:
        dict: Detailed business information.
    """
    # Load Yelp API key from .env
    YELP_API_KEY = os.getenv("YELP_API_KEY")
    if not YELP_API_KEY:
        print("Yelp API key not found. Please check your .env file.")
        return None

    url = f"https://api.yelp.com/v3/businesses/{business_id}"
    headers = {
        "Authorization": f"Bearer {YELP_API_KEY}"
    }

    try:
        response = requests.get(url, headers=headers, timeout=10)
        response.raise_for_status()  # Raise an error for bad status codes
        return response.json()
    except requests.exceptions.RequestException as e:
        print(f"Error fetching business details for ID {business_id}: {e}")
        return None

def scrape_brands_and_quantities(url):
    """
    Scrape brand and quantity data from a business website.
    
    Args:
        url (str): The URL of the business website.
    
    Returns:
        dict: A dictionary containing brands and quantities (if found).
    """
    try:
        # Send a GET request to the website
        response = requests.get(url, timeout=10)
        response.raise_for_status()  # Raise an error for bad status codes

        # Parse the HTML content
        soup = BeautifulSoup(response.text, "html.parser")

        # Example: Look for brands and quantities in specific HTML elements
        # Adjust the selectors based on the website's structure
        brands = []
        quantities = []

        # Example: Scrape brands (adjust the selector as needed)
        brand_elements = soup.select(".brand-name")  # Replace with the correct CSS selector
        for element in brand_elements:
            brands.append(element.text.strip())

        # Example: Scrape quantities (adjust the selector as needed)
        quantity_elements = soup.select(".quantity")  # Replace with the correct CSS selector
        for element in quantity_elements:
            quantities.append(element.text.strip())

        # Return the scraped data
        return {
            "brands": ", ".join(brands) if brands else "N/A",
            "quantities": ", ".join(quantities) if quantities else "N/A"
        }

    except requests.exceptions.RequestException as e:
        print(f"Error scraping {url}: {e}")
        return {
            "brands": "N/A",
            "quantities": "N/A"
        }

def get_initial_user_input():
    """
    Get initial search parameters from user input.
    
    Returns:
        tuple: A tuple containing latitude, longitude, radius, limit, and total_results.
    """
    use_coordinates = input("Do you want to use latitude and longitude? (y/n): ").strip().lower()
    if use_coordinates == "y":
        latitude = float(input("Enter the latitude (e.g., 19.4326): "))
        longitude = float(input("Enter the longitude (e.g., -99.1332): "))
    else:
        location = input("Enter the location (e.g., 'Mexico City, Mexico'): ")
        # Geocode the location to get latitude and longitude
        geocode_url = "https://maps.googleapis.com/maps/api/geocode/json"
        geocode_params = {"address": location, "key": os.getenv("GOOGLE_API_KEY")}
        geocode_response = requests.get(geocode_url, params=geocode_params).json()
        if not geocode_response.get("results"):
            print(f"❌ Geocoding failed for {location}.")
            return None, None, None, None, None
        latitude = geocode_response["results"][0]["geometry"]["location"]["lat"]
        longitude = geocode_response["results"][0]["geometry"]["location"]["lng"]

    radius = int(input("Enter the search radius in meters (e.g., 5000): "))
    limit = int(input("Enter the maximum number of results per page (1-50): "))
    total_results = int(input("Enter the total number of results to fetch: "))
    return latitude, longitude, radius, limit, total_results

def get_search_term():
    """
    Get the search term for subsequent searches.
    
    Returns:
        str: The search term (e.g., "beauty supply, spa, department store, pharmacy, boutique").
    """
    return input("Enter the search type (e.g., 'beauty supply, spa, department store, pharmacy, boutique'): ")

def save_to_csv(data, file_path):
    """
    Save business data to a CSV file.
    
    Args:
        data (list): List of business data dictionaries.
        file_path (str): Path to the CSV file.
    """
    df = pd.DataFrame(data)
    if not os.path.exists(file_path):
        df.to_csv(file_path, index=False)
    else:
        df.to_csv(file_path, mode='a', header=False, index=False)

def main():
    """
    Main function to execute the Yelp search and save results to a CSV file.
    """
    # Step 1: Get initial user input
    latitude, longitude, radius, limit, total_results = get_initial_user_input()
    if latitude is None or longitude is None:
        return

    # Ensure datasets directory exists
    DATASET_DIR = "datasets"
    os.makedirs(DATASET_DIR, exist_ok=True)
    file_path = os.path.join(DATASET_DIR, "yelp_businesses.csv")

    while True:
        # Step 2: Print "=== New Search ==="
        print("\n=== New Search ===")

        # Step 3: Get search term for the current search
        term = get_search_term()

        # Step 4: Initialize variables for pagination
        businesses = []
        offset = 0

        # Step 5: Fetch results with pagination
        while len(businesses) < total_results:
            results = search_yelp_businesses(latitude, longitude, term, radius, limit, offset)

            if not results:
                print("No more results found.")
                break

            # Step 6: Fetch detailed information for each business
            for business in results:
                # Filter businesses with ratings of 4.0 or above
                if business.get("rating", 0) < 4.0:
                    continue

                business_id = business["id"]
                business_details = get_yelp_business_details(business_id)

                if business_details:
                    # Extract the business website from the details
                    website_url = business_details.get("website", "N/A")

                    # Scrape brand and quantity data from the business website
                    if website_url != "N/A":
                        scraped_data = scrape_brands_and_quantities(website_url)
                    else:
                        scraped_data = {"brands": "N/A", "quantities": "N/A"}

                    businesses.append({
                        "Business ID": business_id,
                        "Name": business_details.get("name", "N/A"),
                        "Rating": business_details.get("rating", "N/A"),
                        "Review Count": business_details.get("review_count", "N/A"),
                        "Address": business_details.get("location", {}).get("address1", "N/A"),
                        "City": business_details.get("location", {}).get("city", "N/A"),
                        "State": business_details.get("location", {}).get("state", "N/A"),
                        "Zip Code": business_details.get("location", {}).get("zip_code", "N/A"),
                        "Phone": business_details.get("phone", "N/A"),
                        "Website": website_url,
                        "Categories": ", ".join([cat["title"] for cat in business_details.get("categories", [])]),
                        "Brands": scraped_data["brands"],
                        "Quantities": scraped_data["quantities"],
                        "Latitude": latitude,
                        "Longitude": longitude
                    })

            # Step 7: Update offset for pagination
            offset += limit

            # Step 8: Save results to CSV
            save_to_csv(businesses, file_path)

            # Step 9: Add a delay to avoid hitting rate limits
            time.sleep(1)

            # Step 10: Stop if we've reached the total_results limit
            if len(businesses) >= total_results:
                break

        # Step 11: Print success message
        print(f"✅ Successfully saved {len(businesses)} records to {file_path}.")

        # Step 12: Ask if the user wants to perform another search
        another_search = input("Do you want to perform another search? (y/n): ").strip().lower()
        if another_search != "y":
            print("Exiting the program. Goodbye!")
            break
