import requests
import pandas as pd
import json

# Your Google API Key
API_KEY = GOOGLE_API_KEY

# Search query for beauty stores in Mexico City
SEARCH_QUERY = "beauty store in Mexico City"

# Google Places API URL
PLACES_URL = "https://maps.googleapis.com/maps/api/place/textsearch/json"

# ======queries Google Places API for beauty stores in Mexico City

# Function to fetch stores
def get_beauty_stores():
    params = {
        "query": SEARCH_QUERY,
        "key": API_KEY
    }
    
    response = requests.get(PLACES_URL, params=params)
    data = response.json()

    if "results" not in data:
        print("Error fetching data:", data.get("error_message", "Unknown error"))
        return None

    stores = []
    for place in data["results"]:
        store = {
            "Name": place.get("name", "N/A"),
            "Location": place.get("formatted_address", "N/A"),
            "Rating": place.get("rating", "N/A"),
            "Place ID": place.get("place_id", "N/A"),
            "Phone": "Needs Scraping",
            "Brands Sold": "Needs Scraping",
            "Email": "Needs Scraping"
        }
        stores.append(store)

    df = pd.DataFrame(stores)
    df.to_csv("beauty_stores_google.csv", index=False)
    print("✅ Google Places data saved as 'beauty_stores_google.csv'")
    return df

# Fetch beauty stores
beauty_stores = get_beauty_stores()

# =================Scrape Additional Details (Emails, Brands Sold)

import time
import pandas as pd
from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.chrome.service import Service
from selenium.webdriver.chrome.options import Options

# Set up Brave browser for Selenium
CHROMEDRIVER_PATH = "C:/chromedriver/chromedriver.exe"  # Update path for Windows
BRAVE_PATH = "C:/Program Files/BraveSoftware/Brave-Browser/Application/brave.exe"

options = Options()
options.binary_location = BRAVE_PATH
options.add_argument("--headless")  # Run without opening browser (optional)
options.add_argument("--disable-gpu")

service = Service(CHROMEDRIVER_PATH)
driver = webdriver.Chrome(service=service, options=options)

# Search Query
SEARCH_QUERY = "beauty store in Mexico City"

def scrape_google():
    driver.get(f"https://www.google.com/search?q={SEARCH_QUERY}")
    time.sleep(5)

    # Scrape store names
    stores = driver.find_elements(By.XPATH, "//h3")
    store_names = [store.text for store in stores if store.text.strip()]

    # Scrape phone numbers
    phones = driver.find_elements(By.XPATH, "//span[contains(text(), '+52')]")
    phone_numbers = [phone.text for phone in phones]

    # Scrape websites
    links = driver.find_elements(By.XPATH, "//a[contains(@href, 'http')]")
    websites = [link.get_attribute("href") for link in links]

    # Ensure lists are of the same length
    max_length = max(len(store_names), len(phone_numbers), len(websites))
    store_names += ["N/A"] * (max_length - len(store_names))
    phone_numbers += ["N/A"] * (max_length - len(phone_numbers))
    websites += ["N/A"] * (max_length - len(websites))

    # Save data to CSV
    df = pd.DataFrame({
        "Store Name": store_names,
        "Phone Number": phone_numbers,
        "Website": websites
    })
    df.to_csv("beauty_stores_selenium.csv", index=False)
    print("✅ Scraped data saved to 'beauty_stores_selenium.csv'.")

# Run scraper
scrape_google()
driver.quit()


# ========scrape store websites to extract brands sold=============================

from bs4 import BeautifulSoup
import requests

def get_brands_from_website(website_url):
    try:
        response = requests.get(website_url, timeout=10)
        soup = BeautifulSoup(response.text, 'html.parser')

        # Example: Looking for sections listing brands
        brands = [tag.text.strip() for tag in soup.find_all("div", class_="brand-name")]
        return ", ".join(brands) if brands else "N/A"
    except:
        return "N/A"

# Scrape brand data for each store
for index, row in beauty_stores.iterrows():
    if row["Website"] != "N/A":
        brands = get_brands_from_website(row["Website"])
        beauty_stores.at[index, "Brands Sold"] = brands

# Save final dataset
beauty_stores.to_csv("beauty_stores_final.csv", index=False)
print("✅ Final dataset with brands saved.")



