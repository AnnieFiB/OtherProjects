import time
import random
import requests
import re
import pandas as pd
from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.common.keys import Keys
from bs4 import BeautifulSoup

# Set Brave as browser
brave_path = "/usr/bin/brave-browser"  # Update path if needed
# Windows users: brave_path = "C:/Program Files/BraveSoftware/Brave-Browser/Application/brave.exe"

chrome_options = webdriver.ChromeOptions()
chrome_options.binary_location = brave_path
chrome_options.add_argument("--start-maximized")

# Set path to ChromeDriver (update this path)
driver = webdriver.Chrome(executable_path="/path/to/chromedriver", options=chrome_options)

# Search queries
search_terms = [
    "Beauty shops in Mexico City",
    "Pharmacies selling beauty products in Mexico City",
    "Local department stores selling beauty products in Mexico City",
    "Spas selling beauty products in Mexico City"
]

# Store results
all_data = []

# Function to scrape email from a website
def extract_email_from_website(website_url):
    try:
        headers = {"User-Agent": "Mozilla/5.0"}
        response = requests.get(website_url, headers=headers, timeout=5)
        if response.status_code == 200:
            emails = re.findall(r"[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}", response.text)
            return emails[0] if emails else "N/A"
    except Exception:
        return "N/A"
    return "N/A"

for search_query in search_terms:
    print(f"Searching: {search_query}")

    # Open Google Maps
    driver.get("https://www.google.com/maps")
    time.sleep(random.uniform(3, 6))  # Random delay between 3-6 seconds

    # Search for the query
    search_box = driver.find_element(By.XPATH, '//input[@id="searchboxinput"]')
    search_box.clear()
    search_box.send_keys(search_query)
    search_box.send_keys(Keys.RETURN)
    time.sleep(random.uniform(5, 10))  # Random delay to allow results to load

    # Scroll to load more results
    for _ in range(random.randint(4, 6)):  # Scroll 4-6 times
        driver.find_element(By.TAG_NAME, 'body').send_keys(Keys.PAGE_DOWN)
        time.sleep(random.uniform(2, 5))  # Random delay per scroll

    # Extract page source and parse with BeautifulSoup
    soup = BeautifulSoup(driver.page_source, "html.parser")

    # Find all business listings
    businesses = soup.find_all("div", class_="Nv2PK")

    for business in businesses:
        try:
            # Click on each business to open details
            business.click()
            time.sleep(random.uniform(4, 8))  # Random delay before extracting details

            # Extract business details
            page_source = driver.page_source
            detail_soup = BeautifulSoup(page_source, "html.parser")

            name = detail_soup.find("h1", class_="DUwDvf").text if detail_soup.find("h1", class_="DUwDvf") else "N/A"
            address = detail_soup.find("button", {"data-tooltip": "Copy address"}).text if detail_soup.find("button", {"data-tooltip": "Copy address"}) else "N/A"
            rating = detail_soup.find("span", class_="MW4etd").text if detail_soup.find("span", class_="MW4etd") else "N/A"

            # Extract phone number
            phone_number = "N/A"
            phone_element = detail_soup.find("button", {"data-tooltip": "Copy phone number"})
            if phone_element:
                phone_number = phone_element.text

            # Extract website link
            website = "N/A"
            website_element = detail_soup.find("a", {"data-tooltip": "Open website"})
            if website_element:
                website = website_element["href"]

            # Extract email from website (if website exists)
            email = extract_email_from_website(website) if website != "N/A" else "N/A"

            # Extract brands sold (if mentioned in descriptions)
            description = detail_soup.find("div", class_="UXwVOc").text if detail_soup.find("div", class_="UXwVOc") else "N/A"
            brands_sold = "N/A"
            known_brands = ["L'Oréal", "Maybelline", "MAC", "Revlon", "Dior", "Sephora", "Neutrogena", "Clinique"]
            for brand in known_brands:
                if brand.lower() in description.lower():
                    brands_sold = brand if brands_sold == "N/A" else f"{brands_sold}, {brand}"

            # Estimate quantity sold per week (if mentioned in reviews)
            quantity_sold = "N/A"
            review_section = detail_soup.find_all("span", class_="wiI7pd")
            for review in review_section:
                if "per week" in review.text.lower() or "weekly sales" in review.text.lower():
                    quantity_sold = review.text
                    break

            # Append data
            all_data.append([search_query, name, address, rating, phone_number, email, website, brands_sold, quantity_sold])

            # Random pause after scraping a business to avoid detection
            time.sleep(random.uniform(5, 10))

        except Exception as e:
            print(f"Error extracting details: {e}")
            time.sleep(random.uniform(2, 5))  # Short delay before continuing

# Convert to DataFrame
df = pd.DataFrame(all_data, columns=["Category", "Name", "Address", "Rating", "Phone Number", "Email", "Website", "Brands Sold", "Quantity Sold per Week"])

# Save to CSV
df.to_csv("beauty_shops_mexico_city_extended.csv", index=False)

# Close browser
driver.quit()

print("Scraping completed! Data saved as 'beauty_shops_mexico_city_extended.csv'.")
