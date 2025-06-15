# ==========================================================================================================================
# Capstone Project Summary: Web Scraping Case Study
#  --------------------------------------------------
# This capstone project focuses on building a web scraping solution to extract product data from Alibaba's website [Industrial-Machinery](https://www.alibaba.com/trade/search?fsb=y&IndexArea=product_en&keywords=industrial+machinery&originKeywords=industrial+machinery&tab=all&&page=1&spm=a2700.galleryofferlist.pagination.0). 

# The project involves the following key components:
# - Web Scraping: Implementing a web scraper using Python libraries such as Selenium, BeautifulSoup and Requests to navigate the dynamic website and extract relevant product information.
# - Data Cleaning: Processing the raw data to remove duplicates, handle missing values, and ensure consistency.
# - Data Storage: Storing the cleaned data in a structured format, such as CSV or a database, for further analysis.

# ============================================================================================================================

import random
from selenium import webdriver
from selenium.webdriver.chrome.service import Service
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from webdriver_manager.chrome import ChromeDriverManager
from bs4 import BeautifulSoup
import requests
import pandas as pd
import time

# =================================================
# STEP 1: Test url for response.status_code == 200 
# =================================================

base_url = "https://www.alibaba.com/trade/search?fsb=y&IndexArea=product_en&keywords=industrial+machinery&originKeywords=industrial+machinery&tab=all&&page="

response = requests.get(base_url, headers={"User-Agent": "Mozilla/5.0"})

if response.status_code == 200:
    print("URL is accessible")
else:
    print("Failed to retrieve the webpage")


# =================================
# STEP 2: Setup Selenium for Chrome
# ==================================
 
options = webdriver.ChromeOptions()
options.add_argument("--headless")
options.add_argument("--disable-blink-features=AutomationControlled")  # Reduce bot detection
options.add_argument('--log-level=3') 
driver = webdriver.Chrome(service=Service(ChromeDriverManager().install()), options=options)


# ==========================================================================
# STEP 3: Initialise Product list and Extract Basic Info
# ===========================================================================

product_data = []

#  Option 1: using for loop to extract specific numbers of pages

max_pages = 2

for page in range(1, max_pages + 1):

    url = f"{base_url}{page}&spm=a2700.galleryofferlist.pagination.0"
    print(f"\n=== Loading and Scraping page {page}")

    for attempt in range(3):
        try:
            driver.get(url)
            break  # success
        except Exception:
            print(f"⚠️ Retry {attempt+1}/3 for page {page}")
            time.sleep(15)
    else:
        print(f"❌ Skipping page {page} after 3 failed attempts.")
        continue
   
    time.sleep(15)

    # Ensure full page load before extracting data
    WebDriverWait(driver, 10).until(lambda d: d.execute_script("return document.readyState") == "complete")
    soup = BeautifulSoup(driver.page_source, "html.parser")
    
    product_cards = soup.find_all("div", class_="card-info list-card-layout__info")
    print(f"🔍 Number of products found: {len(product_cards)}")  
    print("=== Extracting link, title, price, MOQ, amount sold...")

    if not product_cards:
        print("\n❌ No more products found, stopping pagination.")
        break

    # Extract product data

    for card in product_cards:
        try:
            # Extract product link
            a_tag = card.find("a", href=True)
            link = ("https:" + a_tag["href"]) if a_tag and not a_tag["href"].startswith("http") else a_tag["href"] if a_tag else "Link not found"

            # Extract title
            title_tag = card.find("h2", class_="search-card-e-title")
            title = title_tag.get_text(strip=True) if title_tag else "Title not found"

            # Extract price
            price_tag = card.find("div", class_="search-card-e-price-main")
            price = price_tag.get_text(strip=True) if price_tag else "Price not found"

            # Extract MOQ (Min. Order Quantity)
            moq_tag = card.find("div", class_="search-card-m-sale-features__item tow-line")
            moq = moq_tag.get_text(strip=True).replace("Min. order:", "").strip() if moq_tag else "MOQ not found"

            # Extract Amount Sold
            sold_tag = card.find("div", class_="search-card-e-market-power-common")
            amount_sold = sold_tag.get_text(strip=True) if sold_tag else "No sales data"

            # Extract Manufacturer
            manufacturer_tag = card.find("a", class_="search-card-e-company margin-bottom-12")
            manufacturer = manufacturer_tag.get_text(strip=True) if manufacturer_tag else "Manufacturer not found"

            product_data.append({
                "link": link,
                "title": title,
                "price": price,
                "moq": moq,
                "amount_sold": amount_sold,
                "manufacturer": manufacturer
            })
        
        except Exception:
            print("⚠️ Error extracting product data")
  
    '''# Option 2: Using a while loop to extract all available pages dynamically

page = 1

while True:
    url = f"{base_url}{page}&spm=a2700.galleryofferlist.pagination.0"
    print(f"\n=== Loading and Scraping page {page}")

    for attempt in range(3):
        try:
            driver.get(url)
            break  # success
        except Exception:
            print(f"⚠️ Retry {attempt+1}/3 for page {page}")
            time.sleep(15)
    else:
        print(f"❌ Skipping page {page} after 3 failed attempts.")
        continue
   
    time.sleep(15)

    # Wait for cards to load
    WebDriverWait(driver, 20).until(
        EC.presence_of_all_elements_located((By.CSS_SELECTOR, "div.card-info.list-card-layout__info"))
    )

    soup = BeautifulSoup(driver.page_source, "html.parser")
    product_cards = soup.find_all("div", class_="card-info list-card-layout__info")
    print(f"🔍 Found {len(product_cards)} product cards")
    print("=== Extracting link, title, price, MOQ, amount sold, manufacturer/supplier...")


    if not product_cards:
        print("❌ No more products found. Stopping...")
        break

    for card in product_cards:
        try:
            a_tag = card.find("a", href=True)
            link = ("https:" + a_tag["href"]) if a_tag and not a_tag["href"].startswith("http") else a_tag["href"]

            title_tag = card.find("h2", class_="search-card-e-title")
            title = title_tag.get_text(strip=True) if title_tag else "N/A"

            price_tag = card.find("div", class_="search-card-e-price-main")
            price = price_tag.get_text(strip=True) if price_tag else "N/A"

            moq_tag = card.find("div", class_="search-card-m-sale-features__item tow-line")
            moq = moq_tag.get_text(strip=True).replace("Min. order:", "").strip() if moq_tag else "N/A"

            sold_tag = card.find("div", class_="search-card-e-market-power-common")
            amount_sold = sold_tag.get_text(strip=True) if sold_tag else "N/A"

            manufacturer_tag = card.find("a", class_="search-card-e-company margin-bottom-12")
            manufacturer = manufacturer_tag.get_text(strip=True) if manufacturer_tag else "N/A"

            product_data.append({
                "link": link,
                "title": title,
                "price": price,
                "moq": moq,
                "amount_sold": amount_sold,
                "manufacturer": manufacturer
            })
            print(f"✅ Extracted {len(product_data)} product records from page {page}")

        except Exception:
            print("⚠️ Error extracting product data")

    page += 1
    time.sleep(random.uniform(15, 30))   # Throttle to avoid being blocked
'''

# ======================================
# STEP 4: Save , Display and Quit driver
# =======================================

df = pd.DataFrame(product_data)
df.to_csv("industrial_machinery_products_all.csv", index=False)
print(f"\n✅ Total products scraped: {len(product_data)}\n Sample product data:")
print(df.head())

driver.quit()

