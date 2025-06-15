"""
Capstone Project Summary: Web Scraping Case Study
--------------------------------------------------
This capstone project focuses on building a web scraping solution to extract product data from Alibaba's website [Industrial-Machinery](https://www.alibaba.com/Industrial-Machinery_p43?spm=a2700.product_home_newuser.category_overview.category-43). 

The project involves the following key components:
- Web Scraping: Implementing a web scraper using Python libraries such as BeautifulSoup and Requests to navigate the website and extract relevant product information.
- Data Cleaning: Processing the raw data to remove duplicates, handle missing values, and ensure consistency.
- Data Storage: Storing the cleaned data in a structured format, such as CSV or a database, for further analysis.

"""

import time
import pandas as pd
from selenium import webdriver
from selenium.webdriver.chrome.service import Service
from webdriver_manager.chrome import ChromeDriverManager
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from bs4 import BeautifulSoup

# Setup Selenium with Chrome
options = webdriver.ChromeOptions()
options.add_argument("--headless")  # Run in headless mode for speed
options.add_argument("--disable-blink-features=AutomationControlled")  # Reduce bot detection
options.add_argument("user-agent=Mozilla/5.0 (Windows NT 10.0; Win64; x64)")
driver = webdriver.Chrome(service=Service(ChromeDriverManager().install()), options=options)

#full_url= "https://www.alibaba.com/trade/search?fsb=y&IndexArea=product_en&keywords=industrial+machinery&originKeywords=industrial+machinery&tab=all&&page=1&spm=a2700.galleryofferlist.pagination.0" 

base_url = "https://www.alibaba.com/trade/search?fsb=y&IndexArea=product_en&keywords=industrial+machinery&originKeywords=industrial+machinery&tab=all&&page="

product_data = []
max_pages = 2

# Loop through pages dynamically
for page in range(1, max_pages + 1):

    url = f"{base_url}{page}&spm=a2700.galleryofferlist.pagination.0"
    print(f"\n📄 Scraping page {page}: {url}")
    driver.get(url)
    time.sleep(5)  # Allow page to load

    # Ensure full page load before extracting data
    WebDriverWait(driver, 10).until(lambda d: d.execute_script("return document.readyState") == "complete")
    soup = BeautifulSoup(driver.page_source, "html.parser")
    
    product_cards = soup.find_all("div", class_="card-info list-card-layout__info")
    print(f"\n Number of products found: {len(product_cards)}")  
    print("\n🔍 Extracting link, title, price, MOQ, amount sold...")

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

# ==============================
# STEP 3: Save and Display
# ==============================

df= pd.DataFrame(product_data)
df.to_csv("industrial_machinery_products_all.csv", index=False)
print(f"\n🛍️ Total products scraped: {len(product_data)}\n Sample product data:")
print(df.head(5))

# Close WebDriver
driver.quit()



