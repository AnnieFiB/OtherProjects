"""
Capstone Project Summary: Web Scraping Case Study
--------------------------------------------------
This capstone project focuses on building a web scraping solution to extract product data from Alibaba's website [Industrial-Machinery](https://www.alibaba.com/Industrial-Machinery_p43?spm=a2700.product_home_newuser.category_overview.category-43). 

The project involves the following key components:
- Web Scraping: Implementing a web scraper using Python libraries such as BeautifulSoup and Requests to navigate the website and extract relevant product information.
- Data Cleaning: Processing the raw data to remove duplicates, handle missing values, and ensure consistency.
- Data Storage: Storing the cleaned data in a structured format, such as CSV or a database, for further analysis.

"""
# Import necessary libraries
from selenium.webdriver.chrome.options import Options
from selenium import webdriver
from selenium.webdriver.chrome.service import Service
from webdriver_manager.chrome import ChromeDriverManager
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from bs4 import BeautifulSoup
import pandas as pd
import requests
import time
import os

# Test request to ensure the URL is accessible

url= "https://www.alibaba.com/trade/search?fsb=y&IndexArea=product_en&keywords=industrial+machinery&originKeywords=industrial+machinery&tab=all&&page=1&spm=a2700.galleryofferlist.pagination.0" 
""
response = requests.get(url, headers={"User-Agent": "Mozilla/5.0"})

if response.status_code == 200:
    print("URL is accessible")
else:
    print("Failed to retrieve the webpage")



# Set up Chrome options for headless browsing
options = webdriver.ChromeOptions()
options.add_argument("--headless")
options.add_argument("--disable-blink-features=AutomationControlled")
options.add_argument('--log-level=3') 
options.add_argument("user-agent=Mozilla/5.0 (Windows NT 10.0; Win64; x64)")
driver = webdriver.Chrome(service=Service(ChromeDriverManager().install()), options=options)
driver.execute_script("window.scrollTo(0, document.body.scrollHeight);")

# Step 1: Load Main Category Page
print("🔄 Loading main page...")
driver.get(url)
time.sleep(10) 
WebDriverWait(driver, 10).until(lambda d: d.execute_script("return document.readyState") == "complete")

soup = BeautifulSoup(driver.page_source, "html.parser") 
print("\n Page loaded successfully")

# ==============================
# STEP 1: Extract Product Links
# ==============================
print("\n🔗 Extracting product links...")
product_links = [
    ("https:" + a_tag["href"]) if not a_tag["href"].startswith("http") else a_tag["href"]
    for a_tag in soup.find_all("a", href=True)
    if "/product-detail/" in a_tag["href"]
]
product_links = list(set(product_links))  # Deduplicate
print(f"\n🛍️ Total links scraped: {len(product_links)}")

# ==============================
# STEP 2: Extract Basic Info
# ==============================
product_data = []

product_cards = soup.find_all("div", class_="card-info list-card-layout__info")
print(f"\n Number of products found: {len(product_cards)}")  

print("\n🔍 Extracting title, price, MOQ, amount sold...")
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
df.to_csv("industrial_machinery_products.csv", index=False)
print(f"\n🛍️ Total products scraped: {len(product_data)}\n Sample product data:")
print(df.head(5))

# Close WebDriver
driver.quit()


'''
























# Import necessary libraries
from selenium.webdriver.chrome.options import Options
from selenium import webdriver
from selenium.webdriver.chrome.service import Service
from webdriver_manager.chrome import ChromeDriverManager
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC

from bs4 import BeautifulSoup
import pandas as pd
import requests
import time
import os

# Test request to ensure the URL is accessible

url = 'https://www.alibaba.com/Industrial-Machinery_p43?spm=a2700.product_home_newuser.category_overview.category-43'

response = requests.get(url, headers={"User-Agent": "Mozilla/5.0"})

if response.status_code == 200:
    print("URL is accessible")
else:
    print("Failed to retrieve the webpage")



# Set up Chrome options for headless browsing
options = webdriver.ChromeOptions()
options.add_argument("--headless")
options.add_argument("--disable-blink-features=AutomationControlled")
options.add_argument('--log-level=3')  # Suppress warnings
options.add_argument("user-agent=Mozilla/5.0 (Windows NT 10.0; Win64; x64)")
driver = webdriver.Chrome(service=Service(ChromeDriverManager().install()), options=options)
driver.execute_script("window.scrollTo(0, document.body.scrollHeight);")
WebDriverWait(driver, 10).until(lambda d: d.execute_script("return document.readyState") == "complete")

# Step 1: Load Main Category Page
print("🔄 Loading main page...")
driver.get(url)
time.sleep(10) 

soup = BeautifulSoup(driver.page_source, "html.parser") 
print("\n Page loaded successfully")


# ================================
# STEP 1: Extract product links
# ================================
print("\n🔗 Extracting product links from main page..." )
product_links = [
    ("https:" + a_tag['href']) if not a_tag['href'].startswith("http") else a_tag['href']
    for a_tag in soup.find_all('a', href=True)
    if '/product-detail/' in a_tag['href']
]

product_links = list(set(product_links))
print(f"\n🛍️ Total links scraped: {len(product_links)}\n Sample product links:")
for link in product_links[:5]:
   print(link)


# ================================
# STEP 2: Extract details from product cards (main page)
# ================================
product_cards = soup.find_all("div", class_="card-info list-card-layout__info")
print(f"\n Number of products found: {len(product_cards)}")  
if not product_cards:
    raise RuntimeError("\n No product cards found. Please check the HTML structure or the URL.")

product_data = []

print("\n🔍 Extracting title, price, MOQ from main page cards...")
# Loop through each product card to extract details
for card in product_cards:
    try:

        # Link: Match only if the card contains a link that is in our extracted deduplicated links
        a_tag = card.find('a', href=True)
        if a_tag:
            href = ("https:" + a_tag['href']) if not a_tag['href'].startswith("http") else a_tag['href']
            if href in product_links:

                # Title
                title_tag = card.find("span", title=True)
                title = title_tag["title"].strip() if title_tag else None

                # Price
                price_tag = card.select_one("div.hugo3-util-ellipsis.hugo3-fw-heavy.hugo3-fz-medium")
                price = price_tag.get_text(strip=True) if price_tag else None

                # MOQ
                moq_tag = card.select_one("span.moq-number")
                moq = moq_tag.get_text(strip=True) if moq_tag else None

                product_data.append({"link": href, "title": title, "price": price, "moq": moq})

    except Exception as e:
        print(f"⚠️ Error parsing card: {e}")

print(f"\n✅ Extracted {len(product_data)} product records from main page cards.")
pd.DataFrame(product_data).to_csv("industrial_machinery_products.csv", index=False)

# Display results
print(f"\n🛍️ Total products scraped: {len(product_data)}\n Sample product links:")
for product in product_data[:5]:
   print(product)

# ================================
# STEP 3: Visit product links to get manufacturer/supplier info
# ================================
print("\n🏭 Fetching manufacturer info for each link...")

for product in product_data:
    try:
        driver.get(product["link"])

        # Wait for the manufacturer element to appear
        company_tag = WebDriverWait(driver, 10).until(
            EC.presence_of_element_located((By.CSS_SELECTOR, "div.product-company-info span.company-name.detail-separator a"))
            )
        manufacturer = company_tag.text.strip() if company_tag else "Not specified"

        # Wait for star rating (updated structure)
        rating_tag = WebDriverWait(driver, 10).until(
            EC.presence_of_element_located((By.CSS_SELECTOR, "span.star-rating"))
        )
        product["rating"] = rating_tag.text.strip() if rating_tag else "No rating"

        # Wait for review count (updated structure)
        review_tag = WebDriverWait(driver, 10).until(
            EC.presence_of_element_located((By.CSS_SELECTOR, "div.review-count"))
        )
        product["reviews"] = review_tag.text.strip() if review_tag else "0 reviews"



         # Wait for review details
        review_tag = WebDriverWait(driver, 10).until(
            EC.presence_of_element_located((By.CSS_SELECTOR, "span.detail-review-item.detail-review"))
        )
        product["reviews"] = review_tag.text.strip() if review_tag else "0 reviews"
        
       
        product_data.append({
            "manufacturer": product["manufacturer"],
            "reviews": product["reviews"],
            "rating": product["rating"]
        })

    except Exception:
        print(f"⚠️ Could not extract details for {link}")
        product["manufacturer"] = "Not specified"
        product["reviews"] = "0 reviews"      
        product["rating"] = "No rating"

# Close the Selenium driver
driver.quit()

print(f"\n🛍️ Total full info scraped: {len(product)}\n Sample product full data:")
for product in product_data[:5]:
   print(product)
pd.DataFrame(product_data).to_csv("industrial_machinery_products_with_supplier.csv", index=False)   



# ================================# STEP 4: Combine and display final data
# ==================================

for item in product_data:
    link = item["link"]
    try:
        # Fetch the product details page
        driver.get(link)
        time.sleep(10)
        detail_soup = BeautifulSoup(driver.page_source, 'html.parser')


        # Extract manufacturer details
        company_tag = detail_soup.select_one("span.company-name.detail-separator")
        manufacturer = company_tag.get_text(strip=True) if company_tag else "Not specified"

        # Extract review details
        review_tag = detail_soup.find("span.detail-review-item.detail-review")
        reviews = review_tag.text.strip() if review_tag else "0 reviews"

        # Extract star rating (from the new HTML structure you provided)
        rating_tag = detail_soup.select_one("div.detail-product-comment div")
        rating = rating_tag.text.strip() if rating_tag else "No rating"

        product_data.append({
            "link": link,
            "manufacturer": manufacturer,
            "reviews": reviews,
            "rating": rating
        })

    except Exception as e:
        print(f"⚠️ Could not extract details for {link}: {e}")

print(product_data)  # View the final extracted data






for link in product_links:
    data = next((item for item in product_data if item["link"].split("?")[0] in link), {
        "link": link,
        "title": None,
        "price": None,
        "moq": None
    })

    try:
        driver.get(link)
        time.sleep(10)
        detail_soup = BeautifulSoup(driver.page_source, 'html.parser')

        company_tag = detail_soup.select_one("span.company-name detail-separator")
        manufacturer = company_tag.get_text(strip=True) if company_tag else "Not specified"
        data["manufacturer"] = manufacturer

        review_tag = detail_soup.find("span.detail-review-item.detail-review")
        data["reviews"] = review_tag.text.strip() if review_tag else "0 reviews"

    except Exception:
        print(f"⚠️ Could not extract manufacturer for {link}")

    product_data.append(data)


# ================================
# STEP 4: Filter & Display
# ==================================
driver.quit()

# Keep only complete records
cleaned = [
    p for p in product_data
    if all([p.get("link"), p.get("title"), p.get("price"), p.get("moq"), p.get("manufacturer")])
]


df = pd.DataFrame(cleaned)
print(f"\n✅ Extracted {len(df)} complete product records out of {len(product_data)}")
print(df.head())
df.to_csv("industrial_machinery_products_with_supplier.csv", index=False)
'''

'''

from selenium import webdriver
from selenium.webdriver.chrome.service import Service
from selenium.webdriver.chrome.options import Options
from webdriver_manager.chrome import ChromeDriverManager
from bs4 import BeautifulSoup
import pandas as pd
import time



def scrape_alibaba_industrial_products(base_url: str, scroll_pause: float = 2.0, scroll_times: int = 5):
    
    # Setup headless browser
    options = Options()
    options.add_argument("--headless")
    options.add_argument("--disable-blink-features=AutomationControlled")
    options.add_argument('--log-level=3')
    driver = webdriver.Chrome(service=Service(ChromeDriverManager().install()), options=options)

    print("🔄 Loading main page...")
    driver.get(base_url)
    time.sleep(5)

    # Scroll to load dynamic content
    for _ in range(scroll_times):
        driver.execute_script("window.scrollTo(0, document.body.scrollHeight);")
        time.sleep(scroll_pause)

    print("✅ Page loaded and scrolled.")
    soup = BeautifulSoup(driver.page_source, "html.parser")

    # Extract product cards
    product_cards = soup.find_all("div", class_="hugo4-pc-grid-item")
    print(f"📦 Found {len(product_cards)} product cards.")

    product_data = []

    for card in product_cards:
        try:
            a_tag = card.find("a", class_="hugo-dotelement")
            link = a_tag["href"] if a_tag and a_tag.has_attr("href") else ""
            if link.startswith("//"):
                link = "https:" + link
            elif link.startswith("/"):
                link = "https://www.alibaba.com" + link
            elif not link.startswith("http"):
                continue
            link = link.strip().split("?")[0]

            title_tag = card.find("span", title=True)
            title = title_tag["title"].strip() if title_tag else None

            price_tag = card.select_one("div.hugo3-util-ellipsis.hugo3-fw-heavy.hugo3-fz-medium")
            price = price_tag.get_text(strip=True) if price_tag else None

            moq_tag = card.select_one("span.moq-number")
            moq = moq_tag.get_text(strip=True) if moq_tag else None

            product_data.append({
                "link": link,
                "title": title,
                "price": price,
                "moq": moq
            })

        except Exception as e:
            print(f"⚠️ Error parsing card: {e}")

    print(f"🔗 Extracted {len(product_data)} initial product entries.")

    # Now extract manufacturer info from each link
    print("🏭 Fetching manufacturer info...")
    for data in product_data:
        try:
            driver.get(data["link"])
            time.sleep(5)
            detail_soup = BeautifulSoup(driver.page_source, 'html.parser')
            company_tag = detail_soup.select_one("div.product-company-info a")
            manufacturer = company_tag.get_text(strip=True) if company_tag else "Not specified"
            data["manufacturer"] = manufacturer
        except Exception as e:
            print(f"⚠️ Could not extract manufacturer for {data['link']}: {e}")
            data["manufacturer"] = "Not specified"

    driver.quit()

    # Filter out incomplete records
    final_data = [
        p for p in product_data
        if all([p.get("link"), p.get("title"), p.get("price"), p.get("moq"), p.get("manufacturer")])
    ]

    df = pd.DataFrame(final_data)
    print(f"\n✅ Final product count: {len(df)}")
    return df


# Run the function
url = 'https://www.alibaba.com/Industrial-Machinery_p43?spm=a2700.product_home_newuser.category_overview.category-43'
final_df = scrape_alibaba_industrial_products(url)
# Save to CSV
final_df.to_csv("industrial_machinery_products.csv", index=False)   
print("📊 Data saved to 'industrial_machinery_products.csv'." )

print("\nSample data:")
print(final_df.head(3))
'''