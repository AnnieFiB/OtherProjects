import os
import socket
import time
import pandas as pd
import requests
from bs4 import BeautifulSoup
from selenium import webdriver
from selenium.webdriver.chrome.service import Service
from selenium.webdriver.chrome.options import Options
from webdriver_manager.chrome import ChromeDriverManager
from fake_useragent import UserAgent

# Configuration
CSV_FILE = "datasets/websites.csv"
OUTPUT_FILE = "datasets/products.csv"
MAX_RETRIES = 2
TIMEOUT = 15
DELAY = 1  # Seconds between requests

# Initialize tools
ua = UserAgent()

def setup_driver():
    """Configure Chrome with automatic driver management"""
    try:
        options = Options()
        options.add_argument("--headless")
        options.add_argument("--disable-blink-features=AutomationControlled")
        options.add_argument(f"user-agent={ua.random}")
        
        service = Service(ChromeDriverManager().install())
        return webdriver.Chrome(service=service, options=options)
    except Exception as e:
        print(f"Browser setup failed: {str(e)}")
        return None

def safe_scrape(url):
    """Try requests first, fallback to Selenium"""
    try:
        headers = {"User-Agent": ua.random}
        response = requests.get(url, headers=headers, timeout=TIMEOUT)
        if response.status_code == 200:
            return response.text
    except Exception as e:
        print(f"Requests failed: {str(e)[:100]}...")
    
    # Fallback to Selenium
    driver = setup_driver()
    if driver:
        try:
            driver.get(url)
            time.sleep(DELAY * 2)  # Extra time for JS rendering
            return driver.page_source
        except Exception as e:
            print(f"Selenium failed: {str(e)[:100]}...")
        finally:
            driver.quit()
    return None

def extract_product_data(html, url):
    """Flexible product data extraction"""
    if not html:
        return []
    
    soup = BeautifulSoup(html, 'html.parser')
    products = []
    
    # Find potential product containers
    containers = soup.find_all(class_=lambda x: x and 'product' in x.lower())
    containers = containers or soup.find_all(['article', 'section'])
    
    for item in containers[:5]:  # Limit to first 5 products
        product = {
            'website': url,
            'name': "Not found",
            'price': "Not found",
            'description': "Not found",
            'brand': "Not found"
        }
        
        # Name detection
        name_selectors = ['h1', 'h2', 'h3', 'div.product-name', 'span.title']
        product['name'] = next((e.text.strip() for s in name_selectors 
                              for e in item.find_all(s)), product['name'])
        
        # Price detection
        price_text = item.find(string=lambda x: x and any(c in x for c in ['$', '€', '£']))
        if price_text:
            product['price'] = ''.join(filter(lambda x: x.isdigit() or x in ['.', ','], price_text))
        
        # Description detection
        desc = item.find('p') or item.find('div', class_=lambda x: x and 'desc' in x.lower())
        if desc:
            product['description'] = desc.text.strip()[:200]  # Truncate long descriptions
        
        # Brand detection
        product['brand'] = url.split('//')[-1].split('.')[0].title()
        
        products.append(product)
    
    return products

def process_url(url):
    """Handle full processing for one URL"""
    print(f"\nProcessing: {url}")
    
    # Domain verification
    try:
        domain = url.split('//')[-1].split('/')[0]
        socket.gethostbyname(domain)
    except:
        print(f"Invalid domain: {url}")
        return []
    
    # Attempt scraping
    html = None
    for attempt in range(MAX_RETRIES):
        html = safe_scrape(url)
        if html:
            break
        print(f"Retry {attempt + 1}/{MAX_RETRIES}")
        time.sleep(DELAY)
    
    return extract_product_data(html, url) if html else []

if __name__ == "__main__":
    # Load websites
    websites = pd.read_csv(CSV_FILE)['Website'].tolist()
    
    # Process all websites
    results = []
    for url in websites:
        results.extend(process_url(url))
        time.sleep(DELAY)  # Be polite
    
    # Save results
    if results:
        pd.DataFrame(results).to_csv(OUTPUT_FILE, index=False)
        print(f"\nSuccessfully saved {len(results)} products to {OUTPUT_FILE}")
    else:
        print("\nNo products found. Check your websites list and internet connection.")