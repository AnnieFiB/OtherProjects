import requests
from bs4 import BeautifulSoup
import pandas as pd
import time
import random

# Configuration
BASE_URL = "https://www.elpalaciodehierro.com/belleza/maquillaje"
CATEGORY = "Makeup"
HEADERS = {
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36"
}
DELAY = 2  # Seconds between requests to avoid blocking
PAGES = 36  # Total pages to scrape

def scrape_el_palacio():
    products = []
    
    for page in range(1, PAGES + 1):
        # Add pagination and random delay
        url = f"{BASE_URL}?page={page}"
        time.sleep(DELAY + random.uniform(0, 1))  # Randomized delay
        
        try:
            response = requests.get(url, headers=HEADERS)
            response.raise_for_status()
            soup = BeautifulSoup(response.text, "html.parser")
            
            # Find all product items (update class based on actual HTML)
            product_items = soup.find_all("div", class_="product-item")
            
            for item in product_items:
                # Extract data with error handling
                try:
                    # Product Name & Brand from alt text
                    img_tag = item.find("img", class_="h-blend_mode_img")
                    alt_text = img_tag.get("alt", "")
                    brand = alt_text.split()[0] if alt_text else "Not Available"
                    name = " ".join(alt_text.split()[1:-3]) if alt_text else "Not Available"
                    
                    # Price (update class based on actual HTML)
                    price_tag = item.find("span", class_="price")
                    price = price_tag.text.strip() if price_tag else "Not Available"
                    
                    # Quantity (from alt text)
                    quantity = alt_text.split()[-3] if alt_text else "Not Available"
                    
                    # Image URL
                    image_url = img_tag.get("data-src") or img_tag.get("src")
                    
                    # Product Page URL
                    link_tag = item.find("a", href=True)
                    product_url = link_tag["href"] if link_tag else "Not Available"
                    
                    # Append to list
                    products.append([
                        name, brand, CATEGORY, price, quantity,
                        image_url, product_url
                    ])
                    
                except Exception as e:
                    print(f"Error parsing item: {e}")
                    continue
                    
        except Exception as e:
            print(f"Failed to scrape page {page}: {e}")
            continue
    
    # Create DataFrame
    df = pd.DataFrame(products, columns=[
        "Product Name", "Brand", "Category", "Price (MXN)",
        "Quantity", "Image URL", "Product Page URL"
    ])
    df.to_csv("el_palacio_makeup_products.csv", index=False)
    print("Scraping completed. Data saved to CSV.")

if __name__ == "__main__":
    scrape_el_palacio()