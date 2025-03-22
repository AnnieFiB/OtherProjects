import requests
from bs4 import BeautifulSoup
import pandas as pd
import time

# Function to scrape product data from xabone.com
def scrape_xabone():
    base_url = "https://www.xabone.com"
    collection_url = f"{base_url}/collections/maquillaje-organico-mexico"
    headers = {'User-Agent': 'Mozilla/5.0'}

    response = requests.get(collection_url, headers=headers)
    soup = BeautifulSoup(response.text, 'html.parser')

    products = soup.select('div.grid-product__content')
    product_data = []

    for product in products:
        name_tag = product.select_one('.grid-product__title')
        price_tag = product.select_one('.grid-product__price span')
        link_tag = product.select_one('a')

        name = name_tag.text.strip() if name_tag else 'N/A'
        price = price_tag.text.strip() if price_tag else 'N/A'
        relative_url = link_tag['href'] if link_tag else ''
        product_url = base_url + relative_url

        # Visit product page for description
        prod_response = requests.get(product_url, headers=headers)
        prod_soup = BeautifulSoup(prod_response.text, 'html.parser')
        desc_tag = prod_soup.select_one('.product-single__description')
        desc = desc_tag.text.strip() if desc_tag else 'N/A'

        # Quantity information is not available
        quantity = 'N/A'

        product_data.append([
            'xabone.com',
            name,
            desc,
            product_url,
            price,
            quantity,
            'XABONE Cosméticos Orgánicos',
            'Mario Pani 750, Santa Fe, CDMX',
            '+52 (55) 7394 7600',
            'hola@xabone.com'
        ])

        time.sleep(1)  # Be respectful with a delay between requests

    return product_data

# Main function to execute scraping and save to CSV
def main():
    all_product_data = []

    # Scrape data from xabone.com
    xabone_data = scrape_xabone()
    all_product_data.extend(xabone_data)

    # Add more scraping functions for other websites here
    # e.g., liverpool_data = scrape_liverpool()
    # all_product_data.extend(liverpool_data)

    # Define CSV column headers
    columns = [
        'Website', 'Product Name', 'Description', 'Product URL',
        'Price', 'Quantity', 'Website Name', 'Address', 'Phone', 'Email'
    ]

    # Create a DataFrame and save to CSV
    df = pd.DataFrame(all_product_data, columns=columns)
    df.to_csv('products.csv', index=False, encoding='utf-8')
    print("Data has been scraped and saved to 'products.csv'.")

