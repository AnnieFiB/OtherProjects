import requests
from bs4 import BeautifulSoup
import pandas as pd
import logging
import os

# Configure logging
logging.basicConfig(
    filename='scrape_errors.log',  # Log file name
    level=logging.WARNING,         # Log level (WARNING, ERROR, INFO, etc.)
    format='%(asctime)s - %(levelname)s - %(message)s'  # Log format
)

def scrape_brand_and_quantity(url):
    """
    Scrapes the brand and quantity from a given URL.
    """
    try:
        # Send a GET request to the website
        response = requests.get(url)
        response.raise_for_status()  # Raise an error for bad status codes

        # Parse the HTML content using BeautifulSoup
        soup = BeautifulSoup(response.text, 'html.parser')

        # Example: Scrape the brand name (adjust the selector as needed)
        brand_element = soup.find('h1', class_='brand-name')
        brand = brand_element.text.strip() if brand_element else None

        # Example: Scrape the quantity (adjust the selector as needed)
        quantity_element = soup.find('span', class_='quantity')
        quantity = quantity_element.text.strip() if quantity_element else None

        return brand, quantity

    except requests.exceptions.RequestException as e:
        logging.error(f"Request failed for {url}: {e}")
    except Exception as e:
        logging.error(f"Unexpected error scraping {url}: {e}")

    return None, None

def scrape_data(df):
    """
    Scrapes brand and quantity for all URLs in the DataFrame.
    Returns a new DataFrame with the scraped data.
    """
    # Convert column names to lowercase
    df.columns = df.columns.str.lower()

    # Check if 'website' column exists
    if 'website' not in df.columns:
        raise ValueError("The DataFrame must contain a 'website' column.")

    # Create a new DataFrame to store the scraped data
    scraped_data = pd.DataFrame(columns=['website', 'brand', 'quantity'])

    # Notify that scraping has started
    print("Scraping started...")

    # Apply the scraping function to each URL in the DataFrame
    for index, row in df.iterrows():
        url = row['website']

        # Skip rows with NaN or empty URLs
        if pd.isna(url) or url == "":
            logging.warning(f"Skipping row {index} due to missing or empty URL.")
            continue

        print(f"Scraping {url}...")  # Show progress for each URL
        brand, quantity = scrape_brand_and_quantity(url)

        # Append the new data to the DataFrame using pd.concat
        new_row = pd.DataFrame({
            'website': [url],
            'brand': [brand],
            'quantity': [quantity]
        })
        scraped_data = pd.concat([scraped_data, new_row], ignore_index=True)

    # Notify that scraping has completed
    print("Scraping completed!")

    return scraped_data

def save_scraped_data(scraped_data, output_folder='datasets', output_file='scraped_data.csv'):
    """
    Saves the scraped data to a CSV file in the specified output folder.
    """
    # Create the output folder if it doesn't exist
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    # Save the scraped data to a CSV file
    output_path = os.path.join(output_folder, output_file)
    scraped_data.to_csv(output_path, index=False)
    print(f"Scraped data saved to {output_path}")