import json
import pandas as pd

def extract_product_info_to_csv(json_path: str, output_csv_path: str) -> str:
    """
    Extracts website, item name, prices, and description info from a JSON file 
    and writes the data to a CSV file.

    Args:
        json_path (str): Path to the JSON input file.
        output_csv_path (str): Path where the CSV output file should be saved.

    Returns:
        str: Path to the saved CSV file.
    """
    with open(json_path, "r", encoding="utf-8") as f:
        json_data = json.load(f)

    product_rows = []

    for entry in json_data:
        url = entry.get("url")
        site_name = None
        title = None
        description = None
        low_price = None
        high_price = None

        if "metadata" in entry:
            title = entry["metadata"].get("title", "").strip()
            description = entry["metadata"].get("description")

            if "openGraph" in entry["metadata"]:
                og = entry["metadata"]["openGraph"]
                og_dict = {
                    item["property"]: item["content"]
                    for item in og if "property" in item and "content" in item
                }
                site_name = og_dict.get("og:site_name")
                low_price = og_dict.get("og:price:amount")
                high_price = og_dict.get("og:price:amount")
                description = og_dict.get("og:description", description)

            if "jsonLd" in entry["metadata"] and isinstance(entry["metadata"]["jsonLd"], list):
                for block in entry["metadata"]["jsonLd"]:
                    if block.get("@type") == "Product":
                        title = block.get("name", title)
                        description = block.get("description", description)
                        offers = block.get("offers", {})
                        if isinstance(offers, dict):
                            low_price = offers.get("lowPrice", low_price)
                            high_price = offers.get("highPrice", high_price)

        if title and low_price:
            product_rows.append({
                "Website": url,
                "Website Name": site_name,
                "Item Name": title,
                "Description": description,
                "Low Price (MXN)": low_price,
                "High Price (MXN)": high_price,
                "Quantity": "",  # Field not available
            })

    df_products = pd.DataFrame(product_rows)
    df_products.to_csv(output_csv_path, index=False, encoding="utf-8-sig")

    return output_csv_path
