
from bs4 import BeautifulSoup
import requests
import re
import pandas as pd


url = 'https://www.walmart.com/shop/deals/flash-deals?athAsset=eyJhdGhjcGlkIjoiZDk5NWUwNjUtMDQ1Yi00NWRmLTgwMTEtYmRhYWFiY2M3NmUyIiwiYXRoZ2FpIjoiMSIsImFld3IiOiJNUF9DVFIifQ==&athena=true'

headers = {
    "User-Agent": "Mozilla/5.0"
}

r = requests.get(url, headers=headers)
soup = BeautifulSoup(r.text, 'html.parser')
#print(soup.prettify())

# Find all flashdeal products title, original price, discounted price, price, ratings , review and links for each product block
cards = soup.find_all("div", class_="flex bg-white br3 mb3")
print(f"Found {len(cards)} product cards.")

products = []

for card in cards:
    product = {
        "title": "",
        "price": "",
        "old_price": "",
        "savings": "",
        "rating": "",
        "review": "",
        "link": ""
    }

    # Link
    link_tag = card.find("a", class_="z-1")
    if link_tag and link_tag.has_attr("href") and link_tag["href"].startswith("/ip/"):
        product["link"] = "https://www.walmart.com" + link_tag["href"]

    # Text spans
    spans = card.find_all("span", class_="w_iUH7")
    for span in spans:
        text = span.get_text(strip=True)
        if not product["title"]:
            product["title"] = text
        elif "current price" in text.lower():
            product["price"] = text.replace("current price Now", "").strip()
        elif text.startswith("Was"):
            product["old_price"] = text.replace("Was", "").strip()
        elif "you save" in text.lower():
            product["savings"] = text.replace("You save", "").strip()
        elif "out of" in text:
            rating_match = re.search(r"([\d.]+) out of 5", text)
            review_match = re.search(r"(\d+(,\d+)*) reviews", text)
            if rating_match:
                product["rating"] = rating_match.group(1)
            if review_match:
                product["review"] = review_match.group(1)

    # Save only valid records
    if product["link"] and product["title"]:
        products.append(product)

# Output
for p in products:
    print(p)

# ✅ Save to CSV
df = pd.DataFrame(products)
df.to_csv("walmart_flash_deals.csv", index=False)


print(f"\n Scraped {len(products)} products.")
print(" Saved to walmart_flash_deals.csv")




