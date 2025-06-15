<div style="text-align:center; border: 1px solid #808080; padding: 10px;">
  <h1 style="color: gray; font-weight: bold;">Web Scraping Capstone Project</h1>
  <h2 style="color: gray; font-weight: bold;">Automated Data Extraction from Alibaba Product Listings</h2>
  <p><strong>Student:</strong> Anthonia Fisuyi &nbsp; | &nbsp; <strong>Date:</strong> 2025-06-13</p>
</div>

---

## Project Overview

This project demonstrates how to design and implement a web scraping pipeline that extracts structured product data from Alibaba. The objective is to transform unstructured online listings into a business-ready dataset using Python tools.

---

## Objective

Build an automated scraping solution that:
- Extracts product name, price, manufacturer, and link
- Navigates multiple pages (pagination)
- Outputs the data to a `.csv` or tabular format

---

## Business Problem

Businesses lack efficient ways to collect online product data for competitor analysis, trend spotting, and vendor discovery. Manual data entry is inefficient and error-prone. This scraper addresses that gap by automating data collection from the source.

---

## Business Value

- Access to real-time product and pricing insights  
- Faster competitor monitoring, market and supplier analysis
- Automation of manual data collection workflows  
- Structured datasets for dashboards and analytics

---

## Tools & Technologies

- **Python 3**  
- `requests` and `BeautifulSoup` for scraping  
- `pandas` for data processing  
- CSV/Excel for structured output  
- *(Optionally: `Selenium` for dynamic content or persistence)*

---

## Workflow

1. **Target Website**: [Alibaba](https://www.alibaba.com)
2. **Selected Category**: _Industrial-Machinery_  
3. **Scraping Steps**:
   - Inspect site using browser dev tools  
   - Extract product name, price, link, and manufacturer using selenium & beautifulsoup 
   - Handle pagination (for loop) 
4. **Output**: Save clean data to a `.csv` file

---

## Sample Output

| Title                   | Price   | Manufacturer | Link                        |Mog  |Amount Sold|
|-------------------------|---------|--------------|-----------------------------|-----|-----------|
| Wireless Earbuds Pro    | $39.99  | SoundPro     | alibaba.com/item/xyz123     |1 set	|502 views|
| Solar Powered Lantern   | $12.50  | EcoLite      | alibaba.com/item/eco567     |1 piece	|No sales data|


---

## Learning Outcomes

- Built scrapers using BeautifulSoup/Selenium  
- Practiced data validation and error handling  
- Converted HTML tables into structured datasets  
- Learned best practices for ethical scraping (throttle time)

---

## ✅ Project Status

✔️ Completed and tested on one product category  
📂 Output file contains 5000+ clean rows in CSV format  

---

## ⚠️ Ethics Note

This project was conducted for academic purposes only. Scraping was performed with care, respecting site structure and access rules.
