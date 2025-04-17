<p align="center"><strong>Sales Performance Analysis for Product Insights</strong></p>
<p align="center"><strong>Author: Anthonia, Specialization: Analysis,Business Focus: Sales,Tool: SQL</strong></p>

---

## Case Study Overview

The company specializes in delivering **data-driven insights** through intuitive reporting and analytics. By using **simple SQL fundamentals**, the goal is to help businesses:

- Uncover trends  
- Optimize operations  
- Drive smarter decision-making

---

## 🧠 Problem Statement

A **retail company** wants to explore its historical sales data to better understand product and regional performance. Key business questions include:

1. What are the **top 5 selling products** by quantity?  
2. What is the **total revenue per product**?  
3. How does **sales performance vary by state**?  
4. Which **customers contributed most to revenue**?  
5. Which **month had the most sales**?  
6. Who are the **top 5 recurring customers**?  
7. Which **product category performs best** (top 5)?  
8. Which **sub-category performs best** (top 5)?  
9. What is the **most frequent payment method**?  
10. Which **city had the highest sales**?

---

## Business Value

This analysis will enable the business to:

- Gain **clear insights** into product and customer performance  
- **Enhance customer satisfaction** and profitability  
- Make **data-informed decisions** instead of relying on assumptions  
- Optimize inventory and resource allocation  

---

## Data Description

<a href="https://www.kaggle.com/datasets/shantanugarg274/sales-dataset?resource=download">sales_csv</a>

The dataset includes historical sales records such as:

| **Column Name**   | **Description**                                                                 |
|-------------------|----------------------------------------------------------------------------------|
| Order ID          | Unique identifier for each order.                                               |
| Amount            | **Total sale value** of the order (includes both cost and profit).              |
| Profit            | Profit earned from the order (i.e., `Amount - Cost`).                           |
| Quantity          | Number of items sold in the order.                                              |
| Category          | Broad classification of the product (e.g., Electronics).                        |
| Sub-Category      | Specific type of product within the category (e.g., Printers, Electronic Games).|
| PaymentMode       | Payment method used (e.g., UPI, Credit Card).                                   |
| Order Date        | Date when the order was placed.                                                 |
| CustomerName      | Name of the customer who placed the order.                                      |
| State             | State where the order was delivered.                                            |


This provides a **comprehensive transaction-level view**, ideal for sales and regional trend analysis.

---

## 🔧 Tech Stack

- **PostgreSQL**  
- **pgAdmin**  
- **SQL**

---

## 🔁 Project Workflow

### ✅ Step 1:  
**Import** the dataset into **pgAdmin**.

### ✅ Step 2:  
Start  **SQL analysis** to answer the business questions using queries that extract insights from the sales data.

---

## Outcome Summary Aligned with Project Objective & Problem Statement

The project aimed to uncover data-driven insights for RetailX by addressing inefficiencies in sales reporting, product performance tracking, and customer segmentation. Through structured analysis, we achieved the following:

- **Product Performance:**  
  - `Tables` topped sales volume (1,306 units), while `Markers` led in total revenue (£627K), showing that high-margin items may not always be top-selling by quantity.  
  - Sub-category performance shows balanced demand across both office and home product segments, supporting informed inventory and bundling strategies.

- 🌍 **Regional & Geographic Trends:**  
  - High revenue concentrations were found in `New York (£1.13M)`, `Florida (£1.09M)`, and `California (£1.08M)`, with `Orlando (£452K)` leading all cities.  
  - These insights emphasize the need for geo-targeted marketing and localized fulfillment.

- 👥 **Customer Insights:**  
  - The top customers (e.g., `Cory Evans` – £28.4K) significantly influence revenue, while recurring buyers (e.g., `Scott Lewis`, `Christina Davis`) show high retention potential.  
  - Tailored engagement strategies such as loyalty programs and personalized offers can further boost lifetime value.

- **Time-Based Trends:**  
  - Sales patterns across `2020–2025` show seasonal peaks in `May (~£580K)` and `December (~£670K)`, with `December 2022` alone bringing in **£204,413**.  
  - These findings validate the need for seasonal campaigns and supply planning aligned with demand spikes.

- **Consumer Behavior:**  
  - `Debit Card` was the most frequent payment method (260 transactions), signaling a customer preference that can inform payment channel reliability and promotions.

##  Conclusion
The analysis delivers actionable insights that address the original problem statement—helping RetailX overcome fragmented data and gain visibility into **what sells, who buys, when, and where**. These insights lay the foundation for data-driven decisions in **marketing**, **inventory planning**, and **customer engagement**.
