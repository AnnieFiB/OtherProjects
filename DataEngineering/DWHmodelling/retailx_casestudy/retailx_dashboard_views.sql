CREATE OR REPLACE VIEW retailx_oltp.view_retailx_query_1 AS
-- RetailX Sales Insights: SQL Dashboard Queries

-- 1. Top 5 selling products by quantity;

CREATE OR REPLACE VIEW retailx_oltp.top_5_selling_products_by_quantity AS
SELECT sub_category, SUM(quantity) AS total_quantity
FROM retailx_oltp.orders
GROUP BY sub_category
ORDER BY total_quantity DESC
LIMIT 5;

-- 2. Total revenue per product;

CREATE OR REPLACE VIEW retailx_oltp.total_revenue_per_product AS
SELECT sub_category, SUM(amount) AS total_revenue
FROM retailx_oltp.orders
GROUP BY sub_category
ORDER BY total_revenue DESC;

-- 3. Sales performance by state;

CREATE OR REPLACE VIEW retailx_oltp.sales_performance_by_state AS
SELECT state, SUM(amount) AS total_sales, COUNT(order_sk) AS total_orders
FROM retailx_oltp.orders
GROUP BY state
ORDER BY total_sales DESC;

-- 4. Customers contributing most to revenue;

CREATE OR REPLACE VIEW retailx_oltp.top_customers_by_revenue AS
SELECT order_sk,first_name || ' ' || last_name AS customer_name, SUM(amount) AS total_revenue
FROM retailx_oltp.orders
GROUP BY order_sk,customer_name
ORDER BY total_revenue DESC
LIMIT 10;

-- 5. Month with the most sales;

CREATE OR REPLACE VIEW retailx_oltp.top_sales_year_month AS
SELECT order_date,year_month AS sales_trend,sub_category, SUM(amount) AS total_sales
FROM retailx_oltp.orders
GROUP BY order_date,year_month, sub_category
ORDER BY year_month ;

CREATE OR REPLACE VIEW retailx_oltp.top_sales_month AS
SELECT order_date,
    TO_CHAR(order_date, 'Month') AS month_name,
    EXTRACT(MONTH FROM order_date) AS month_number,
    SUM(amount) AS total_sales
FROM retailx_oltp.orders
GROUP BY order_date,month_name, month_number
ORDER BY month_number;

CREATE OR REPLACE VIEW retailx_oltp.best_sales_month AS
SELECT DATE_TRUNC('month', year_month) AS sales_month, SUM(amount) AS total_sales
FROM retailx_oltp.orders
GROUP BY sales_month
ORDER BY total_sales DESC
LIMIT 1;

-- 6. Top 5 recurring customers;

CREATE OR REPLACE VIEW retailx_oltp.top_5_recurring_customers AS
SELECT first_name || ' ' || last_name AS customer_name, COUNT(*) AS order_count
FROM retailx_oltp.orders
GROUP BY customer_name
ORDER BY order_count DESC
LIMIT 5;

-- 7. Best performing product categories (top 5 by revenue);

CREATE OR REPLACE VIEW retailx_oltp.top_5_product_categories AS
SELECT category, SUM(amount) AS total_revenue
FROM retailx_oltp.orders
GROUP BY category
ORDER BY total_revenue DESC
LIMIT 5;

-- 8. Best performing sub-categories (top 5 by revenue);

CREATE OR REPLACE VIEW retailx_oltp.top_5_product_subcategories AS
SELECT sub_category, SUM(amount) AS total_revenue
FROM retailx_oltp.orders
GROUP BY sub_category
ORDER BY total_revenue DESC
LIMIT 5;

-- 9. Most frequent payment method;

CREATE OR REPLACE VIEW retailx_oltp.most_frequent_payment_method AS
SELECT payment_mode, COUNT(*) AS usage_count
FROM retailx_oltp.orders
GROUP BY payment_mode
ORDER BY usage_count DESC
LIMIT 1;

-- 10. City with the highest sales;

CREATE OR REPLACE VIEW retailx_oltp.city_with_highest_sales AS
SELECT city, SUM(amount) AS total_sales
FROM retailx_oltp.orders
GROUP BY city
ORDER BY total_sales DESC
LIMIT 1;