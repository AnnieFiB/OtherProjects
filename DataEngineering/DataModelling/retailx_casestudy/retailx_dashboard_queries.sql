
-- RetailX Sales Insights: SQL Dashboard Queries

-- 1. Top 5 selling products by quantity
SELECT sub_category, SUM(quantity) AS total_quantity
FROM retailx_oltp.orders
GROUP BY sub_category
ORDER BY total_quantity DESC
LIMIT 5;

-- 2. Total revenue per product
SELECT sub_category, SUM(amount) AS total_revenue
FROM retailx_oltp.orders
GROUP BY sub_category
ORDER BY total_revenue DESC;

-- 3. Sales performance by state
SELECT state, SUM(amount) AS total_sales, COUNT(order_sk) AS total_orders
FROM retailx_oltp.orders
GROUP BY state
ORDER BY total_sales DESC;

-- 4. Customers contributing most to revenue
SELECT first_name || ' ' || last_name AS customer_name, SUM(amount) AS total_revenue
FROM retailx_oltp.orders
GROUP BY customer_name
ORDER BY total_revenue DESC
LIMIT 10;

-- 5. Month with the most sales
SELECT year_month AS sales_trend, SUM(amount) AS total_sales
FROM retailx_oltp.orders
GROUP BY year_month
ORDER BY year_month ;

SELECT 
    TO_CHAR(order_date, 'Month') AS month_name,
    EXTRACT(MONTH FROM order_date) AS month_number,
    SUM(amount) AS total_sales
FROM retailx_oltp.orders
GROUP BY month_name, month_number
ORDER BY month_number;


SELECT DATE_TRUNC('month', year_month) AS sales_month, SUM(amount) AS total_sales
FROM retailx_oltp.orders
GROUP BY sales_month
ORDER BY total_sales DESC
LIMIT 1;

-- 6. Top 5 recurring customers
SELECT first_name || ' ' || last_name AS customer_name, COUNT(*) AS order_count
FROM retailx_oltp.orders
GROUP BY customer_name
ORDER BY order_count DESC
LIMIT 5;

-- 7. Best performing product categories (top 5 by revenue)
SELECT category, SUM(amount) AS total_revenue
FROM retailx_oltp.orders
GROUP BY category
ORDER BY total_revenue DESC
LIMIT 5;

-- 8. Best performing sub-categories (top 5 by revenue)
SELECT sub_category, SUM(amount) AS total_revenue
FROM retailx_oltp.orders
GROUP BY sub_category
ORDER BY total_revenue DESC
LIMIT 5;

-- 9. Most frequent payment method
SELECT payment_mode, COUNT(*) AS usage_count
FROM retailx_oltp.orders
GROUP BY payment_mode
ORDER BY usage_count DESC
LIMIT 1;

-- 10. City with the highest sales
SELECT city, SUM(amount) AS total_sales
FROM retailx_oltp.orders
GROUP BY city
ORDER BY total_sales DESC
LIMIT 1;
