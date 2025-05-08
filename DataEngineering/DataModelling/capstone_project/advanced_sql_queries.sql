-- ===============================================
-- ADVANCED SQL QUERIES FOR SALES INTELLIGENCE PLATFORM
-- ===============================================

-- 1. Profit & Loss by Category and City
SELECT dp.category, dl.city, SUM(fs.amount) AS total_sales, SUM(fs.profit) AS total_profit
FROM olap.fact_sales fs
JOIN olap.dim_product dp ON fs.product_sk = dp.product_sk
JOIN olap.dim_location dl ON fs.location_sk = dl.location_sk
GROUP BY dp.category, dl.city
ORDER BY total_profit DESC;

-- 2. Repeating Customers with RANK (Full List)
SELECT 
    dc.customer_name,
    COUNT(fs.order_sk) AS order_count,
    RANK() OVER (ORDER BY COUNT(fs.order_sk) DESC) AS order_rank
FROM olap.fact_sales fs
JOIN olap.dim_customer dc ON fs.customer_sk = dc.customer_sk
GROUP BY dc.customer_name
ORDER BY order_rank;

-- 3. Top Customer(s) by Repeat Orders
WITH customer_orders AS (
    SELECT dc.customer_name, COUNT(fs.order_sk) AS order_count
    FROM olap.fact_sales fs
    JOIN olap.dim_customer dc ON fs.customer_sk = dc.customer_sk
    GROUP BY dc.customer_name
),
ranked_orders AS (
    SELECT customer_name, order_count, RANK() OVER (ORDER BY order_count DESC) AS order_rank
    FROM customer_orders
)
SELECT customer_name, order_count
FROM ranked_orders
WHERE order_rank = 1;

-- 4. Top-Performing Sub-Categories by Average Profit
SELECT dp.sub_category, ROUND(AVG(fs.profit), 2) AS avg_profit
FROM olap.fact_sales fs
JOIN olap.dim_product dp ON fs.product_sk = dp.product_sk
GROUP BY dp.sub_category
ORDER BY avg_profit DESC
LIMIT 10;

-- 5. Monthly Sales Trends
SELECT dd.year, dd.month, SUM(fs.amount) AS total_sales
FROM olap.fact_sales fs
JOIN olap.dim_date dd ON fs.date_id = dd.date_id
GROUP BY dd.year, dd.month
ORDER BY dd.year, dd.month;

-- 6. Cities Above 95th Percentile in Revenue
WITH city_sales AS (
    SELECT dl.city, SUM(fs.amount) AS revenue
    FROM olap.fact_sales fs
    JOIN olap.dim_location dl ON fs.location_sk = dl.location_sk
    GROUP BY dl.city
),
threshold AS (
    SELECT PERCENTILE_CONT(0.95) WITHIN GROUP (ORDER BY revenue) AS cutoff FROM city_sales
)
SELECT cs.city, cs.revenue
FROM city_sales cs
JOIN threshold t ON cs.revenue > t.cutoff
ORDER BY cs.revenue DESC;

-- 7. Cities in Top 5% with Revenue Percentage
WITH city_sales AS (
    SELECT dl.city, SUM(fs.amount) AS revenue
    FROM olap.fact_sales fs
    JOIN olap.dim_location dl ON fs.location_sk = dl.location_sk
    GROUP BY dl.city
),
threshold AS (
    SELECT PERCENTILE_CONT(0.95) WITHIN GROUP (ORDER BY revenue) AS cutoff FROM city_sales
),
total AS (
    SELECT SUM(revenue) AS total_revenue FROM city_sales
)
SELECT 
    cs.city,
    cs.revenue,
    ROUND((cs.revenue / t.total_revenue) * 100, 2) AS revenue_pct
FROM city_sales cs
JOIN threshold th ON cs.revenue > th.cutoff
JOIN total t ON TRUE
ORDER BY cs.revenue DESC;

-- 8. Cities Contributing Up to 80% of Total Revenue
WITH city_sales AS (
    SELECT dl.city, SUM(fs.amount) AS revenue
    FROM olap.fact_sales fs
    JOIN olap.dim_location dl ON fs.location_sk = dl.location_sk
    GROUP BY dl.city
),
total AS (
    SELECT SUM(revenue) AS total_revenue FROM city_sales
),
ranked AS (
    SELECT 
        cs.city,
        cs.revenue,
        SUM(cs.revenue) OVER (ORDER BY cs.revenue DESC) AS cumulative_revenue,
        t.total_revenue
    FROM city_sales cs
    CROSS JOIN total t
)
SELECT 
    city,
    revenue,
    ROUND((revenue / total_revenue) * 100, 2) AS revenue_pct,
    ROUND((cumulative_revenue / total_revenue) * 100, 2) AS cumulative_pct
FROM ranked
WHERE (cumulative_revenue / total_revenue) <= 0.80
ORDER BY cumulative_revenue;

-- 9. Monthly average per product
SELECT dp.product_sk, dp.sub_category, ROUND(AVG(fs.amount), 2) AS avg_monthly_sales
FROM olap.fact_sales fs
JOIN olap.dim_product dp ON fs.product_sk = dp.product_sk
JOIN olap.dim_date dd ON fs.date_id = dd.date_id
GROUP BY dp.product_sk, dp.sub_category
ORDER BY avg_monthly_sales DESC;

