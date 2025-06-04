
-- SCHEMA SETUP

CREATE SCHEMA IF NOT EXISTS olap;

-- OLAP DIMENSION TABLES

CREATE TABLE IF NOT EXISTS olap.dim_customer (
    customer_sk INT PRIMARY KEY,
    customer_name VARCHAR(100)
);

CREATE TABLE IF NOT EXISTS olap.dim_product (
    product_sk INT PRIMARY KEY,
    category VARCHAR(50),
    sub_category VARCHAR(50)
);

CREATE TABLE IF NOT EXISTS olap.dim_payment (
    payment_sk INT PRIMARY KEY,
    payment_mode VARCHAR(50)
);

CREATE TABLE IF NOT EXISTS olap.dim_location (
    location_sk INT PRIMARY KEY,
    city VARCHAR(100),
    state VARCHAR(100)
);

CREATE TABLE IF NOT EXISTS olap.dim_date (
    date_id SERIAL PRIMARY KEY,
    order_date DATE,
    day INT,
    month INT,
    quarter INT,
    year INT,
    day_of_week VARCHAR(15)
);

-- OLAP FACT TABLE

CREATE TABLE IF NOT EXISTS olap.fact_order_sales (
    order_sk INT PRIMARY KEY,
    customer_sk INT REFERENCES olap.dim_customer(customer_sk),
    product_sk INT REFERENCES olap.dim_product(product_sk),
    payment_sk INT REFERENCES olap.dim_payment(payment_sk),
    location_sk INT REFERENCES olap.dim_location(location_sk),
    date_id INT REFERENCES olap.dim_date(date_id),
    quantity INT,
    amount NUMERIC(10, 2),
    profit NUMERIC(10, 2)
);

