
import streamlit as st
import pandas as pd
import psycopg2

st.set_page_config(page_title="RetailX Sales Dashboard", layout="wide")
st.title("📊 RetailX Sales Intelligence Dashboard")

@st.cache_resource
def get_connection():
    return psycopg2.connect(
        dbname=st.secrets["DB_NAME"],
        user=st.secrets["DB_USER"],
        password=st.secrets["DB_PASSWORD"],
        host=st.secrets["DB_HOST"],
        port=st.secrets["DB_PORT"]
    )

conn = get_connection()

def run_query(query):
    return pd.read_sql_query(query, conn)

# Main Tabs
main_tabs = st.tabs([
    "Top-Selling Products",
    "Regional Insights",
    "Customer Value",
    "Seasonal Trends",
    "Payment Patterns",
])

with main_tabs[0]:
    st.subheader('Top 5 Selling Products by Quantity')
    query = '''SELECT sub_category, SUM(quantity) AS total_quantity
        FROM retailx_oltp.orders
        GROUP BY sub_category
        ORDER BY total_quantity DESC
        LIMIT 5;'''
    df = run_query(query)
    st.dataframe(df)
    if df.shape[1] >= 2:
        x_col = df.columns[0]
        y_col = df.columns[1]
        st.bar_chart(data=df, x=x_col, y=y_col, use_container_width=True)
    st.subheader('Total Revenue Per Product')
    query = '''SELECT sub_category, SUM(amount) AS total_revenue
        FROM retailx_oltp.orders
        GROUP BY sub_category
        ORDER BY total_revenue DESC;'''
    df = run_query(query)
    st.dataframe(df)
    if df.shape[1] >= 2:
        x_col = df.columns[0]
        y_col = df.columns[1]
        st.bar_chart(data=df, x=x_col, y=y_col, use_container_width=True)
    st.subheader('Top 5 Product Subcategories')
    query = '''SELECT sub_category, SUM(amount) AS total_revenue
        FROM retailx_oltp.orders
        GROUP BY sub_category
        ORDER BY total_revenue DESC
        LIMIT 5;'''
    df = run_query(query)
    st.dataframe(df)
    if df.shape[1] >= 2:
        x_col = df.columns[0]
        y_col = df.columns[1]
        st.bar_chart(data=df, x=x_col, y=y_col, use_container_width=True)

with main_tabs[1]:
    st.subheader('Sales Performance by State')
    query = '''SELECT state, SUM(amount) AS total_sales, COUNT(order_sk) AS total_orders
        FROM retailx_oltp.orders
        GROUP BY state
        ORDER BY total_sales DESC;'''
    df = run_query(query)
    st.dataframe(df)
    if df.shape[1] >= 2:
        x_col = df.columns[0]
        y_col = df.columns[1]
        st.bar_chart(data=df, x=x_col, y=y_col, use_container_width=True)
    st.subheader('City with Highest Sales')
    query = '''SELECT city, SUM(amount) AS total_sales
        FROM retailx_oltp.orders
        GROUP BY city
        ORDER BY total_sales DESC
        LIMIT 1;'''
    df = run_query(query)
    st.dataframe(df)
    if df.shape[1] >= 2:
        x_col = df.columns[0]
        y_col = df.columns[1]
        st.bar_chart(data=df, x=x_col, y=y_col, use_container_width=True)

with main_tabs[2]:
    st.subheader('Top Customers by Revenue')
    query = '''SELECT first_name || ' ' || last_name AS customer_name, SUM(amount) AS total_revenue
        FROM retailx_oltp.orders
        GROUP BY customer_name
        ORDER BY total_revenue DESC
        LIMIT 10;'''
    df = run_query(query)
    st.dataframe(df)
    if df.shape[1] >= 2:
        x_col = df.columns[0]
        y_col = df.columns[1]
        st.bar_chart(data=df, x=x_col, y=y_col, use_container_width=True)
    st.subheader('Top 5 Recurring Customers')
    query = '''SELECT first_name || ' ' || last_name AS customer_name, COUNT(*) AS order_count
        FROM retailx_oltp.orders
        GROUP BY customer_name
        ORDER BY order_count DESC
        LIMIT 5;'''
    df = run_query(query)
    st.dataframe(df)
    if df.shape[1] >= 2:
        x_col = df.columns[0]
        y_col = df.columns[1]
        st.bar_chart(data=df, x=x_col, y=y_col, use_container_width=True)

with main_tabs[3]:
    st.subheader('Monthly Sales Trend')
    query = '''SELECT DATE_TRUNC('month', order_date) AS sales_month, SUM(amount) AS total_sales
        FROM retailx_oltp.orders
        GROUP BY sales_month
        ORDER BY total_sales DESC
        LIMIT 1;'''
    df = run_query(query)
    st.dataframe(df)
    if df.shape[1] >= 2:
        x_col = df.columns[0]
        y_col = df.columns[1]
        st.bar_chart(data=df, x=x_col, y=y_col, use_container_width=True)
    st.subheader('Year-Month Trend')
    query = '''SELECT TO_CHAR(order_date, 'YYYY-MM') AS sales_month, SUM(amount) AS total_sales
        FROM retailx_oltp.orders
        GROUP BY sales_month
        ORDER BY sales_month;'''
    df = run_query(query)
    st.dataframe(df)
    if df.shape[1] >= 2:
        x_col = df.columns[0]
        y_col = df.columns[1]
        st.bar_chart(data=df, x=x_col, y=y_col, use_container_width=True)

with main_tabs[4]:
    st.subheader('Most Frequent Payment Method')
    query = '''SELECT payment_mode, COUNT(*) AS usage_count
        FROM retailx_oltp.orders
        GROUP BY payment_mode
        ORDER BY usage_count DESC
        LIMIT 1;'''
    df = run_query(query)
    st.dataframe(df)
    if df.shape[1] >= 2:
        x_col = df.columns[0]
        y_col = df.columns[1]
        st.bar_chart(data=df, x=x_col, y=y_col, use_container_width=True)

