#!/usr/bin/env python
# coding: utf-8

# # Batch ETL Pipeline and Task Scheduler Orchestration (On-Premise)

# ## Notes

# In[9]:


from IPython.display import Image, display
display(Image("images/de_process_flow.png"))
display(Image("images/staging.png"))
display(Image("images/schematypes.png"))
display(Image("images/datapipeline.png"))


# ## Case Study - On-Premises -Azure Blob Storage

# ### Dataset & Data Architecture
# - [Azure: zikologistics dataset]("https://drive.google.com/file/d/1O-wEH1FYyeAE3hygOUjmso0GgaOIsAVm/view?usp=drive_link")
# - [Spark: nugabank dataset]("https://drive.google.com/file/d/1bHbh1M5cExdzwhzSaHNUVz3s9Jxtm2E5/view?usp=sharing)

# In[14]:


display(Image("images/data architecture.png"))


# ### Libraries and Dependencies

# In[ ]:


# Import necessary dependencies
import pandas as pd
import gdown
import tempfile
import shutil
import os
from dotenv import load_dotenv
load_dotenv()
#import psycopg2
from psycopg2 import sql
from sqlalchemy import create_engine, Column, Integer, String, Float, BigInteger, Text, TIMESTAMP, ForeignKey, MetaData, Table

import io
from azure.storage.blob import BlobServiceClient, BlobClient, ContainerClient


# ### Extraction Layer
# - Download CSV content from Google Drive into a temporary file path 

# In[4]:


with tempfile.NamedTemporaryFile(delete=False, suffix=".csv") as tmp:
    temp_path = tmp.name

file_id = "1O-wEH1FYyeAE3hygOUjmso0GgaOIsAVm"
gdown.download(f"https://drive.google.com/uc?id={file_id}", temp_path, quiet=False)
#shutil.copy(temp_path, "ziko_logistics.csv") # Save a permanent copy of the downloaded CSV

ziko_df = pd.read_csv(temp_path)
display(ziko_df.head(3))

# Clean up the temporary file
os.remove(temp_path)


# ### Transformation Layer

# In[5]:


display(ziko_df.info())
print(ziko_df.columns)


# #### Data Modelling with storage into Azure

# Fact Table: fact_sales
# Captures transactional and measurable data:
# | Column                  | Type    | Notes                            |
# | ----------------------- | ------- | -------------------------------- |
# | `transaction_id`        | PK      | Primary key                      |
# | `date`                  | FK      | → `dim_date.date`                |
# | `customer_id`           | FK      | → `dim_customer.customer_id`     |
# | `product_id`            | FK      | → `dim_product.product_id`       |
# | `warehouse_code`        | FK      | → `dim_warehouse.warehouse_code` |
# | `payment_type`          | FK      | → `dim_payment.payment_type`     |
# | `quantity`              | Integer | Metric                           |
# | `unit_price`            | Float   | Metric                           |
# | `total_cost`            | Float   | Metric                           |
# | `discount_rate`         | Float   | Metric                           |
# | `taxable`               | Boolean | Metric                           |
# | `customer_satisfaction` | String  | Could be moved to attribute dim  |
# | `item_returned`         | Boolean | Metric                           |
# 
# 
# Dimension Tables
# dim_customer
# | Column             | Type   |
# | ------------------ | ------ |
# | `customer_id`      | PK     |
# | `customer_name`    | String |
# | `customer_email`   | String |
# | `customer_phone`   | String |
# | `customer_address` | String |
# | `region`           | String |
# | `country`          | String |
# 
# dim_product
# | Column               | Type   |
# | -------------------- | ------ |
# | `product_id`         | PK     |
# | `product_list_title` | String |
# 
# dim_warehouse
# | Column            | Type   |
# | ----------------- | ------ |
# | `warehouse_code`  | PK     |
# | `ship_mode`       | String |
# | `delivery_status` | String |
# 
# dim_payment
# | Column         | Type |
# | -------------- | ---- |
# | `payment_type` | PK   |
# 
# dim_order
# | Column           | Type               |
# | ---------------- | ------------------ |
# | `transaction_id` | PK (or FK to fact) |
# | `sales_channel`  | String             |
# | `order_priority` | String             |
# | `return_reason`  | String             |
# 
# Optional — if sales channel and priority have analytical value.
# 
# dim_date
# | Column                                        | Type |
# | --------------------------------------------- | ---- |
# | `date`                                        | PK   |
# | Add year, month, etc. for hierarchy if needed |      |
# 

# #### Cleaning and transformation

# In[6]:


# Step 1: Handle missing values
df = ziko_df.copy()
df.fillna({
    "Unit_Price": df["Unit_Price"].mean(),
    "Total_Cost": df["Total_Cost"].mean(),
    "Discount_Rate": 0.0,
    "Return_Reason": "Unknown"
}, inplace=True)
df.head(3)


# In[7]:


# Step 2: Normalize columns — cast to correct types
df["Date"] = pd.to_datetime(df["Date"], errors='coerce')

# Convert all object-type columns to lowercase strings (where applicable)
df = df.apply(lambda col: col.str.lower() if col.dtype == "object" else col)

# clean column names to snake_case if needed
df.columns = df.columns.str.lower().str.replace(" ", "_")
df.head(3)  


# #### Step 3: Create Dimension Tables

# In[8]:


dim_customer = df[[
    "customer_id", "customer_name", "customer_phone",
    "customer_email", "customer_address", "region", "country"
]].drop_duplicates().reset_index(drop=True)

dim_customer.head(3)


# In[9]:


dim_product = df[[
    "product_id", "product_list_title"
]].drop_duplicates().reset_index(drop=True)

dim_product.head(3)


# In[10]:


dim_warehouse = df[[
    "warehouse_code", "ship_mode", "delivery_status"
]].drop_duplicates().reset_index(drop=True)
dim_warehouse["warehouse_id"] = dim_warehouse.index + 1
dim_warehouse = dim_warehouse[[
    "warehouse_id", "warehouse_code", "ship_mode", "delivery_status"
]]

dim_warehouse.head(3)   


# In[11]:


dim_payment = df[["payment_type"]].drop_duplicates().reset_index(drop=True)
dim_payment["payment_id"] = dim_payment.index + 1
dim_payment = dim_payment[["payment_id", "payment_type"]]         
dim_payment.head(3)     


# In[12]:


dim_order = df[["transaction_id", "sales_channel", "order_priority", "return_reason"]].drop_duplicates().reset_index(drop=True)
dim_order = dim_order[[
    "transaction_id", "sales_channel", "order_priority", "return_reason"]]  
dim_order.head(3)


# In[13]:


dim_date = df[["date"]].drop_duplicates().reset_index(drop=True)
dim_date["date_id"] = dim_date.index + 1
dim_date = dim_date[["date_id", "date"]]
dim_date.head(3)  


# #### Step 4: Build Fact Table

# In[14]:


fact_sales = df.merge(dim_payment, on="payment_type", how="left") \
               .merge(dim_warehouse, on="warehouse_code", how="left") \
               .merge(dim_date, on="date", how="left")

fact_sales = fact_sales[[
    "transaction_id", "date_id", "customer_id", "product_id", "warehouse_id",
    "quantity", "unit_price", "total_cost", "discount_rate",
    "taxable", "payment_id", "customer_satisfaction", "item_returned"
]]

# Optional: clean column names to snake_case if needed
fact_sales.columns = fact_sales.columns.str.lower().str.replace(" ", "_")
display(fact_sales.head(3))

# Display output
print("✅ Data cleaned and transformed.")


# ### Loading Layer using columnar packages

# #### Storage in Azure Blob Storage

# visit azure portal:  
# - create subscription
# - create resource group (RG) in subscription
# - create storage account in RG (LRS)
# - in storage account, create a container
# - get acces key connection string and container name

# #### Set up a connection to azure blob storage

# In[15]:


# Set up a connection to azure blob storage
connection_string = os.getenv("ZIKO_AZURE_STORAGE_CONNECTION_STRING")
blob_service_client = BlobServiceClient.from_connection_string(connection_string)

container_name = os.getenv("ziko_container_name")
container_client = blob_service_client.get_container_client(container_name)


# #### Load data into azure storage blob uing parquet file

# In[18]:


# create a function to upload a DataFrame to Azure Blob Storage

def upload_dataframe_to_blob(df, container_name, blob_name):
    # Convert DataFrame to Parquet format
    buffer = io.BytesIO()
    df.to_parquet(buffer, index=False)
    buffer.seek(0)  # Reset buffer position to the beginning

    # Upload to Azure Blob Storage
    blob_client = container_client.get_blob_client(blob_name)
    blob_client.upload_blob(buffer, overwrite=True)
    print(f"=== DataFrame uploaded to {container_name}/{blob_name}")     


# In[20]:


# Load the DataFrames to Azure Blob Storage
upload_dataframe_to_blob(dim_customer, container_name, "rawdata/dim_customer.parquet")
upload_dataframe_to_blob(dim_product, container_name, "rawdata/dim_product.parquet")
upload_dataframe_to_blob(dim_warehouse, container_name, "rawdata/dim_warehouse.parquet")
upload_dataframe_to_blob(dim_payment, container_name, "rawdata/dim_payment.parquet")
upload_dataframe_to_blob(dim_order, container_name, "rawdata/dim_order.parquet")
upload_dataframe_to_blob(dim_date, container_name, "rawdata/dim_date.parquet")
upload_dataframe_to_blob(fact_sales, container_name, "rawdata/fact_sales.parquet")

# Display confirmation message
print(f"\n ✅ All DataFrames uploaded successfully to Azure Blob Storage Container: {container_name}.")


# ### Create a ETL Pipeline
