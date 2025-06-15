# ====================================================================
# # Batch ETL Pipeline and Task Scheduler Orchestration (On-Premise)
# ## Case Study - On-Premises -Azure Blob Storage
# ====================================================================
# ### Libraries and Dependencies

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

# ====================
# ### Extraction Layer
# ====================
print("Extracting ziko raw data (csv) from google drive")

# - Download CSV content from Google Drive into a temporary file path 
with tempfile.NamedTemporaryFile(delete=False, suffix=".csv") as tmp:
    temp_path = tmp.name

file_id = "1O-wEH1FYyeAE3hygOUjmso0GgaOIsAVm"
gdown.download(f"https://drive.google.com/uc?id={file_id}", temp_path, quiet=False)
ziko_df = pd.read_csv(temp_path)
os.remove(temp_path)

# ==========================
# ### Transformation Layer
# ==========================
print("\n Cleaning and transforming rawdata into a normalised star schema")

# Step 1: Handle missing values
df = ziko_df.copy()
df.fillna({
    "Unit_Price": df["Unit_Price"].mean(),
    "Total_Cost": df["Total_Cost"].mean(),
    "Discount_Rate": 0.0,
    "Return_Reason": "Unknown"
}, inplace=True)
df.head(3)

# Step 2: Normalize columns — cast to correct types
df["Date"] = pd.to_datetime(df["Date"], errors='coerce')


# Step 3: Create Dimension Tables
dim_customer = df[[
    "Customer_ID", "Customer_Name", "Customer_Phone", "Customer_Email",
    "Customer_Address", "Region", "Country"
]].drop_duplicates(subset=["Customer_ID"]).reset_index(drop=True)

dim_product = df[[
    "Product_ID", "Product_List_Title"
]].drop_duplicates(subset=["Product_ID"]).reset_index(drop=True)

dim_warehouse = df[[
    "Warehouse_Code", "Ship_Mode", "Delivery_Status"]].drop_duplicates(subset=["Warehouse_Code"]).reset_index(drop=True)

dim_warehouse["Warehouse_ID"] = dim_warehouse.index + 1
dim_warehouse = dim_warehouse[["Warehouse_ID", "Warehouse_Code", "Ship_Mode", "Delivery_Status"]]

dim_payment = df[["Payment_Type"]].drop_duplicates().reset_index(drop=True)
dim_payment["Payment_ID"] = dim_payment.index + 1
dim_payment = dim_payment[["Payment_ID", "Payment_Type"]]

dim_order = df[[
    "Transaction_ID", "Sales_Channel", "Order_Priority", "Taxable",
    "Item_Returned", "Return_Reason", "Customer_Satisfaction"
]].drop_duplicates(subset=["Transaction_ID"]).reset_index(drop=True)

dim_date = df[["Date"]].drop_duplicates().reset_index(drop=True)
dim_date["Date_ID"] = dim_date.index + 1
dim_date = dim_date[["Date_ID", "Date"]]

# Derived Metrics
df["Revenue"] = df["Quantity"] * df["Unit_Price"]
df["Discount_Amount"] = df["Revenue"] * df["Discount_Rate"]
df["Net_Revenue"] = df["Revenue"] - df["Discount_Amount"]
df["Profit"] = df["Net_Revenue"] - df["Total_Cost"]
df[["Revenue", "Discount_Amount", "Net_Revenue", "Unit_Price", "Total_Cost", "Profit"]] = \
    df[["Revenue", "Discount_Amount", "Net_Revenue","Unit_Price" , "Total_Cost", "Profit"]].round(2)


# Step 4: Build Fact Table
fact_sales = df \
    .merge(dim_customer[["Customer_ID"]], on="Customer_ID", how="left") \
    .merge(dim_product[["Product_ID"]], on="Product_ID", how="left") \
    .merge(dim_warehouse, on=["Warehouse_Code"], how="left") \
    .merge(dim_payment, on="Payment_Type", how="left") \
    .merge(dim_order[["Transaction_ID"]], on="Transaction_ID", how="left") \
    .merge(dim_date, on="Date", how="left")

fact_sales = fact_sales[[  
    "Transaction_ID",      # (natural key)
    "Date_ID",
    "Customer_ID",
    "Product_ID",
    "Warehouse_ID",
    "Payment_ID",

    # Measures
    "Quantity",
    "Unit_Price",
    "Total_Cost",
    "Revenue",
    "Discount_Amount",
    "Net_Revenue",        
    "Profit"           
]]

print("\n Ziko raw data cleaned and normalised into: six (6) dimensions and one (1) fact table")

# ==================================
# ## Loading Layer
# ### Storage in Azure Blob Storage
# ==================================
print("\n Connecting to azure Blob Storage")

# Set up a connection to azure blob storage
connection_string = os.getenv("ZIKO_AZURE_STORAGE_CONNECTION_STRING")
blob_service_client = BlobServiceClient.from_connection_string(connection_string)

container_name = os.getenv("ziko_container_name")
container_client = blob_service_client.get_container_client(container_name)
print('connection succesful')


# create a function to upload a DataFrame to Azure Blob Storage
print("\n Creating a data upload function and loading normalised tables to azure blob storage")
def upload_dataframe_to_blob(df, container_name, blob_name, table_name):
    # Convert DataFrame to Parquet format
    buffer = io.BytesIO()
    df.to_parquet(buffer, index=False)
    buffer.seek(0)  # Reset buffer position to the beginning

    # Upload to Azure Blob Storage
    blob_client = container_client.get_blob_client(blob_name)
    blob_client.upload_blob(buffer, overwrite=True)
    print(f"✅ Table '{table_name}' uploaded to '{container_name}/{blob_name}' with {df.shape[0]} rows.")   


# Load the DataFrames to Azure Blob Storage
upload_dataframe_to_blob(dim_customer, container_name, "rawdata/dim_customer.parquet", "dim_customer")
upload_dataframe_to_blob(dim_product, container_name, "rawdata/dim_product.parquet", "dim_product")
upload_dataframe_to_blob(dim_warehouse, container_name, "rawdata/dim_warehouse.parquet", "dim_warehouse")
upload_dataframe_to_blob(dim_payment, container_name, "rawdata/dim_payment.parquet", "dim_payment")
upload_dataframe_to_blob(dim_order, container_name, "rawdata/dim_order.parquet", "dim_order")
upload_dataframe_to_blob(dim_date, container_name, "rawdata/dim_date.parquet", "dim_date")
upload_dataframe_to_blob(fact_sales, container_name, "rawdata/fact_sales.parquet", "fact_sales")

# Display confirmation message
print(f"\n ✅ All DataFrames uploaded successfully to Azure Blob Storage Container: {container_name}.")
