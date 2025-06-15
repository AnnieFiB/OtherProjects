# import libraries
from IPython.display import Image, display
from pyspark.sql import SparkSession, DataFrameWriter
from pyspark.sql.functions import monotonically_increasing_id
import os
import psycopg2
from dotenv import load_dotenv
load_dotenv()
from urllib.parse import urlparse
import gdown

## Initiate Spark Session
spark = SparkSession.builder \
    .appName("pyspark batch case study") \
    .getOrCreate()

# ==========================================================================
# Extraction  & Data Cleaning Layer
# ==========================================================================

# Download CSV content from Google Drive into a temporary file path 
'''
with tempfile.NamedTemporaryFile(delete=False, suffix=".csv") as tmp:
    temp_path = tmp.name

file_id = "1WvnxUWIUQcRSXB5so7-PoGPxU9ekSkYb"
gdown.download(f"https://drive.google.com/uc?id={file_id}", temp_path, quiet=False)

shutil.copy(temp_path, "nuga_bank_transactions.csv") # Save a permanent copy of the downloaded CSV

nuga_bank_df = spark.read.csv(temp_path, header=True, inferSchema=True) # Load into Spark

nuga_bank_df.cache()  # Force read and keep in memory
nuga_bank_df.count()  # Triggers actual file read
os.remove(temp_path) # delete temp file
'''
print("\n EXTRACTION LAYER: Reading raw data from csv /google using pyspark")

df = spark.read.csv(r'D:\Portfolio\my_projects\DataEngineering\PySpark\nuga_bank_transactions.csv', header=True, inferSchema=True)
print(f"Data Shape: {df.count()} rows × {len(df.columns)} columns")


print("\n Cleaning Data: Handling Null, Correcting Datatypes and converting to snake_case")

# 1. Handle null
df_clean = df.fillna ({
    "Customer_Name": "Unknown",
    "Customer_Address": "Unknown",
    "Customer_City": "Unknown",
    "Customer_State": "Unknown",
    "Customer_Country": "Unknown",
    "Email": "unknown@example.com",
    "Phone_Number": "Unknown",
    "Company": "Undisclosed",
    "Job_Title": "Unemployed",
    "Currency_Code": "N/A",
    "Category": "N/A",
    "Group": "N/A",
    "Is_Active": "Unknown",
    "Description": "No description provided",
    "Gender": "Unspecified",
    "Marital_Status": "Unspecified",
    "IBAN": "Unknown",
    "Random_Number": -1.0,
    "Credit_Card_Number": 0 
  #  "Last_Updated": "1900-01-01"
})

# Drop last updated null rows
df_clean = df_clean.na.drop(subset=["Last_Updated"])

# Checking for null values again
for col in df_clean.columns:
    null_count = df_clean.filter(df_clean[col].isNull()).count()
    if null_count > 0:
        print(f"{col}: {null_count} nulls")
print(f"All null values handled: \n New Data Shape: {df_clean.count()} rows × {len(df_clean.columns)} columns")

#  Convert all column names to lowercase
from pyspark.sql.functions import col
df_clean = df_clean.select([col(c).alias(c.lower()) for c in df_clean.columns])

# ===================================
# Transformation Layer
# ===================================
print("\n TRANSFORMATION LAYER: Normalising tables to 2NF : transaction, customer, employee and fact_table")

transaction = df_clean \
    .select('transaction_date',  'amount','transaction_type','description') \
    .distinct() \
    .withColumn("transaction_id", monotonically_increasing_id()).cache()
transaction = transaction.select('transaction_id','transaction_date',  'amount','transaction_type','description')

customer = df_clean \
    .select('customer_name', 'customer_address', 'customer_city', 'customer_state', 'customer_country') \
    .distinct() \
    .withColumn("customer_id", monotonically_increasing_id()).cache()
customer =customer.select('customer_id','customer_name', 'customer_address', 'customer_city', 'customer_state','customer_country')

employee = df_clean \
    .select('company', 'job_title', 'email', 'phone_number','gender', 'marital_status') \
    .distinct() \
    .withColumn("employee_id", monotonically_increasing_id()).cache()
employee = employee.select('employee_id','company', 'job_title', 'email', 'phone_number','gender', 'marital_status')

# Build fact_transaction with LEFT JOINs to preserve as many rows as possible
fact_table = df_clean \
      .join(transaction,['transaction_date', 'amount','transaction_type', 'description'],'inner')\
      .join(customer, ['customer_name', 'customer_address', 'customer_city','customer_state', 'customer_country'],'inner')\
      .join(employee, ['company', 'job_title', 'email', 'phone_number','gender', 'marital_status'], 'inner') \
      .select('transaction_id','customer_id', 'employee_id','amount', 'credit_card_number', 'iban','currency_code', 'random_number',\
        'category', 'group','is_active','last_updated' )

print("All 2NF Tables created")
# ===================================
# Loading Layer
# =====================================
print("\n LOADING LAYER: Connecting to db , creating db tables and loading data to created tables")

# connect to db
def get_db_connection(db_url):
    conn = psycopg2.connect(db_url)
    print("\n ✅ Database connected successfully.")
    return conn

db_url = os.getenv("NUGA_BANK") 

# Create a function to create all tables
def create_tables(db_url):
    conn = get_db_connection(db_url)
    cursor = conn.cursor()

    create_table_query = '''
        DROP TABLE IF EXISTS fact_table;
        DROP TABLE IF EXISTS employee;
        DROP TABLE IF EXISTS transaction;
        DROP TABLE IF EXISTS customer;

        CREATE TABLE customer (
            customer_id BIGINT PRIMARY KEY,
            customer_name TEXT NOT NULL,
            customer_address TEXT NOT NULL,
            customer_city TEXT NOT NULL,
            customer_state TEXT NOT NULL,
            customer_country TEXT NOT NULL
        );

        CREATE TABLE transaction (
            transaction_id BIGINT PRIMARY KEY,
            transaction_date TIMESTAMP NOT NULL,
            amount FLOAT NOT NULL,
            transaction_type TEXT NOT NULL,
            description TEXT NOT NULL
        );

        CREATE TABLE employee (
            employee_id BIGINT PRIMARY KEY,
            company TEXT NOT NULL,
            job_title TEXT NOT NULL,
            email TEXT NOT NULL,
            phone_number TEXT NOT NULL,
            gender TEXT NOT NULL,
            marital_status TEXT NOT NULL
        );

        CREATE TABLE fact_table (
            transaction_id BIGINT REFERENCES transaction(transaction_id) ON DELETE CASCADE,
            customer_id BIGINT REFERENCES customer(customer_id) ON DELETE CASCADE,
            employee_id BIGINT REFERENCES employee(employee_id) ON DELETE CASCADE,
            amount FLOAT NOT NULL,
            credit_card_number BIGINT NOT NULL,
            iban TEXT NOT NULL,
            currency_code VARCHAR(10) NOT NULL,
            random_number FLOAT NOT NULL,
            category TEXT NOT NULL,
            "group" TEXT NOT NULL, -- group is a reserved word
            is_active TEXT NOT NULL,
            last_updated TIMESTAMP NOT NULL
        );
    '''

    cursor.execute(create_table_query)
    conn.commit()
    print("✅ All tables created successfully.")
    cursor.close()
    conn.close()

create_tables(db_url)


# Define JDBC Settings
parsed = urlparse(db_url);\
jdbc_url = f"jdbc:postgresql://{parsed.hostname}:{parsed.port}{parsed.path}";\
jdbc_props= {"user": parsed.username, "password": parsed.password, "driver": "org.postgresql.Driver"}
print(f"\n jdbc connection parameters check: {jdbc_url}")

# 3. Create a Write Function
def write_to_postgres(df, table_name, jdbc_url, jdbc_props, mode="append"):
    try:
        print(f"\n 📝 Writing table: {table_name} ({df.count()} rows)...")
        df.write.jdbc(
            url=jdbc_url,
            table=table_name,
            mode=mode,
            properties=jdbc_props
        )
        print(f"✅ Successfully wrote: {table_name}")
        
    except Exception as e:
        print(f"❌ Failed to write table: {table_name}")
        print(f"Error: {str(e)}")


write_to_postgres(transaction,"transaction",jdbc_url,jdbc_props)
write_to_postgres(customer,"customer",jdbc_url,jdbc_props)
write_to_postgres(employee,"employee",jdbc_url,jdbc_props)
write_to_postgres(fact_table,"fact_table",jdbc_url,jdbc_props)

# Display confirmation message
print(f"\n ✅ All DataFrames uploaded successfully to postgres db: {parsed.path}.")
