# Dynamic ETL Pipeline for OLTP and OLAP Schemas

## Overview
This ETL pipeline dynamically processes raw data into a normalized OLTP schema, and optionally transforms it into an OLAP schema with support for dimensions, fact tables, surrogate keys, and indexing.

## Features
- **Automatic Primary Key Inference**
- **Date Dimension Generation and Mapping**
- **Surrogate Key Creation for OLAP**
- **Config-Driven Dimension and Fact Table Definitions**
- **Support for Foreign Key Constraints**
- **Automatic Indexing for *_sk and *_date_id columns**
- **Calculated Fields for OLAP Fact Tables**

---

## ETL Pipeline Structure

### 1. Extraction Phase

The ETL process begins with dynamic source detection and configuration-driven routing. It supports multiple sources defined in a central configuration.

#### Supported Input Types
- **Local filesystem paths** - CSV, Excel
- APIs (future support)
- Multiple datasets via `dataset_key`
- **Public URLs**
- **Google Drive (shared/public file links)**

#### Example Source Configuration

```python
"datasets": {
  "yanki_ecom": {
    "path": "1NWER....",# Google Drive file ID
    "type": "gdrive",
  },
  "zulo_bank": {
    "path": "data/zulo_bank.csv",
    "type": "file",
    "delimiter": ",",
    "date_columns": ["start_date", "end_date", "transaction_date"]
  },
  "external_transactions": {
    "source_type": "url",
    "url": "https://example.com/data/transactions.xlsx",
    "format": "excel",
    "sheet_name": "Sheet1"
  }
}
```

#### Extraction Workflow

1. The pipeline accepts a `dataset_key` (e.g., `"zulo_bank"`).
2. The source config maps this key to:
   - the correct file
   - the file type
   - any pre-load options (date parsing, column renaming)
3. The helper function `process_data_source()`:
   - reads the raw file using the right parser
   - returns a clean DataFrame for transformation

---

##### Behavior

- The ETL pipeline dynamically detects if a dataset is stored remotely or locally.
- If `source_type = url`, the file is downloaded to a temporary local path before being processed.
- Parsing options like `delimiter`, `sheet_name`, `encoding`, and `date_columns` are respected regardless of file origin.

This flexibility makes the ETL pipeline cloud-ready and adaptable for external data ingestion.

---

## Example Pipeline Trigger

```python
result = select_and_load_source(ETL_CONFIG)
raw_df = result["raw_df"]
dataset_key = result["selected_source"]
cfg = ETL_CONFIG["data_sources"][dataset_key]
```

If `raw_df` is not passed, it is loaded using the `dataset_key` and the corresponding entry in the config.

---

### 2. OLTP Transformation
- Cleans and preprocesses raw data.
- Infers or generates primary keys based on `critical_columns` in config.
- Optionally generates `date_dim` and maps date IDs if `olap` is defined in `cfg["pipelines"]`.
- Normalizes flat data into 3NF OLTP tables.
- Applies or infers foreign key relationships.
- Generates surrogate keys (`*_sk`) if OLAP is enabled.

### 3. OLTP Schema Creation and Load
- Creates OLTP tables in topological FK-safe order.
- Adds PK and FK constraints.
- Upserts data safely using temporary tables.

### 4. OLAP Pipeline (Optional)
- Triggered only if `olap` is defined in `cfg["pipelines"]`.
- Calculates any derived fields defined as expressions in the OLAP `fact` config.
- Copies date dimension to OLAP schema if exists.
- Builds and loads dimensions and fact tables.
- Adds foreign key constraints to OLAP fact tables.
- Indexes SK/date ID columns in both OLAP and OLTP.

---

## Config Structure

### OLTP Section
```json
"oltp": {
  "schema": "zulo_oltp",
  "table_mapping": { ... },
  "critical_columns": { ... },
  "foreign_keys": { ... },
  "date_columns": ["start_date", "end_date"],
  "date_mapping": { ... }
}
```

### OLAP Section
```json
"olap": {
  "schema": "zulo_olap",
  "dimensions": {
    "dim_customers": ["customer_id", "first_name", "last_name"],
    ...
  },
  "facts": {
    "fact_loans": [
      "loans_sk", "customers_sk", "start_date_id",
      {"interest": "loan_amount * interest_rate"}
    ]
  }
},
"olap_foreign_keys": {
  "fact_loans": [
    ["start_date_id", "date_dim", "date_id"]
  ]
}
```

---

## Output Tables

### OLTP Schema (`zulo_oltp`)
- `customers(customer_id, ...)`
- `accounts(account_id, customer_id, ...)`
- `loans(loan_id, customer_id, ...)`
- `transactions(transaction_id, account_id, ...)`
- `date_dim(date_id, full_date, ...)`

### OLAP Schema (`zulo_olap`)
- `dim_customers(customers_sk, customer_id, ...)`
- `dim_accounts(accounts_sk, account_id, ...)`
- `dim_loans(loans_sk, loan_id, ...)`
- `dim_transactions(transactions_sk, transaction_id, ...)`
- `fact_loans(loans_sk, customers_sk, start_date_id, end_date_id, interest, ...)`
- `fact_transactions(transactions_sk, accounts_sk, transaction_date_id, ...)`
- `date_dim(date_id, full_date, ...)`

---

## Entity Relationship Diagram (ERD)

A sample ERD for OLAP schema:

```
          +----------------+
          |  date_dim      |
          |----------------|
          | date_id (PK)   |
          | full_date      |
          +----------------+
                 ^
                 |
+----------------+-----------------+
|                                  |
|                                  |
v                                  v
fact_loans                   fact_transactions
------------                -------------------
loans_sk (PK)               transactions_sk (PK)
customers_sk (FK)           accounts_sk (FK)
start_date_id (FK)          transaction_date_id (FK)
end_date_id (FK)            amount
interest

^
|
dim_loans, dim_customers, dim_accounts, dim_transactions
```

## Requirements
- PostgreSQL database connection
- Pandas, psycopg2, numpy, and optionally seaborn for exploration

---

## Author
Anthonia Fisuyi
---

