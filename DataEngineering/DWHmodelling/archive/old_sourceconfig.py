
# config.py
ETL_CONFIG = {
    "data_sources": {
        "zulo_bank": {
            "type": "file",
            #"path": "1DdmNsrdBRLfzBdgtvzvFHZ7ejFpLlpwW",  # Google Drive file ID
            "path": "dataset/zulo_bank.csv",  # flat file path
            "pipelines": ["oltp", "olap"],
            "use_surrogate_keys": True,

            "oltp": {
                "schema": "zulo_oltp",
                "critical_columns": {"loans" : ["loan_id"],  "customers": ["customer_id"],
                                "accounts": ["account_id"],"transactions": ["transaction_id"]},
                "date_columns": ["transaction_date", "opening_date", "start_date", "end_date"],
                "missingvalue_columns": ["TransactionID", "CustomerID", "AccountID", "LoanID"],
                "keys": ["transaction_id", "customer_id", "loan_id", "account_id"],
                "contact_columns": {'email_col': 'Email', 'phone_col': 'Phone'},
                "derived_fields": {"price_col": "interest_rate", "qty_col": "loan_amount", "new_col": "interest"},
                "date_mapping": {
                    "transaction_date": "transaction_date_id",
                    "opening_date": "opening_date_id",
                    "start_date": "start_date_id",
                    "end_date": "end_date_id",
                },
                "split_columns": {"FullName": ["first_name", "last_name"]},
                "table_mapping": {
                       "customers": ["customer_id", "first_name", "last_name", "email", "phone"],
                       "accounts": ["account_id", "customer_id", "account_type", "balance", "opening_date"],
                       "loans": ["loan_id", "customer_id", "loan_amount", "loan_type","start_date", "end_date", "interest_rate"],
                       "transactions": ["transaction_id", "transaction_type", "amount","transaction_date","account_id", "customer_id"],
                }
            },
            "olap": {
                "schema": "zulo_olap",
                "dimensions": {
                    "dim_customers": ["customer_id", "first_name", "last_name", "email", "phone"],
                    "dim_accounts": ["account_id", "account_type", "balance"],
                    "dim_loans": ["loan_id", "loan_type", "loan_amount","interest_rate"],
                    "dim_transactions": ["transaction_id", "transaction_type"]
                },
                "facts": {
                    "fact_transactions": ["transactions_sk","accounts_sk","opening_date_id",
                                           "transaction_date_id", "amount"],
                    "fact_loans": ["loans_sk", "customers_sk", "start_date_id", "end_date_id", "loan_amount",
                                   {"interest": "loan_amount * interest_rate"}  ]
                  }
            },
            "olap_foreign_keys": {
                    "fact_transactions": [
                        ["accounts_sk", "dim_accounts", "accounts_sk"],
                        ["transactions_sk", "dim_transactions", "transactions_sk"],
                        ["transaction_date_id", "date_dim", "date_id"]
                    ],
                    "fact_loans": [
                        ["loans_sk", "dim_loans", "loans_sk"],
                        ["customers_sk", "dim_customers", "customers_sk"],
                        ["start_date_id", "date_dim", "date_id"],
                        ["end_date_id", "date_dim", "date_id"]
                    ]
                }
                  },
        "yanki_ecom": {
            "type": "gdrive",
            "path": "1lPmrM-4EJLfM14E_3pWF2aLl_Y7HsoHa",  # Google Drive file ID
            #path": "dataset/yanki_ecommerce.csv",  # flat file path
            "pipelines": ["oltp"],
            "oltp": {
                "schema": "yanki_oltp",
                "missingvalue_columns": ['Order_ID', 'Customer_ID'],
                "critical_columns": {"orders" : ["order_id"],  "customers": ["customer_id"],
                                     "products": ["product_id"],"payments": ["order_id"]},
                "date_columns": ["Order_Date"],
                "keys" :['Order_ID', 'Product_ID', 'Customer_ID'],
                "contact_columns": {'email_col': 'Email','phone_col': 'Phone_Number'},
                "derived_fields": {"price_col": "Price", "qty_col": "Quantity", "new_col": "Total_Price"},
                "split_columns": {"Customer_Name": ["First_Name", "Last_Name"],
                                "Shipping_Address": ["Address", "City", "State", "Postal_Code"]},

                "foreign_keys": [{"from": "customers", "to": "orders"},
                            {"from": "customers", "to": "locations"},
                            {"from": "products", "to": "orders"},
                            {"from": "locations", "to": "orders"},
                            {"from": "orders", "to": "payments"}],
                "table_mapping": {
                        "customers": ["customer_id", "first_name", "last_name", "email", "phone_number"],
                        "products": ["product_id", "product_name", "brand", "category", "price"],
                        "orders": ["order_id", "customer_id", "product_id", "quantity", "total_price", "order_date"],
                        "locations": ["customer_id", "address", "city", "state", "postal_code", "country"],
                        "payments": ["order_id", "payment_method", "transaction_status"]}
            }
        },

        "csv_ex": {
            "type": "gdrive",
            "path": "1DdmNsrdBRLfzBdgtvzvFHZ7ejFpLlpwW",
            "pipelines": ["olap"],
            "olap": {
                "dimensions": {
                    "dim_warehouses": ["warehouse_id", "location", "manager"],
                    "dim_inventory": ["sku", "product_name", "category"]
                },
                "facts": {
                    "fact_stock": ["sku", "warehouse_id", "last_restock_date", "quantity"]
                }
            }
        },

        "url_ex": {
            "type": "url",
            "path": "https://example.com/pricing_data.json",
            "pipelines": ["oltp"],
            "oltp": {
                "critical_columns": ["product_id", "effective_date"],
                "date_columns": ["effective_date"],
                "table_mapping": {
                    "price_history": ["product_id", "effective_date", "price"]
                }
            }
        }
    },
    
    "system_config": {
        "default_date_format": "%Y-%m-%d",
       # "currency_columns": ["unit_price", "line_total"],
        "country_holidays": "UK"
    }
}
