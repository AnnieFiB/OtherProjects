# config.py
# this is a sample config file for ETL process.
# it contains the configuration for different data sources, including their types, paths, and pipelines.  

ETL_CONFIG = {
    "data_sources": {
        "zulo_bank": {
            "type": "file",
            "path": "dataset/zulo_bank.csv",
            "pipelines": ["oltp", "olap"],
            "oltp": {
                "schema": "zulo_oltp",
                "critical_columns": {
                    "loan": ["LoanID"],
                    "customer": ["CustomerID"],
                    "account": ["AccountID"],
                    "transaction": ["TransactionID"]},
                "dedup_columns": {
                    "loan": ["loan_id"],
                    "customer": ["customer_id"],
                    "account": ["account_id"],
                    "transaction": ["transaction_id"]
                    },
                "date_columns": ["TransactionDate", "OpeningDate", "StartDate", "EndDate"],
                "odate_columns": ["transaction_date", "opening_date", "start_date", "end_date"],
                "duplicate_key_columns": ["TransactionID", "CustomerID"],
                "contact_columns": {"email_col": "Email", "phone_col": "Phone"},
                "derived_fields": {"price_col": "interest_rate", "qty_col": "loan_amount", "new_col": "interest"},
                "date_mapping": {
                        "transaction_date": "transactiondate_id",
                        "opening_date": "openingdate_id",
                        "start_date": "startdate_id",
                        "end_date": "enddate_id"
                                },
                "foreign_keys": {
                        "loan": [
                                    ("customer_id", "customer", "customer_id"),
                                    ("startdate_id", "date_dim", "date_id"),
                                    ("enddate_id", "date_dim", "date_id")
                                ],
                        "transaction": [
                                    ("account_id", "account", "account_id"),
                                    ("transactiondate_id", "date_dim", "date_id")
                                ],
                        "account": [
                                    ("customer_id", "customer", "customer_id"),
                                     ("openingdate_id", "date_dim", "date_id") ],           
                                },
                "split_columns": {"FullName": ["first_name", "last_name"]},
                "table_mapping": {
                            "customer": ["customer_id", "first_name", "last_name", "email", "phone"],
                            "account": ["account_id", "customer_id", "account_type", "balance", "opening_date", "openingdate_id"],
                            "loan": ["loan_id", "customer_id", "loan_amount", "loan_type","start_date", "startdate_id","end_date", "enddate_id","interest_rate", "interest"],
                            "transaction": ["transaction_id", "transaction_type", "amount","transaction_date", "transactiondate_id", "account_id"]         
                                   },
                "use_surrogate_keys": True
            },
            "olap": {
                "schema": "zulo_olap",
                "dimensions": {
                    "dim_customer": ["customer_id", "first_name", "last_name", "email", "phone"],
                    "dim_account": ["account_id", "account_type", "balance"],
                    "dim_transaction": ["transaction_id", "transaction_type"],
                    "dim_loan": ["loan_id", "loan_type", "interest_rate"]
                },
                "facts": {
                    "fact_transaction": {
                        "source_table": "transaction",
                        "fields": [
                            "transaction_sk",
                            "account_sk",
                            "customer_sk",
                            "transactiondate_id",
                            "openingdate_id",
                            "amount"
                        ],
                        "joins": [
                            ["transaction", "transaction_id", ["amount", "transactiondate_id", "account_id"]],
                            ["account", "account_id", ["customer_id", "openingdate_id"]]
                        ]
                    },
                    "fact_loan": {
                        "source_table": "loan",
                        "fields": [
                            "loan_sk",
                            "customer_sk",
                            "startdate_id",
                            "enddate_id",
                            "loan_amount",
                            {"interest": "loan_amount * interest_rate"}
                        ],
                        "joins": [
                            ["loan", "loan_id", ["loan_amount", "loan_type", "interest_rate", "startdate_id", "enddate_id"]],
                            ["customer", "customer_id", []]
                        ]
                    }
                },
                "olap_foreign_keys": {
                    "fact_transaction": [
                        ["transaction_sk", "dim_transaction", "transaction_sk"],
                        ["account_sk", "dim_account", "account_sk"],
                        ["customer_sk", "dim_customer", "customer_sk"],
                        ["transactiondate_id", "date_dim", "date_id"],
                        ["openingdate_id", "date_dim", "date_id"]
                    ],
                    "fact_loan": [
                        ["loan_sk", "dim_loan", "loan_sk"],
                        ["customer_sk", "dim_customer", "customer_sk"],
                        ["startdate_id", "date_dim", "date_id"],
                        ["enddate_id", "date_dim", "date_id"]
                    ]
                }
            },
            "olap_foreign_keys": {
                "fact_transaction": [
                    ["transaction_sk", "dim_transaction", "transaction_sk"],
                    ["account_sk", "dim_account", "account_sk"],
                    ["customer_sk", "dim_customer", "customer_sk"],
                    ["transactiondate_id", "date_dim", "date_id"],
                    ["openingdate_id", "date_dim", "date_id"]
                ],
                "fact_loan": [
                    ["loan_sk", "dim_loan", "loan_sk"],
                    ["customer_sk", "dim_customer", "customer_sk"],
                    ["startdate_id", "date_dim", "date_id"],
                    ["enddate_id", "date_dim", "date_id"]
                ]
            }
        },

        "yanki_ecom": {
            "type": "gdrive",
            "path": "1lPmrM-4EJLfM14E_3pWF2aLl_Y7HsoHa",
            "pipelines": ["oltp"],
            "oltp": {
                "schema": "yanki_oltp",
                "missingvalue_columns": ["Order_ID", "Customer_ID"],
                "critical_columns": {
                    "order": ["order_id"],
                    "customer": ["customer_id"],
                    "product": ["product_id"],
                    "payment": ["order_id"],
                    "location": ["location_id"]
                                    },
                "date_columns": ["Order_Date"],
                "duplicate_key_columns": ["Order_ID", "Product_ID", "Customer_ID"],
                "contact_columns": {"email_col": "Email", "phone_col": "Phone_Number"},
                "derived_fields": {"price_col": "Price", "qty_col": "Quantity", "new_col": "Total_Price"},
                "split_columns": {
                    "Customer_Name": ["First_Name", "Last_Name"],
                    "Shipping_Address": ["Address", "City", "State", "Postal_Code"]
                                 },
                "foreign_keys": {
                        "order": [
                                    ("customer_id", "customer", "customer_id"),
                                    ("product_id", "product", "product_id")],
                        "location": [
                                    ("customer_id", "customer", "customer_id")],
                        "payment": [
                                    ("order_id", "order", "order_id") ]
                                },
                "table_mapping": {
                    "customer": ["customer_id", "first_name", "last_name", "email", "phone_number"],
                    "product": ["product_id", "product_name", "brand", "category", "price"],
                    "location": ["location_id","customer_id", "address", "city", "state", "postal_code", "country"],
                    "order": ["order_id", "customer_id", "product_id","quantity", "total_price", "order_date"],
                    "payment": ["order_id", "payment_method", "transaction_status"]
                }
            }
        },

        "sales_retailx": {
    "type": "kaggle",
    "path": "shantanugarg274/sales-dataset",  # adjust path as needed
    "pipelines": ["oltp"],
    "oltp": {
        "schema": "sales_oltp",
        "critical_columns": {
            "orders": ["order_id"],
            "customers": ["customer_id"],
            "locations": ["location_id"],
            "products": ["product_id"]
        },
        "date_columns": ["Order Date"],
        #"keys": ["order_id"],
        "split_columns": {
            "CustomerName": ["first_name", "last_name"]
        },
        "derived_fields": {
            "price_col": "Amount",
            "qty_col": "Quantity",
            "profit_col": "Profit",
            "new_col1": "cost(Amount / Quantity)",
            "new_col2": ",cost(Amount - Profit)"

        },
        "table_mapping": {
            "orders": ["order_id", "amount", "profit", "quantity", "order_date", "payment_mode"],
            "customers": ["customer_name", "first_name", "last_name"],
            "locations": ["state", "city"],
            "products": ["category", "sub_category"]
        }
    }
}
,
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
        "country_holidays": "UK"
    }
}

  