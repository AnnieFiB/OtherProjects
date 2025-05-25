
-- OLTP Tables
CREATE TABLE IF NOT EXISTS "{oltp_schema}"."customer" (
    customer_id INTEGER PRIMARY KEY,
    first_name VARCHAR(100),
    last_name VARCHAR(100),
    email VARCHAR(255),
    phone VARCHAR(50),
    customer_sk SERIAL UNIQUE
);

CREATE TABLE IF NOT EXISTS "{oltp_schema}"."account" (
    account_id INTEGER PRIMARY KEY,
    account_type VARCHAR(50),
    balance DECIMAL(15,2),
    opening_date DATE,
    opening_date_id INTEGER,
    account_sk SERIAL UNIQUE
);

CREATE TABLE IF NOT EXISTS "{oltp_schema}"."transaction" (
    transaction_id INTEGER PRIMARY KEY,
    transaction_type VARCHAR(50),
    amount DECIMAL(15,2),
    transaction_date DATE,
    transaction_date_id INTEGER,
    transaction_sk SERIAL UNIQUE
);

CREATE TABLE IF NOT EXISTS "{oltp_schema}"."loan" (
    loan_id INTEGER PRIMARY KEY,
    loan_amount DECIMAL(15,2),
    loan_type VARCHAR(50),
    start_date DATE,
    start_date_id INTEGER,
    end_date DATE,
    end_date_id INTEGER,
    interest_rate DECIMAL(5,2),
    loan_sk SERIAL UNIQUE
);

CREATE TABLE IF NOT EXISTS "{oltp_schema}"."zulo_lookup" (
    customer_sk INTEGER REFERENCES "{oltp_schema}"."customer"(customer_sk),
    account_sk INTEGER REFERENCES "{oltp_schema}"."account"(account_sk),
    transaction_sk INTEGER REFERENCES "{oltp_schema}"."transaction"(transaction_sk),
    loan_sk INTEGER REFERENCES "{oltp_schema}"."loan"(loan_sk),
    PRIMARY KEY (customer_sk, account_sk, transaction_sk, loan_sk)
);

-- OLAP Tables
CREATE TABLE IF NOT EXISTS "{olap_schema}"."date_dim" (
    date_id INTEGER PRIMARY KEY,
    date DATE,
    year INTEGER,
    month INTEGER,
    month_name VARCHAR(20),
    quarter INTEGER,
    day INTEGER,
    day_of_week VARCHAR(20),
    is_weekend BOOLEAN,
    is_month_end BOOLEAN
);

CREATE TABLE IF NOT EXISTS "{olap_schema}"."customer_dim" (
    customer_sk INTEGER PRIMARY KEY REFERENCES "{oltp_schema}"."customer"(customer_sk),
    first_name VARCHAR(100),
    last_name VARCHAR(100),
    email VARCHAR(255),
    phone VARCHAR(50)
);

CREATE TABLE IF NOT EXISTS "{olap_schema}"."account_dim" (
    account_sk INTEGER PRIMARY KEY REFERENCES "{oltp_schema}"."account"(account_sk),
    account_type VARCHAR(50)
);

CREATE TABLE IF NOT EXISTS "{olap_schema}"."transaction_dim" (
    transaction_sk INTEGER PRIMARY KEY REFERENCES "{oltp_schema}"."transaction"(transaction_sk),
    transaction_type VARCHAR(50)
);

CREATE TABLE IF NOT EXISTS "{olap_schema}"."loan_dim" (
    loan_sk INTEGER PRIMARY KEY REFERENCES "{oltp_schema}"."loan"(loan_sk),
    loan_type VARCHAR(50)
);

CREATE TABLE IF NOT EXISTS "{olap_schema}"."transaction_fact" (
    transaction_sk INTEGER REFERENCES "{olap_schema}"."transaction_dim"(transaction_sk),
    account_sk INTEGER REFERENCES "{olap_schema}"."account_dim"(account_sk),
    customer_sk INTEGER REFERENCES "{olap_schema}"."customer_dim"(customer_sk),
    transaction_date_id INTEGER REFERENCES "{olap_schema}"."date_dim"(date_id),
    opening_date_id INTEGER REFERENCES "{olap_schema}"."date_dim"(date_id),
    amount DECIMAL(15,2),
    balance DECIMAL(15,2),
    PRIMARY KEY (transaction_sk, account_sk, customer_sk, transaction_date_id, amount, balance)
);

CREATE TABLE IF NOT EXISTS "{olap_schema}"."loan_fact" (
    loan_sk INTEGER REFERENCES "{olap_schema}"."loan_dim"(loan_sk),
    customer_sk INTEGER REFERENCES "{olap_schema}"."customer_dim"(customer_sk),
    start_date_id INTEGER REFERENCES "{olap_schema}"."date_dim"(date_id),
    end_date_id INTEGER REFERENCES "{olap_schema}"."date_dim"(date_id),
    loan_amount DECIMAL(15,2),
    interest DECIMAL(15,2),
    PRIMARY KEY (loan_sk, customer_sk, start_date_id, end_date_id, loan_amount, interest)
);
