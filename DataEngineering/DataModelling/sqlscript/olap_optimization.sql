-- ================================
-- 🚀 ZULO_OLAP OPTIMIZATION SCRIPT
-- ================================

-- === [1] Indexes on SK Columns ===
CREATE INDEX IF NOT EXISTS idx_fact_transactions_account_sk ON zulo_olap.fact_transactions(account_sk);
CREATE INDEX IF NOT EXISTS idx_fact_transactions_transaction_date_id ON zulo_olap.fact_transactions(transaction_date_id);

CREATE INDEX IF NOT EXISTS idx_fact_loans_customer_sk ON zulo_olap.fact_loans(customer_sk);
CREATE INDEX IF NOT EXISTS idx_fact_loans_start_date_id ON zulo_olap.fact_loans(start_date_id);
CREATE INDEX IF NOT EXISTS idx_fact_loans_end_date_id ON zulo_olap.fact_loans(end_date_id);

-- === [2] Materialized View for Monthly Transactions ===
DROP MATERIALIZED VIEW IF EXISTS zulo_olap.mv_monthly_transaction_summary;

CREATE MATERIALIZED VIEW zulo_olap.mv_monthly_transaction_summary AS
SELECT
    d.year,
    d.month,
    a.account_type,
    COUNT(f.transaction_sk) AS transaction_count,
    SUM(f.amount) AS total_amount
FROM zulo_olap.fact_transactions f
JOIN zulo_olap.dim_accounts a ON f.account_sk = a.account_sk
JOIN zulo_olap.dim_date d ON f.transaction_date_id = d.date_sk
GROUP BY d.year, d.month, a.account_type;

-- === [3] View: Total Transactions per Account ===
CREATE OR REPLACE VIEW zulo_olap.v_account_transaction_totals AS
SELECT
    a.account_sk,
    a.account_type,
    COUNT(f.transaction_sk) AS txn_count,
    SUM(f.amount) AS total_amount
FROM zulo_olap.fact_transactions f
JOIN zulo_olap.dim_accounts a ON f.account_sk = a.account_sk
GROUP BY a.account_sk, a.account_type;

-- === [4] View: Loan Count by Country ===
CREATE OR REPLACE VIEW zulo_olap.v_loans_by_country AS
SELECT
    c.country,
    COUNT(f.loan_sk) AS loan_count
FROM zulo_olap.fact_loans f
JOIN zulo_olap.dim_customers c ON f.customer_sk = c.customer_sk
GROUP BY c.country;
