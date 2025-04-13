-- === [1] OLTP → OLAP Row Counts ===
SELECT
    (SELECT COUNT(*) FROM zulo_oltp.transactions) AS oltp_transactions,
    (SELECT COUNT(*) FROM zulo_olap.fact_transactions) AS olap_fact_transactions,
    (SELECT COUNT(*) FROM zulo_oltp.loans) AS oltp_loans,
    (SELECT COUNT(*) FROM zulo_olap.fact_loans) AS olap_fact_loans;

-- === [2] Null SK or Date IDs in fact_* ===
SELECT COUNT(*) AS missing_account_sk FROM zulo_olap.fact_transactions WHERE account_sk IS NULL;
SELECT COUNT(*) AS missing_date_id FROM zulo_olap.fact_transactions WHERE transaction_date_id IS NULL;

SELECT COUNT(*) AS missing_customer_sk FROM zulo_olap.fact_loans WHERE customer_sk IS NULL;
SELECT COUNT(*) AS missing_start_date_id FROM zulo_olap.fact_loans WHERE start_date_id IS NULL;

-- === [3] Orphan SKs (fact SK not found in dim) ===
SELECT f.account_sk FROM zulo_olap.fact_transactions f
LEFT JOIN zulo_olap.dim_accounts d ON f.account_sk = d.account_sk
WHERE d.account_sk IS NULL;

SELECT f.customer_sk FROM zulo_olap.fact_loans f
LEFT JOIN zulo_olap.dim_customers d ON f.customer_sk = d.customer_sk
WHERE d.customer_sk IS NULL;

-- === [4] Sample Business Queries ===
SELECT a.account_type, COUNT(*) AS txn_count, SUM(f.amount) AS total_amount
FROM zulo_olap.fact_transactions f
JOIN zulo_olap.dim_accounts a ON f.account_sk = a.account_sk
GROUP BY a.account_type
ORDER BY txn_count DESC;

SELECT c.country, COUNT(*) AS total_loans
FROM zulo_olap.fact_loans f
JOIN zulo_olap.dim_customers c ON f.customer_sk = c.customer_sk
GROUP BY c.country
ORDER BY total_loans DESC;
