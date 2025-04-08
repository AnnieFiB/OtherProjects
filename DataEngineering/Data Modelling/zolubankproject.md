<p align="center">
  <img src="../assets/banklogo.png" alt="Zulo Bank Logo" width="150"/>
</p>

<h1 align="center"><strong>Case Study - Zulo Bank: Database Design & Datawarehouse Modelling</strong></h1>

---

## Table of Contents
- [1. Executive Summary](#1-executive-summary)
- [2. Business Problem Statement](#2-business-problem-statement)
- [3. Objectives](#3-objectives)
- [4. Benefits](#4-benefits)
- [5. Tech Stack](#5-tech-stack)
- [6. Project Scope](#6-project-scope)
- [7. Dataset](#7-dataset)
- [8. Database Modelling](#8-database-modelling)
- [9. Datawarehouse Modelling](#9-datawarehouse-modelling)

---

## 1. Executive Summary
*Step-by-step guide for designing Zulo Bank’s database and data warehouse, covering business requirement gathering, normalization, key constraints, and schema transformation to address real-world banking data challenges.*

---

## 2. Business Problem Statement
*Zulo Bank’s legacy data system suffers from redundancy, inconsistency, and limited accessibility, disrupting reporting and analytics. A redesigned architecture is needed to ensure integrated, reliable, and analysis-ready data.*

---

## 3. Objectives
1. **Understand Business Requirements:**
   - Establish a clear understanding of Zulo Bank’s business needs, focus on real-time data access, efficient transaction processing, and support for advanced analytics.
2. **Apply Database Design Principles:**
   - Design normalized schemas using primary and foreign keys, ensure compliance with Third Normal Form (3NF), and selectively denormalize where performance demands.
3. **Develop ER Diagrams (ERDs):**
   - Create logical models to visualize relationships between core banking entities such as customers, accounts, transactions, and loans.
4. **Evaluate Schema Approaches:**
    - Assess OLTP (normalized) vs. OLAP (denormalized) schema structures and choose appropriately based on operational vs. analytical needs.
5. **Implement Schema Transformation:**
    - Convert the normalized database into a denormalized, analytics-ready data warehouse schema using star schema principles.

---

## 4. Benefits
1. Hands-on banking data modeling with real-world structures like customers, accounts, and transactions
2. Optimized data architecture balancing normalization for operations and denormalization for analytics
3. Analytics-ready warehouse design enabling insights into customer behavior and financial performance
4. Career-building experience aligned with enterprise-level data engineering practices
5. Structured project workflow reflecting end-to-end delivery in a real business context

---

## 5. Tech Stack
- **PostgreSQL + pgAdmin** – Design and manage a normalized relational database using PostgreSQL, with pgAdmin for schema creation, queries, and data inspection
- **Data Warehouse Modeling**– Build a star schema optimized for analytics, with fact and dimension tables supporting BI use cases
- **Draw.io** – Visualize ERDs and warehouse schemas to clearly document data structures and relationships
- **Python (optional) – For scripting ETL workflows or data preparation if needed**

*This stack supports the complete project lifecycle, from schema design to data warehouse deployment, within the Zulo Bank case study.*

---

## 6. Project Scope
- **Database Design:** Build a normalized relational schema capturing core banking data entities — `customers`, `accounts`, `transactions`, `loans`  — using 3NF for data integrity and reduced redundancy.
- **Data Warehouse Modeling:** Convert the normalized schema into a star schema with a central fact table and surrounding dimension tables to enable high-performance analytics.
- **Diagramming & Documentation:** Use Draw.io to generate ERDs and schema diagrams that communicate structure, relationships, and data flows.
  
*This scope outlines the full data modeling lifecycle — from relational design to warehouse transformation — specific to the operational and analytical needs of Zulo Bank.*

---

## 7. Dataset
- <a href="https://drive.google.com/file/d/1DdmNsrdBRLfzBdgtvzvFHZ7ejFpLlpwW/view?usp=drive_link" target="_blank">Dataset</a>

| TransactionID | TransactionType | Amount | TransactionDate | CustomerID | FullName        | Email                        | Phone                  | AccountID | AccountType | Balance | OpeningDate | LoanID | LoanAmount | LoanType | StartDate  | EndDate                      | InterestRate |
|---------------|-----------------|--------|------------------|------------|------------------|-------------------------------|-------------------------|-----------|-------------|---------|--------------|--------|------------|----------|------------|------------------------------|---------------|
| 1             | withdrawal      | 102.15 | 2023-04-26       | 85         | Carol Miller     | yfisher@example.org          | 6088279027             | 88        | Savings     | 5652.16 | 2019-08-12   | —      | —          | —        | —          | —                            | —             |
| 2             | withdrawal      | 358.80 | 2020-06-13       | 91         | Geoffrey Banks   | gonzalesgeorge@example.net   | 001-546-857-6518x5359  | 26        | Credit      | 2881.24 | 2019-05-06   | 44     | 32428.90   | Mortgage | 2021-06-24 | 2050-01-08 04:59:17.907588  | 2.12          |
| 2             | withdrawal      | 358.80 | 2020-06-13       | 91         | Geoffrey Banks   | gonzalesgeorge@example.net   | 001-546-857-6518x5359  | 26        | Credit      | 2881.24 | 2019-05-06   | 48     | 31406.77   | Personal | 2021-02-27 | 2038-10-12 04:59:17.907821  | 4.63          |
| 2             | withdrawal      | 358.80 | 2020-06-13       | 91         | Geoffrey Banks   | gonzalesgeorge@example.net   | 001-546-857-6518x5359  | 26        | Credit      | 2881.24 | 2019-05-06   | 76     | 27834.00   | Personal | 2019-12-05 | 2037-08-15 04:59:17.909497  | 2.15          |
| 2             | withdrawal      | 358.80 | 2020-06-13       | 91         | Geoffrey Banks   | gonzalesgeorge@example.net   | 001-546-857-6518x5359  | 26        | Credit      | 2881.24 | 2019-05-06   | 138    | 27873.08   | Auto     | 2022-01-19 | 2037-06-03 04:59:17.913974  | 7.03          |


- **Data Preprocessing (For Database & Warehouse Design)** Before designing the database and data warehouse schemas, the following minimal preprocessing steps were performed to ensure data quality and model integrity:
|
Step                     | Description                                                                 |
|--------------------------|-----------------------------------------------------------------------------|
| Convert Date Columns     | Standardize `TransactionDate`, `OpeningDate`, `StartDate`, `EndDate` to `YYYY-MM-DD` format. |
| Ensure Correct Data Types| Verify numeric fields (`Amount`, `Balance`, `LoanAmount`, `InterestRate`) are numbers. |
| Clean Obvious Inconsistencies | Remove any corrupted or incomplete rows (e.g., missing critical IDs like `CustomerID`). |
| Normalize Text Fields    | Lowercase and trim key categorical fields like `TransactionType`, `AccountType`, `LoanType`. |
| Validate Key Fields      | Ensure `CustomerID`, `AccountID`, and `LoanID` (where present) are valid and not null. |

---

## 8. From OLTP to OLAP: Schema Progression
*This project moves from a normalized, operational OLTP schema to a denormalized OLAP star schema to support analytical workloads.*

### Stage 1: OLTP – Normalized Database Design
*A relational schema is designed to support day-to-day banking operations, minimize redundancy, and ensure data integrity through normalization up to 3NF*

- **Normalization Process:** To ensure data integrity and eliminate redundancy, the data is normalized through the following stages:
    - *1NF (First Normal Form)*
      > Eliminate repeating groups
      > Ensure all columns contain atomic (indivisible) values
      
    - *2NF (Second Normal Form)*
      > Remove partial dependencies
      > Ensure all non-key attributes depend on the whole primary key

    - *3NF (Third Normal Form)*
      > Eliminate transitive dependencies
      > Ensure all attributes depend only on the primary key

- **Tables to Be Created**
    1. **Customer**: `CustomerID (PK)`, FullName (`lastname` `firstname`), `Email`, `Phone`
    2. **Account**: `AccountID (PK)`, `CustomerID (FK)`, `AccountType`, `Balance`, `OpeningDate`
    3. **Transaction**:  `TransactionID (PK)`, `AccountID (FK)`, `TransactionType`, `Amount`, `TransactionDate`
    4. **Loan**:  `LoanID (PK)`, `CustomerID (FK)`, `LoanType`, `LoanAmount`, `StartDate`, `EndDate`, `InterestRate`


- **Keys and Relationships**
    - Primary Keys (PK): Unique identifiers for each entity (e.g., `CustomerID`, `AccountID`, `TransactionID`, `LoanID`)
    - Foreign Keys (FK): Define relationships between tables(Example: `Account.CustomerID` → `Customer.CustomerID`)
    - One-to-Many Relationships:
      > Customer → Account
      > Customer → Loan
      > Account → Transaction
      
- Create ERD to visualize structure
  
- Implement schema using PostgreSQL in pgAdmin

### Stage 2: OLAP – Star Schema Design for Analytical Reporting
*The normalized data is then transformed into a star schema to support fast querying, aggregation, and dashboarding in a business intelligence environment.*

- Fact_Activity: 
  > TransactionID (FK), LoanID (FK), CustomerID (FK), AccountID (FK), DateID (FK)
- Dim_Customer
  > CustomerID (PK), FullName, Email, Phone
- Dim_Account
  > AccountID (PK), AccountType, Balance, OpeningDate
- Dim_Transaction
  > TransactionID (PK), TransactionType, Amount, TransactionDate
- Dim_Loan
  > LoanID (PK), LoanType, LoanAmount, StartDate, EndDate, InterestRate
- Dim_Date
  > DateID (PK), Date, Day, Month, Year, Quarter

- Optimize structure for fast querying and analytical reporting
  
---

## 9. Fact Table Considerations

*Two primary fact tables are proposed for this data warehouse design:*

1. Transactions Fact Table
   > TransactionID, AccountID (FK to Dim_Account), DateID (FK to Dim_Date), TransactionType, Amount
   
2. Loans Fact Table:
    > LoanID, CustomerID (FK to Dim_Customer), StartDateID & EndDateID (FKs to Dim_Date), LoanTypeID(FK), LoanAmount, InterestRate

*These fact tables capture measurable banking events and connect to relevant descriptive dimensions, enabling robust, multidimensional analysis.*

- **Star Schema (Used)** Fact tables at the center, connected to denormalized dimension tables, prioritizes simplicity and performance
- *Snowflake Schema (Not Used): Further normalizes dimensions into sub-dimensions, Space-efficient but adds complexity*
  
---

<p align="center">
  <em>End of Case Study</em>
</p> 
