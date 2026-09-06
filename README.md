# Client Churn Risk & Revenue Forecasting for Managed IT Services
**Author:** Manus AI

## Executive Summary
This project analyzes a simulated dataset of 200 enterprise clients and their 3-year transaction history to predict churn risk and forecast revenue. By identifying key churn drivers and implementing a predictive model, we project a baseline annual revenue loss of **$70.5M** due to churn. However, by executing targeted retention strategies, we can reduce this risk by 5%, preserving significant recurring revenue.

## 1. Project Goals
1. **Identify clients likely to churn** in the next 3 months.
2. **Find the top factors driving churn**.
3. **Build a 12-month revenue forecasting model** incorporating churn risk.
4. **Suggest strategies to reduce churn by 5%**.

## 2. Data Strategy & Scalability
### Missing Data Strategy
The transaction dataset contained 12% missing financial records (`amount`). 
- **Imputation Method:** We imputed the missing transaction amounts using the client's baseline `contract_value`. 
- **Justification:** Managed IT services operate on a subscription model. A missing payment record does not necessarily mean zero revenue; it is highly probable that the expected amount is the standard monthly fee defined in the contract.

### Scalability Approach (10 Million Rows)
While this project uses a representative subset (200 clients, ~7,200 transactions), the pipeline is designed to scale to 10 million records:
- **Data Processing:** For 10M rows, we would transition from in-memory Pandas to **Dask** or **PySpark** for chunked, parallel processing.
- **Storage:** Raw CSVs would be converted to **Parquet** format, leveraging columnar storage for faster I/O and reduced memory footprint during aggregation.

## 3. Exploratory Data Analysis (EDA)
Key findings from the visual analysis:

**Satisfaction vs. Churn:** There is a stark contrast in satisfaction scores. Clients who churned consistently reported lower satisfaction scores compared to retained clients.
![Satisfaction vs Churn](https://files.manuscdn.com/user_upload_by_module/session_file/310519663671796482/FArMkdQRuydYJQIM.png)

**Support Tickets vs. Satisfaction:** High volumes of support tickets correlate strongly with lower satisfaction and higher churn rates, indicating that frequent IT issues drive clients away.
![Tickets vs Satisfaction](https://files.manuscdn.com/user_upload_by_module/session_file/310519663671796482/gagRpSivcbjYBpMK.png)

## 4. Churn Classification Model
### Handling Class Imbalance
The dataset exhibited an 18% churn rate, presenting a class imbalance problem.
- **Technique Used:** We utilized a **Random Forest Classifier** with `class_weight='balanced'`.
- **Justification:** Instead of synthetically generating data (SMOTE) or losing data (undersampling), class weighting adjusts the cost function, penalizing the model more heavily for misclassifying the minority class (churners). We also used **stratified sampling** during the train/test split to maintain the 18% ratio.
- **Evaluation:** The model achieved an **ROC-AUC score of 0.7913**. We prioritized ROC-AUC and F1-score over raw accuracy, as accuracy can be misleadingly high in imbalanced datasets by simply predicting the majority class.

### Top Churn Drivers
Feature importance analysis revealed the primary factors driving client churn:
1. **Satisfaction Score**
2. **Average Monthly Tickets**
3. **Contract Value**
4. **Tenure (Months)**

![Feature Importance](https://files.manuscdn.com/user_upload_by_module/session_file/310519663671796482/wpcJkdbPyrLdNMiE.png)

## 5. Revenue Forecasting
The current Monthly Recurring Revenue (MRR) is **$5.87M**. We built a 12-month forecast incorporating the predicted churn probabilities.

- **Baseline (No Churn):** Assumes 100% retention.
- **Risk-Adjusted Forecast:** Projects an annual revenue loss of **$70.5M** based on current churn trajectories.
- **Optimized Forecast:** Models the financial impact of reducing the churn risk by 5% through targeted interventions.

![Revenue Forecast](https://files.manuscdn.com/user_upload_by_module/session_file/310519663671796482/MRZwnStLECOqZwGX.png)

## 6. Strategic Recommendations (Targeting 5% Churn Reduction)
Based on the top churn drivers, we recommend the following concrete actions to achieve a 5% reduction in churn:

1. **Proactive Ticket Management (Addresses Driver #2):**
   - **Action:** Implement an alert system for clients exceeding their historical average monthly ticket volume by 20%.
   - **Impact:** Allows account managers to intervene before high ticket volumes degrade satisfaction scores.

2. **Satisfaction Score Interventions (Addresses Driver #1):**
   - **Action:** Any client reporting a satisfaction score below 3.0 should trigger an immediate executive review and a mandatory "health check" meeting.
   - **Impact:** Directly addresses the strongest predictor of churn.

3. **High-Value Client White-Glove Service (Addresses Driver #3):**
   - **Action:** Assign dedicated Customer Success Managers (CSMs) to clients in the top quartile of contract value, regardless of their current satisfaction score.
   - **Impact:** Protects the largest segments of MRR from unexpected churn.

4. **At-Risk Client Triage:**
   - **Action:** Utilize the generated `at_risk_clients.csv` list (clients with >50% churn probability) for immediate outreach campaigns offering service audits or temporary discounts.
   - **Impact:** Directly targets the clients identified by the ML model as most likely to leave in the next 3 months.
