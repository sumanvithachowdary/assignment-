import pandas as pd
import numpy as np
import os

def clean_and_preprocess(clients_path, trans_path):
    clients = pd.read_csv(clients_path)
    trans = pd.read_csv(trans_path)
    
    print(f"Initial missing values in transactions:\n{trans.isnull().sum()}")
    
    # 1. Missing Data Strategy: Impute 'amount' using 'contract_value' from clients
    # Justification: Managed IT services are subscription-based. 
    # The missing 'amount' is likely the standard monthly fee defined in the contract.
    trans = trans.merge(clients[['client_id', 'contract_value']], on='client_id', how='left')
    trans['amount'] = trans['amount'].fillna(trans['contract_value'])
    
    # Drop the temporary contract_value column from transactions
    trans = trans.drop(columns=['contract_value'])
    
    print(f"Missing values after imputation:\n{trans.isnull().sum()}")
    
    # 2. Feature Engineering from Transactions
    # Aggregate payment history: late/missing payments might indicate churn risk
    payment_reliability = trans.groupby('client_id')['payment_status'].apply(
        lambda x: (x == 'Missing').mean()
    ).reset_index(name='late_payment_rate')
    
    # 3. Merge with Client Data
    final_df = clients.merge(payment_reliability, on='client_id', how='left')
    
    # 4. Save Cleaned Data
    clean_clients_path = 'client_churn_forecast/data/cleaned_clients.csv'
    clean_trans_path = 'client_churn_forecast/data/cleaned_transactions.csv'
    
    final_df.to_csv(clean_clients_path, index=False)
    trans.to_csv(clean_trans_path, index=False)
    
    print(f"Cleaned data saved to {clean_clients_path} and {clean_trans_path}")
    return final_df, trans

if __name__ == "__main__":
    clients_p = 'client_churn_forecast/data/clients.csv'
    trans_p = 'client_churn_forecast/data/transactions.csv'
    clean_and_preprocess(clients_p, trans_p)
