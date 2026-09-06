import pandas as pd
import numpy as np
import os
from datetime import datetime, timedelta

def generate_data(n_clients=200, years=3, seed=42):
    np.random.seed(seed)
    
    # 1. Generate Client Base Data
    client_ids = [f"CL{str(i).zfill(4)}" for i in range(1, n_clients + 1)]
    service_types = ['Infrastructure', 'Cloud Migration', 'Security', '24/7 Support']
    
    # Target churn rate ~18%
    churn_probs = np.random.beta(2, 8, n_clients)
    churn_labels = (churn_probs > np.percentile(churn_probs, 82)).astype(int)
    
    clients = pd.DataFrame({
        'client_id': client_ids,
        'service_type': np.random.choice(service_types, n_clients),
        'contract_value': np.random.uniform(5000, 50000, n_clients),
        'tenure_months': np.random.randint(6, 60, n_clients),
        'satisfaction_score': np.random.uniform(1, 5, n_clients),
        'avg_monthly_tickets': np.random.poisson(5, n_clients),
        'churn': churn_labels
    })
    
    # Adjust features to correlate with churn for realistic modeling
    # Lower satisfaction and higher tickets -> Higher churn probability
    mask = clients['churn'] == 1
    clients.loc[mask, 'satisfaction_score'] = clients.loc[mask, 'satisfaction_score'] * 0.7
    clients.loc[mask, 'avg_monthly_tickets'] = (clients.loc[mask, 'avg_monthly_tickets'] * 1.5).astype(int)
    clients['satisfaction_score'] = clients['satisfaction_score'].clip(1, 5)
    
    # 2. Generate Transaction History (Monthly for 3 years)
    transactions = []
    start_date = datetime(2023, 1, 1)
    
    for _, client in clients.iterrows():
        for m in range(years * 12):
            date = start_date + timedelta(days=30 * m)
            
            # 12% missing financial records constraint
            is_missing = np.random.random() < 0.12
            amount = client['contract_value'] if not is_missing else np.nan
            
            transactions.append({
                'client_id': client['client_id'],
                'date': date,
                'amount': amount,
                'payment_status': 'Paid' if not is_missing else 'Missing'
            })
            
    transactions_df = pd.DataFrame(transactions)
    
    # SCALABILITY NOTE:
    # To scale this to 10 million rows, we would:
    # 1. Use Dask or Spark to parallelize the generation across partitions.
    # 2. Store the data in a distributed format like Parquet (columnar storage) for faster I/O.
    # 3. Process the cleaning and aggregation in chunks using pd.read_csv(chunksize=...) or Dask's lazy evaluation.
    
    return clients, transactions_df

if __name__ == "__main__":
    print("Generating synthetic data...")
    clients_df, trans_df = generate_data()
    
    clients_path = 'client_churn_forecast/data/clients.csv'
    trans_path = 'client_churn_forecast/data/transactions.csv'
    
    clients_df.to_csv(clients_path, index=False)
    trans_df.to_csv(trans_path, index=False)
    
    print(f"Saved {len(clients_df)} clients to {clients_path}")
    print(f"Saved {len(trans_df)} transactions to {trans_path}")
    print(f"Churn Rate: {clients_df['churn'].mean():.2%}")
    print(f"Missing Financial Data: {trans_df['amount'].isnull().mean():.2%}")
