import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os

def run_eda(input_path, output_dir):
    df = pd.read_csv(input_path)
    os.makedirs(output_dir, exist_ok=True)
    
    sns.set_theme(style="whitegrid")
    
    # 1. Churn by Service Type
    plt.figure(figsize=(10, 6))
    sns.barplot(x='service_type', y='churn', data=df, palette='viridis')
    plt.title('Churn Rate by Service Type')
    plt.ylabel('Churn Rate')
    plt.xlabel('Service Type')
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'churn_by_service.png'))
    plt.close()
    
    # 2. Satisfaction Score vs Churn
    plt.figure(figsize=(10, 6))
    sns.boxplot(x='churn', y='satisfaction_score', data=df, palette='magma')
    plt.title('Satisfaction Score Distribution by Churn Status')
    plt.ylabel('Satisfaction Score')
    plt.xlabel('Churn (0 = No, 1 = Yes)')
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'satisfaction_vs_churn.png'))
    plt.close()
    
    # 3. Contract Value vs Churn
    plt.figure(figsize=(10, 6))
    sns.kdeplot(data=df, x='contract_value', hue='churn', fill=True, palette='coolwarm')
    plt.title('Contract Value Distribution by Churn Status')
    plt.ylabel('Density')
    plt.xlabel('Monthly Contract Value ($)')
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'contract_value_dist.png'))
    plt.close()
    
    # 4. Support Tickets vs Churn
    plt.figure(figsize=(10, 6))
    sns.scatterplot(x='avg_monthly_tickets', y='satisfaction_score', hue='churn', data=df, alpha=0.6)
    plt.title('Support Tickets vs Satisfaction (Colored by Churn)')
    plt.ylabel('Satisfaction Score')
    plt.xlabel('Avg Monthly Tickets')
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'tickets_vs_sat.png'))
    plt.close()

    print(f"EDA Visualizations saved to {output_dir}")

if __name__ == "__main__":
    clean_path = 'client_churn_forecast/data/cleaned_clients.csv'
    visuals_dir = 'client_churn_forecast/visuals'
    run_eda(clean_path, visuals_dir)
