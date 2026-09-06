import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os

def run_forecast(input_path, output_dir):
    df = pd.read_csv(input_path)
    os.makedirs(output_dir, exist_ok=True)
    
    # 1. Baseline Monthly Revenue
    monthly_rev = df['contract_value'].sum()
    print(f"Current Monthly Recurring Revenue (MRR): ${monthly_rev:,.2f}")
    
    # 2. Forecast next 12 months
    months = range(1, 13)
    
    # Baseline: No Churn
    baseline = [monthly_rev] * 12
    
    # Risk-Adjusted: Revenue * (1 - Churn Probability)
    # Note: Churn prob is for next 3 months, we'll annualize/distribute it
    # Simplified: monthly_churn_prob = churn_prob / 3
    df['monthly_churn_risk'] = df['churn_probability'] / 3
    
    risk_adjusted = []
    current_rev = monthly_rev
    for m in months:
        lost_rev = (df['contract_value'] * df['monthly_churn_risk']).sum()
        current_rev -= lost_rev
        risk_adjusted.append(max(0, current_rev))
        
    # Optimized: 5% Reduction in Churn Risk
    optimized = []
    current_rev_opt = monthly_rev
    df['opt_churn_risk'] = df['monthly_churn_risk'] * 0.95 # 5% relative reduction
    for m in months:
        lost_rev_opt = (df['contract_value'] * df['opt_churn_risk']).sum()
        current_rev_opt -= lost_rev_opt
        optimized.append(max(0, current_rev_opt))
        
    # 3. Visualization
    plt.figure(figsize=(12, 6))
    plt.plot(months, baseline, label='Baseline (No Churn)', linestyle='--', color='gray')
    plt.plot(months, risk_adjusted, label='Risk-Adjusted Forecast', marker='o', color='red')
    plt.plot(months, optimized, label='Optimized Forecast (5% Churn Reduction)', marker='s', color='green')
    
    plt.title('12-Month Revenue Forecast for Managed IT Services')
    plt.xlabel('Month')
    plt.ylabel('Projected Monthly Revenue ($)')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'revenue_forecast.png'))
    plt.close()
    
    # Save forecast data
    forecast_df = pd.DataFrame({
        'Month': months,
        'Baseline': baseline,
        'Risk_Adjusted': risk_adjusted,
        'Optimized': optimized
    })
    forecast_df.to_csv('client_churn_forecast/data/revenue_forecast.csv', index=False)
    
    print(f"Forecast completed. Visualization saved to {output_dir}")
    print(f"Projected Annual Revenue Loss (Risk-Adjusted): ${(baseline[-1] - risk_adjusted[-1]) * 12:,.2f}")

if __name__ == "__main__":
    risk_path = 'client_churn_forecast/data/clients_with_risk.csv'
    visuals_dir = 'client_churn_forecast/visuals'
    run_forecast(risk_path, visuals_dir)
