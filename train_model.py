import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, roc_auc_score, confusion_matrix
from sklearn.preprocessing import LabelEncoder

def train_churn_model(input_path, output_dir):
    df = pd.read_csv(input_path)
    os.makedirs(output_dir, exist_ok=True)
    
    # 1. Preprocessing
    le = LabelEncoder()
    df['service_type_encoded'] = le.fit_transform(df['service_type'])
    
    features = ['contract_value', 'tenure_months', 'satisfaction_score', 
                'avg_monthly_tickets', 'late_payment_rate', 'service_type_encoded']
    X = df[features]
    y = df['churn']
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=42, stratify=y)
    
    # 2. Model Choice: Random Forest with Class Weighting
    # Justification: Handles imbalance by adjusting the penalty for the minority class.
    # Stratified sampling ensures the 18% churn rate is maintained in train/test splits.
    model = RandomForestClassifier(n_estimators=100, class_weight='balanced', random_state=42)
    model.fit(X_train, y_train)
    
    # 3. Evaluation
    y_pred = model.predict(X_test)
    y_prob = model.predict_proba(X_test)[:, 1]
    
    print("Classification Report:")
    print(classification_report(y_test, y_pred))
    print(f"ROC-AUC Score: {roc_auc_score(y_test, y_prob):.4f}")
    
    # 4. Feature Importance
    importances = pd.DataFrame({
        'feature': features,
        'importance': model.feature_importances_
    }).sort_values(by='importance', ascending=False)
    
    plt.figure(figsize=(10, 6))
    sns.barplot(x='importance', y='feature', data=importances, palette='magma')
    plt.title('Top Factors Driving Churn (Feature Importance)')
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'feature_importance.png'))
    plt.close()
    
    # 5. Predict Churn Probability for all clients
    df['churn_probability'] = model.predict_proba(X)[:, 1]
    at_risk = df[df['churn_probability'] > 0.5].sort_values(by='churn_probability', ascending=False)
    at_risk.to_csv('client_churn_forecast/data/at_risk_clients.csv', index=False)
    
    # Save probabilities for forecasting
    df.to_csv('client_churn_forecast/data/clients_with_risk.csv', index=False)
    
    print(f"Model trained. At-risk clients saved to data/at_risk_clients.csv")
    return model, importances

if __name__ == "__main__":
    clean_path = 'client_churn_forecast/data/cleaned_clients.csv'
    visuals_dir = 'client_churn_forecast/visuals'
    train_churn_model(clean_path, visuals_dir)
