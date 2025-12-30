
import os
import pandas as pd
import numpy as np

np.random.seed(42)
n_samples = 1000

customer_id = np.arange(1, n_samples + 1)
age = np.random.randint(18, 70, size=n_samples)
region = np.random.choice(['North', 'South', 'East', 'West'], size=n_samples)
tenure = np.random.randint(1, 60, size=n_samples)
avg_monthly_spend = np.random.uniform(50, 500, size=n_samples)
support_tickets = np.random.randint(0, 10, size=n_samples)
login_frequency = np.random.randint(0, 30, size=n_samples)

churn_prob = (
    (login_frequency < 5).astype(int) * 0.4 +
    (support_tickets > 5).astype(int) * 0.3 +
    (avg_monthly_spend < 100).astype(int) * 0.3
)
churn_flag = (np.random.rand(n_samples) < churn_prob).astype(int)

df = pd.DataFrame({
    'customer_id': customer_id,
    'age': age,
    'region': region,
    'tenure': tenure,
    'avg_monthly_spend': avg_monthly_spend,
    'support_tickets': support_tickets,
    'login_frequency': login_frequency,
    'churn_flag': churn_flag
})

os.makedirs('data', exist_ok=True)
df.to_csv(os.path.join('data', 'synthetic_churn_data.csv'), index=False)
print("✅ synthetic_churn_data.csv generated in ./data")