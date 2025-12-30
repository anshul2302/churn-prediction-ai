import os
import pandas as pd
import joblib
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.metrics import roc_auc_score, classification_report

def load_data(path: str) -> pd.DataFrame:
    if not os.path.exists(path):
        raise FileNotFoundError(f'Dataset not found at: {path}')
    return pd.read_csv(path)

def preprocess(df: pd.DataFrame):
    # One-hot encode region if present
    if 'region' in df.columns:
        df = pd.get_dummies(df, columns=['region'], drop_first=True)
    # Target and features
    y = df['churn_flag']
    X = df.drop(['customer_id', 'churn_flag'], axis=1)
    feature_names = list(X.columns)
    return X, y, feature_names

def build_pipeline():
    # Standardize numeric features + Logistic Regression classifier
    return Pipeline([
        ('scaler', StandardScaler()),
        ('clf', LogisticRegression(max_iter=1000))
    ])

def main():
    # Paths
    data_path = os.path.join('data', 'synthetic_churn_data.csv')
    model_path = 'model.pkl'
    features_path = 'feature_names.txt'

    # Load & preprocess
    df = load_data(data_path)
    X, y, feature_names = preprocess(df)

    # Train/test split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    # Build & train
    model = build_pipeline()
    model.fit(X_train, y_train)

    # Evaluate (AUC)
    y_pred_prob = model.predict_proba(X_test)[:, 1]
    auc = roc_auc_score(y_test, y_pred_prob)
    print(f'AUC: {auc:.4f}')

    # Save model + feature names
    joblib.dump(model, model_path)
    with open(features_path, 'w', encoding='utf-8') as f:
        for name in feature_names:
            f.write(name + '\n')

    print(f'Model saved to {model_path}')
    print(f'Feature names saved to {features_path}')

    # Optional: Human-readable metrics
    y_pred = (y_pred_prob >= 0.5).astype(int)
    print('\nClassification report (threshold=0.5):')
    print(classification_report(y_test, y_pred))

if __name__ == '__main__':
    main()
