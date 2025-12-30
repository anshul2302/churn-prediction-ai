
import streamlit as st
import pandas as pd
import joblib
import os

# -------------- Load trained model and feature names --------------
MODEL_PATH = 'model.pkl'
FEATURES_PATH = 'feature_names.txt'

# Load the scikit-learn Pipeline saved via joblib
try:
    model = joblib.load(MODEL_PATH)
except Exception as e:
    st.error(f"Failed to load model from {MODEL_PATH}. Error: {e}")
    st.stop()

# Load feature names saved during training
if not os.path.exists(FEATURES_PATH):
    st.error(f"Missing {FEATURES_PATH}. Please run train_model.py to regenerate it.")
    st.stop()

with open(FEATURES_PATH, 'r', encoding='utf-8') as f:
    expected_cols = [line.strip() for line in f if line.strip()]

st.title('Customer Churn Prediction')
st.write('Upload customer data to predict churn probability.')

uploaded_file = st.file_uploader('Upload CSV', type=['csv'])

if uploaded_file:
    new_data = pd.read_csv(uploaded_file)
    st.write('Uploaded Data (first 5 rows):')
    st.dataframe(new_data.head())

    # -------------- Preprocess to match training --------------
    # One-hot encode 'region' if present (same as training)
    if 'region' in new_data.columns:
        new_data = pd.get_dummies(new_data, columns=['region'], drop_first=True)

    # Remove columns not used in training, and add any missing ones as 0
    # Drop target/id if present (the pipeline during training dropped these)
    for col_to_drop in ['customer_id', 'churn_flag']:
        if col_to_drop in new_data.columns:
            new_data = new_data.drop(columns=[col_to_drop])

    # Add missing training columns as 0
    for col in expected_cols:
        if col not in new_data.columns:
            new_data[col] = 0

    # Keep only the training columns and order them exactly
    new_data = new_data[expected_cols]

    # -------------- Predict churn probability --------------
    try:
        proba = model.predict_proba(new_data)[:, 1]
    except Exception as e:
        st.error(f"Prediction failed. Error: {e}")
        st.stop()

    result_df = pd.DataFrame({
        'churn_probability': proba
    })
    st.subheader('Predictions')
    st.dataframe(result_df)

    # -------------- Feature importance (LogisticRegression coefficients) --------------
    # The classifier is the last step in the pipeline: ('clf', LogisticRegression(...))
    clf = None
    try:
        clf = model.named_steps.get('clf', None)
    except Exception:
        pass

    if clf is not None and hasattr(clf, 'coef_'):
        # coef_ shape: (1, n_features) for binary classification
        coefs = clf.coef_.ravel()
        importance_df = pd.DataFrame({
            'feature': expected_cols,
            'coefficient': coefs.abs(),   # absolute magnitude for "importance"
            'signed_coef': coefs
        }).sort_values(by='coefficient', ascending=False)

        st.subheader('Top Features Impacting Churn (by |coefficient|)')
        st.bar_chart(importance_df.set_index('feature')['coefficient'])
        with st.expander("Show signed coefficients (direction)"):
            st.dataframe(importance_df)
    else:
        st.info("Feature importance unavailable: the classifier doesn't expose coefficients.")
