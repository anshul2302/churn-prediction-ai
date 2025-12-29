import streamlit as st
import pandas as pd
import xgboost as xgb
import pickle

# Load trained model
model = pickle.load(open('model.pkl', 'rb'))

st.title('Customer Churn Prediction')
st.write('Upload customer data to predict churn probability.')

uploaded_file = st.file_uploader('Upload CSV', type=['csv'])

if uploaded_file:
    new_data = pd.read_csv(uploaded_file)
    st.write('Uploaded Data:', new_data.head())

    # Preprocess (dummy encoding for region if exists)
    if 'region' in new_data.columns:
        new_data = pd.get_dummies(new_data, columns=['region'], drop_first=True)

    # Align columns with training data
    expected_cols = [col for col in model.get_booster().feature_names]
    for col in expected_cols:
        if col not in new_data.columns:
            new_data[col] = 0
    new_data = new_data[expected_cols]

    # Predict churn probability
    predictions = model.predict_proba(new_data)[:, 1]
    new_data['churn_probability'] = predictions
    st.write('Predictions:', new_data[['churn_probability']])

    # Display feature importance
    st.subheader('Top Features Impacting Churn')
    importance = model.feature_importances_
    importance_df = pd.DataFrame({'feature': expected_cols, 'importance': importance}).sort_values(by='importance', ascending=False)
    st.bar_chart(importance_df.set_index('feature'))
