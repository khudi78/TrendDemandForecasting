import streamlit as st
import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, accuracy_score
import joblib
import zipfile
import os

# Function to preprocess the data
def preprocess_data(df):
    # Drop rows with missing 'Description' or any other critical columns
    df.dropna(subset=['Description'], inplace=True)

    # Encode categorical features
    label_encoders = {}
    for column in ['ProductName', 'ProductBrand', 'Gender', 'PrimaryColor']:
        le = LabelEncoder()
        df[column] = le.fit_transform(df[column].astype(str))
        label_encoders[column] = le

    # Normalize numerical features
    scaler = StandardScaler()
    df[['Price (INR)', 'NumImages']] = scaler.fit_transform(df[['Price (INR)', 'NumImages']])

    # Preprocess the 'Description' column using TF-IDF Vectorizer
    tfidf_vectorizer = TfidfVectorizer(stop_words='english', max_features=5000)
    X_description = tfidf_vectorizer.fit_transform(df['Description']).toarray()

    # Combine all features
    X_numerical = df[['ProductName', 'ProductBrand', 'Gender', 'Price (INR)', 'NumImages']].values
    X = np.hstack([X_numerical, X_description])
    y = df['PrimaryColor']

    return X, y, label_encoders, tfidf_vectorizer, scaler

# Function to build and train the model
def build_and_train_model(X_train, y_train):
    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)
    return model

# Function to evaluate the model
def evaluate_model(model, X_test, y_test, label_encoders):
    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    report = classification_report(y_test, y_pred)
    if 'PrimaryColor' in label_encoders:
        y_test_original = label_encoders['PrimaryColor'].inverse_transform(y_test)
        y_pred_original = label_encoders['PrimaryColor'].inverse_transform(y_pred)
        predictions_df = pd.DataFrame({
            'True Primary Color': y_test_original,
            'Predicted Primary Color': y_pred_original
        })
    else:
        predictions_df = pd.DataFrame({
            'True Labels': y_test,
            'Predicted Labels': y_pred
        })
    return accuracy, report, predictions_df

# Streamlit UI
st.title('Fashion Trend Prediction')

# File upload
uploaded_file = st.file_uploader("Upload a ZIP file containing the fashion data", type=["zip"])
if uploaded_file is not None:
    # Extract the ZIP file
    with zipfile.ZipFile(uploaded_file, 'r') as zip_ref:
        zip_ref.extractall('/content/fashion_data')

    # Check extracted files
    extracted_files = os.listdir('/content/fashion_data')
    st.write("Extracted files:", extracted_files)

    # Load the CSV file
    csv_filename = [f for f in extracted_files if f.endswith('.csv')][0]
    data = pd.read_csv(f'/content/fashion_data/{csv_filename}')

    # Display the first few rows of the dataset
    st.write("First few rows of the dataset:")
    st.write(data.head())

    # Preprocess the data
    X, y, label_encoders, tfidf_vectorizer, scaler = preprocess_data(data)

    # Split the data
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Build and train the model
    model = build_and_train_model(X_train, y_train)

    # Evaluate the model
    accuracy, report, predictions_df = evaluate_model(model, X_test, y_test, label_encoders)

    # Display the results
    st.write("Accuracy Score:", accuracy)
    st.write("Classification Report:")
    st.text(report)

    # Display a sample of predictions
    st.write("Sample of Predictions:")
    st.write(predictions_df.head())

    # Save the model (optional)
    joblib.dump(model, '/content/fashion_trend_model.pkl')
    st.write("Model saved as 'fashion_trend_model.pkl'")
