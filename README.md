# Fashion Trend Prediction

This repository contains a Streamlit application for predicting fashion trends using machine learning. The application allows users to upload a ZIP file containing fashion data, preprocesses the data, trains a model, and displays the evaluation results.

## Features

- Upload a ZIP file containing the fashion data
- Preprocess the data (handling missing values, encoding categorical features, normalizing numerical features)
- Train a RandomForestClassifier to predict fashion trends
- Display the accuracy and classification report
- Show sample predictions
- Save the trained model

## Requirements

- Python 3.12 or higher
- `streamlit`
- `pandas`
- `numpy`
- `scikit-learn`
- `joblib`

## Installation

1. Clone the repository:

   ```bash
   git clone https://github.com/khudi78/TrendDemandForecasting.git
   cd TrendDemandForecasting
   
2. Create and activate a virtual environment (optional but recommended):

   ```bash
   python -m venv env
   source env/bin/activate  # On Windows, use `env\Scripts\activate`
   
3. Install the required packages:

   ```bash
    pip install -r requirements.txt

4. Run the Streamlit app:

   ```bash
    streamlit run app.py
