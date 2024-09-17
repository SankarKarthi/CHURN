# Data Dumper

# FASTAPI
Have implemented FastAPI in this project but due to some unfinished EDA I couldn't record the FastAPI user interface. Hence I have done it using StreamLit

``` main.py,  model.py, and  random_forest_classifie.pkl ``` are part of FastAPI

## Overview

**Data Dumper** is a Streamlit-based application designed to facilitate data analysis and machine learning tasks. It provides a user-friendly interface for data preprocessing, feature extraction, and model training, along with explanations of Random Forest's robustness. The app aims to simplify data analysis tasks for users without coding experience and offer insights into machine learning models.

## Features

1. **Home Page**: Introduction to the application, with contact details and a motivational quote on data analysis.
2. **Data PreProcessing**:
   - Upload CSV files.
   - View data, check for null values, and fill missing values.
   - Detect and remove outliers.
   - Download the cleaned dataset.
   - View descriptive statistics and heatmaps of correlations.
3. **Regression and Prediction**:
   - Train Random Forest Classifier or Regressor models.
   - Save the trained model.
   - Predict outcomes based on user-provided input values.
4. **Feature Extraction**:
   - Delete unwanted columns.
   - Analyze feature importance using Random Forest.
5. **Explanation**:
   - Provides a detailed explanation of Random Forest's robustness to overfitting, outliers, and categorical values.

## Installation

To run this app locally, follow these steps:

1. **Clone the repository**:

    ```bash
    git clone <repository-url>
    cd <repository-directory>
    ```

2. **Install the required packages**:

    ```bash
    pip install -r requirements.txt
    ```

3. **Run the Streamlit app**:

    ```bash
    streamlit run app.py
    ```

4. **Run the FastAPI**
   ```bash
   uvicorn main:app --reload
   ```

    Replace `app.py` with the name of your main Streamlit script if different.

4. **Run the FastAPI**
    ```
    uvicorn main:app --reload
    ```

## Requirements

The `requirements.txt` file includes the necessary Python packages:

- `base64`
- `numpy`
- `seaborn`
- `matplotlib`
- `streamlit`
- `pandas`
- `scikit-learn`
- `joblib`

## Usage

1. **Home Page**: Provides an introduction and contact information.
2. **Data PreProcessing**:
   - Upload a CSV file to preprocess.
   - The app checks for and fills null values, removes outliers, and allows downloading the cleaned data.
3. **Regression and Prediction**:
   - Choose between a Random Forest Classifier and Regressor.
   - Select independent variables and a dependent variable for training.
   - Enter values to predict outcomes using the trained model.
4. **Feature Extraction**:
   - Select columns to delete.
   - Analyze feature importance based on the selected target feature.
5. **Explanation**:
   - View a detailed explanation of why Random Forest models are robust to various data issues.

## Contact

For any inquiries or issues, please contact:

- Phone: 9361381816
- Email: sankarkarthikeyan066@gmail.com
- LinkedIn: [Sankar Karthikeyan](https://www.linkedin.com/in/sankar-karthikeyan/)
- Location: Coimbatore, India

