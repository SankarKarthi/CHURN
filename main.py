from fastapi import FastAPI, File, UploadFile, HTTPException, Form, Body
import pandas as pd
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.impute import SimpleImputer
from math import sqrt
import joblib
import numpy as np
import uvicorn
import io
import json
import os
from typing import List

app = FastAPI()

# Initialize global variables
df = pd.DataFrame()
label_encoders = {}  # Dictionary to store encoders for categorical columns
model_directory = "models"  # Directory to store model files

# Create directory if it doesn't exist
if not os.path.exists(model_directory):
    os.makedirs(model_directory)

@app.post("/upload/")
async def upload_file(file: UploadFile = File(...)):
    global df
    contents = await file.read()
    df = pd.read_csv(io.BytesIO(contents))
    return {"filename": file.filename, "status": "File uploaded successfully!"}

@app.post("/upload_model/")
async def upload_model(file: UploadFile = File(...)):
    model_file_path = os.path.join(model_directory, file.filename)
    with open(model_file_path, "wb") as f:
        f.write(await file.read())
    return {"filename": file.filename, "status": "Model uploaded successfully!"}

@app.get("/data/")
async def show_data():
    global df
    if df.empty:
        return {"error": "No dataset uploaded."}
    return df.to_dict()

@app.get("/columns/")
async def get_columns():
    global df
    if df.empty:
        return {"error": "No dataset uploaded."}
    columns = df.columns.tolist()
    return {"columns": columns}

@app.get("/preprocess/")
async def preprocess_data():
    global df
    if df.empty:
        return {"error": "No dataset uploaded."}

    # Fill missing values with the mean of numeric columns
    numeric_cols = df.select_dtypes(include=np.number).columns
    df[numeric_cols] = df[numeric_cols].fillna(df[numeric_cols].mean())

    return {"status": "Missing values filled with mean for numeric columns"}

@app.get("/statistics/")
async def get_statistics():
    global df
    if df.empty:
        return {"error": "No dataset uploaded."}

    statistics = {
        "max": df.max(numeric_only=True).to_dict(),
        "min": df.min(numeric_only=True).to_dict(),
        "mean": df.mean(numeric_only=True).to_dict(),
        "description": df.describe().to_dict()
    }

    return statistics

@app.post("/train_model/")
async def train_model(
    dependent_variable: str = Form(...), 
    independent_variables: str = Form(...), 
    model_choice: str = Form(...)
):
    global df, label_encoders
    if df.empty:
        raise HTTPException(status_code=400, detail="No dataset uploaded.")
    
    if not dependent_variable or not independent_variables:
        raise HTTPException(status_code=400, detail="Dependent and independent variables must be selected.")

    # Convert independent_variables from JSON string to Python list
    try:
        independent_variables_list = json.loads(independent_variables)
    except json.JSONDecodeError:
        raise HTTPException(status_code=400, detail="Invalid format for independent variables.")

    # Check if all independent variables exist in the dataset
    for column in independent_variables_list:
        if column not in df.columns:
            raise HTTPException(status_code=400, detail=f"Column '{column}' not found in the dataset.")

    # Drop rows with missing dependent variable values
    df.dropna(subset=[dependent_variable], inplace=True)

    # Encode categorical variables except the dependent variable
    for column in df.columns:
        if df[column].dtype == 'object' and column != dependent_variable:
            label_encoders[column] = LabelEncoder()
            df[column] = label_encoders[column].fit_transform(df[column])

    X = df[independent_variables_list].values
    y = df[dependent_variable].values

    # Impute missing values in X
    imputer = SimpleImputer(strategy='mean')
    X = imputer.fit_transform(X)

    # Split the data
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=42)

    # Select and train the model
    if model_choice == "Multiple Linear Regression":
        model = LinearRegression()
    elif model_choice == "Random Forest Classifier":
        model = RandomForestClassifier()
    elif model_choice == "Random Forest Regressor":
        model = RandomForestRegressor()
    else:
        raise HTTPException(status_code=400, detail="Invalid model choice")

    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)

    # Evaluate the model
    if isinstance(model, RandomForestRegressor):
        rms = sqrt(mean_squared_error(y_test, y_pred))
        accuracy_score = None
    elif isinstance(model, RandomForestClassifier):
        rms = None
        accuracy_score = model.score(X_test, y_test)

    # Save the model
    model_file = os.path.join(model_directory, f'{model_choice.lower().replace(" ", "_")}_model.pkl')
    joblib.dump(model, model_file)

    return {
        "model_saved_as": model_file,
        "rms": rms,
        "accuracy_score": accuracy_score
    }



@app.post("/predict/")
async def predict(input_data: dict, model_file: str):
    global label_encoders
    try:
        model = joblib.load(model_file)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error loading model: {str(e)}")

    try:
        input_df = pd.DataFrame([input_data])

        # Handle categorical columns
        for column in input_df.columns:
            if column in label_encoders:
                if input_df[column].dtype == 'object':
                    input_df[column] = label_encoders[column].transform(input_df[column].astype(str))
        
        # Separate numeric and non-numeric columns
        numeric_cols = input_df.select_dtypes(include=np.number).columns
        non_numeric_cols = input_df.select_dtypes(exclude=np.number).columns

        # Impute missing values only for numeric columns
        imputer = SimpleImputer(strategy='mean')
        input_df[numeric_cols] = imputer.fit_transform(input_df[numeric_cols])

        # Ensure all columns are in the correct format for the model
        for column in non_numeric_cols:
            if column not in label_encoders:
                raise HTTPException(status_code=400, detail=f"Column '{column}' not encoded.")
        
        # Prepare the input for prediction
        input_df = input_df.values

        # Predict using the loaded model
        prediction = model.predict(input_df)
        
        return {"prediction": prediction[0]}
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error processing prediction: {str(e)}")



if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
