import pandas as pd
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score

# Function to load datasets
def load_dataset(file_path):
    data = pd.read_csv(file_path)
    return data

# Data preprocessing function
def preprocess_data(df):
    # Handle missing values
    df = df.fillna(df.mean())
    
    # Encode categorical variables
    label_encoders = {}
    for column in df.select_dtypes(include=['object']).columns:
        le = LabelEncoder()
        df[column] = le.fit_transform(df[column])
        label_encoders[column] = le
    
    # Feature scaling
    scaler = StandardScaler()
    df[df.columns] = scaler.fit_transform(df[df.columns])
    
    return df, label_encoders, scaler

# Function to train the model
def train_model(df, target='churn'):
    X = df.drop(columns=[target])
    y = df[target]
    
    # Split into training and test sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
    
    # Train a Random Forest Classifier
    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)
    
    # Evaluate the model
    predictions = model.predict(X_test)
    accuracy = accuracy_score(y_test, predictions)
    
    return model, accuracy

# Function to handle multiple datasets
def process_and_train(file_path):
    data = load_dataset(file_path)
    processed_data, _, _ = preprocess_data(data)
    model, acc = train_model(processed_data)
    print(f"Model trained on {file_path} with accuracy: {acc}")
    return model
