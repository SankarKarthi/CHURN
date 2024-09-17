import base64
import numpy as np
import seaborn as sn
import matplotlib.pyplot as plt
import streamlit as st
import pandas as pd
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.impute import SimpleImputer
from math import sqrt
import joblib
from sklearn.impute import SimpleImputer
import seaborn as sns

df = pd.DataFrame()

st.set_page_config(
    page_title="Data Dumper",
    page_icon=":bar_chart:",
    layout="wide"
)

def home():
    st.markdown("""
        <style>
        body {
          background: #ff0099; 
          background: -webkit-linear-gradient(to right, #ff0099, #493240); 
          background: linear-gradient(to right, #ff0099, #493240); 
        }
        </style>
    """, unsafe_allow_html=True)

    st.markdown("<h1 style='text-align: center; color: white;'>Data Analysis Made Easy</h1>", unsafe_allow_html=True)
    st.markdown("<p style='text-align: center; color: white;'>Without big data analytics, companies are blind and deaf, wandering out onto the web like a deer on the freeway. - Geoffrey Moore</p>", unsafe_allow_html=True)
    st.markdown("<p style='text-align: justify; color: white;'>Writing up the results of a data analysis is not a skill that anyone is born with. It requires practice and, at least in the beginning, a bit of guidance. Thus, we provide easy analysis for the users without the knowledge of coding so that they can infer knowledge from existing data.</p>", unsafe_allow_html=True)
    st.markdown("<p style='text-align: justify; color: white;'>Data is the future and the future is now! Every mouse click, keyboard button press, swipe or tap is used to shape business decisions. Everything is about data these days. Data is information and information is power.</p>", unsafe_allow_html=True)

    st.markdown("<h2 style='text-align: center; color: white;'>Contact</h2>", unsafe_allow_html=True)
    st.write(":telephone_receiver: 9361381816")
    st.write(":e-mail: contact@sankarkarthikeyanp662.com")
    st.write(":iphone: [https://www.linkedin.com/in/sankar-karthikeyan/](https://www.linkedin.com/in/sankar-karthikeyan/)")
    st.write(":round_pushpin: Coimbatore, India")

uploaded_file = st.file_uploader("Upload a file here")
if uploaded_file is not None:
    df = pd.read_csv(uploaded_file, encoding='utf-8')

def upload():
    global df
    if df.empty:
        st.write("Upload a dataset first.")
        return
    
    st.write("The given data is... :")
    st.write(df)

    st.write("Checking null values from the given data:")
    st.write(df.isnull().sum())

    # Fill null values
    numeric_cols = df.select_dtypes(include=np.number).columns
    df[numeric_cols] = df[numeric_cols].fillna(df[numeric_cols].mean())

    st.write("After filling null values:")
    st.write(df)

    Q1 = df[numeric_cols].quantile(0.25)
    Q3 = df[numeric_cols].quantile(0.75)
    IQR = Q3 - Q1
    
    outliers = ((df[numeric_cols] < (Q1 - 1.5 * IQR)) | (df[numeric_cols] > (Q3 + 1.5 * IQR)))
    
    df = df[~outliers.any(axis=1)]

    st.write("After removing outliers:")
    st.write(df)

    def get_table_download_link_csv():
        csv = df.to_csv().encode()
        b64 = base64.b64encode(csv).decode()
        href = f'<a href="data:file/csv;base64,{b64}" download="cleaned_data.csv" target="_blank">Download csv file with filled null values and removed outliers</a>'
        return href

    st.markdown(get_table_download_link_csv(), unsafe_allow_html=True)

    st.write("Description of numerical columns:")
    st.write(df.describe())

    st.write("Maximum values in each column:")
    st.write(df.max(numeric_only=True))
    st.write("Minimum values in each column:")
    st.write(df.min(numeric_only=True))
    st.write("Average values in each column:")
    st.write(df.mean(numeric_only=True))

    fig, ax = plt.subplots(figsize=(6, 6))
    corr_df = df.select_dtypes(include=np.number).corr()
    sn.heatmap(corr_df, ax=ax)
    st.write("Heatmap showing correlations between numeric columns:")
    st.pyplot(fig)


def eda():
    global df
    if df.empty:
        st.write("Upload a dataset first.")
        return
    
    st.markdown("# Exploratory Data Analysis")

    st.subheader("Distribution of Numeric Columns")
    numeric_cols = df.select_dtypes(include=np.number).columns
    for col in numeric_cols:
        fig, ax = plt.subplots(figsize=(10, 4))
        sns.histplot(df[col], bins=30, kde=True, ax=ax)
        ax.set_title(f'Distribution of {col}')
        st.pyplot(fig)
    
    st.subheader("Pairplot of Numeric Columns")
    fig = plt.figure(figsize=(10, 10))
    sns.pairplot(df[numeric_cols])
    st.pyplot(fig)
    
    st.subheader("Correlation Heatmap")
    fig, ax = plt.subplots(figsize=(10, 8))
    corr_df = df[numeric_cols].corr()
    sns.heatmap(corr_df, annot=True, cmap='coolwarm', ax=ax)
    st.pyplot(fig)

    st.subheader("Distribution of Categorical Columns")
    categorical_cols = df.select_dtypes(include=['object']).columns
    for col in categorical_cols:
        st.write(f"**{col}**")
        fig, ax = plt.subplots(figsize=(10, 4))
        sns.countplot(y=col, data=df, ax=ax)
        st.pyplot(fig)

def mlr():
    global df
    if df.empty:
        st.write("Upload a dataset first.")
        return
    
    st.markdown(f"# {list(page_names_to_funcs.keys())[2]}")

    model_options = ("Random Forest Classifier", "Random Forest Regressor")
    model_choice = st.radio("Select Model:", model_options)
    
    dependent_variable = 'churn'
    independent_variables = st.multiselect('Select Independent Variables:', df.columns)

    if not dependent_variable or not independent_variables:
        st.write("Please select the independent variables.")
        return

    df.dropna(subset=[dependent_variable], inplace=True)

    label_encoders = {}
    for column in df.columns:
        if df[column].dtype == 'object':
            label_encoders[column] = LabelEncoder()
            df[column] = label_encoders[column].fit_transform(df[column])

    X = df[independent_variables].values
    y = df[dependent_variable].values

    imputer = SimpleImputer(strategy='mean')
    X = imputer.fit_transform(X)

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33)

    
    if model_choice == "Random Forest Classifier":
        model = RandomForestClassifier()
    elif model_choice == "Random Forest Regressor":
        model = RandomForestRegressor()

    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)

    if isinstance(model, RandomForestRegressor):
        rms = sqrt(mean_squared_error(y_test, y_pred))
        st.write(f"### {model_choice}")
        st.write("Root Mean Square Error:", rms)
    elif isinstance(model, RandomForestClassifier):
        accuracy = model.score(X_test, y_test)
        st.write(f"### {model_choice}")
        st.write("Accuracy Score:", accuracy)

    model_file = f'{model_choice.lower().replace(" ", "_")}_model.pkl'
    joblib.dump(model, model_file)

    st.write("Model Saved as:", model_file)

    st.markdown("## Predict Output")

    with st.form(key='prediction_form'):
        st.write("Enter values for independent variables to predict output:")
        input_values = {}
        for var in independent_variables:
            input_values[var] = st.number_input(f"{var}:")
        submitted = st.form_submit_button("Predict")

    if submitted:
        input_data = pd.DataFrame([input_values])
        for column in input_data.columns:
            if input_data[column].dtype == 'object':
                input_data[column] = label_encoders[column].transform(input_data[column])
        st.write("Input Data for Prediction:", input_data)  
        prediction = model.predict(input_data)
        st.write("Predicted Output:", prediction[0])


def feature():
    global df
    if df.empty:
        st.write("Upload a dataset first.")
        return
    
    st.markdown("# Feature Extraction")

    st.subheader("Delete Unwanted Columns")
    columns_to_delete = st.multiselect("Select columns to delete:", df.columns)
    df = df.drop(columns=columns_to_delete, axis=1)

    label_encoders = {}
    categorical_columns = df.select_dtypes(include=['object']).columns
    
    for column in categorical_columns:
        label_encoders[column] = LabelEncoder()
        df[column] = label_encoders[column].fit_transform(df[column])
    
    feature_options = df.columns.tolist()
    
    feature_name = st.selectbox('Select the feature you want to predict:', feature_options)
    
    if not feature_name:
        st.write("Please select a feature.")
        return
    
    y = df[feature_name].values
    imputer = SimpleImputer(strategy='mean')
    y = imputer.fit_transform(y.reshape(-1, 1)).ravel()
    
    imputer = SimpleImputer(strategy='mean')
    X = imputer.fit_transform(df.drop([feature_name], axis=1))
    
    model = RandomForestRegressor()
    model.fit(X, y)
    
    feature_importances = model.feature_importances_
    
    st.write("Feature Importances:")
    for feature, importance in zip(df.columns, feature_importances):
        st.write(f"{feature}: {importance}")
    
    st.info('Features with higher importance have higher dependencies on the target variable', icon="ℹ️")

def random_forest_explanation():
    """
    Provides a brief explanation of why Random Forest is robust to overfitting, outliers, and categorical values.
    """
    
    explanation = """
    **Random Forest Robustness Explanation**

    **1. Robustness to Overfitting:**
    Random Forest mitigates overfitting by averaging predictions from multiple decision trees. Each tree is trained on a random subset of the data and features, which reduces the risk of overfitting to any single subset. The ensemble approach aggregates these diverse trees, leading to a model that generalizes well to unseen data.

    **2. Robustness to Outliers:**
    Decision trees, the base of Random Forest, handle outliers better than some other models. They create splits based on feature values, and outliers generally have less impact on these splits. The Random Forest further diminishes the effect of outliers by averaging predictions from multiple trees, which reduces their influence.

    **3. Handling Categorical Values:**
    Random Forest can handle categorical features natively. Decision trees within the Random Forest can split on categorical variables by considering each category as a potential split point. This ability allows Random Forest to work effectively with datasets that include categorical variables without requiring extensive preprocessing.

    In summary, Random Forest's ensemble learning, its inherent handling of categorical features, and its aggregation of multiple trees contribute to its robustness against overfitting, outliers, and the complexities of categorical data.
    """
    
    return explanation

def display_explanation():
    st.title("Random Forest Explanation")
    st.markdown(random_forest_explanation(), unsafe_allow_html=True)

    


page_names_to_funcs = {
    "Main": home,
    "Data PreProcessing": upload,
    "EDA" : eda,
    "Regression and Prediction": mlr,
    "Feature Extraction": feature,
    "Explanation" : display_explanation
}

# Allow the user to select a demo from the sidebar
demo_name = st.sidebar.selectbox("Choose a demo", page_names_to_funcs.keys())

# Execute the selected demo function
page_names_to_funcs[demo_name]()
