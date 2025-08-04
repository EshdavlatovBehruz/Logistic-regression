import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression

def train_model():
    df = pd.read_csv('loan_data.csv')
    df['person_emp_exp'] = df['person_emp_exp'].fillna(df['person_emp_exp'].median())
    df.dropna(inplace=True)

    categorical_cols = ['person_home_ownership', 'loan_intent', 'loan_status']
    numerical_cols = df.select_dtypes(include=['Int64', 'float64']).columns.drop('loan_status')

    X = df.drop('loan_status', axis=1)
    y = df['loan_status']

    preprocessor = ColumnTransformer(transformers=[
        ('num', StandardScaler(), numerical_cols),
        ('cat', OneHotEncoder(drop='first'), categorical_cols[:-1])
    ])

    model = Pipeline(steps=[
        ('preprocessor', preprocessor),
        ('classifier', LogisticRegression(solver='liblinear'))
    ])

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)

    log_reg = model.named_steps['classifier']
    feature_names = model.named_steps['preprocessor'].get_feature_names_out()
    coefficients = log_reg.coef_[0]
    coef_df = pd.DataFrame({'Feature': feature_names, 'Coefficient': coefficients})
    coef_df = coef_df.sort_values(by='Coefficient', ascending=False)

    return coef_df, y_test, y_pred
