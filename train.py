# train.py
import pandas as pd
import numpy as np
import pickle
import json
import os
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import LabelEncoder
from xgboost import XGBClassifier

print("Starting training script...")

# --- 1. Load Data ---
try:
    data = pd.read_csv("credit_card_churn.csv")
    print("Data loaded successfully.")
except FileNotFoundError:
    print("Error: 'credit_card_churn.csv' not found.")
    print("Please download the dataset or place it in the correct directory.")
    exit()

# --- 2. Preprocessing ---
# Drop CLIENTNUM
if 'CLIENTNUM' in data.columns:
    data = data.drop(columns=['CLIENTNUM'])

# Map target variable
mapping = {'Attrited Customer': 1, 'Existing Customer': 0}
data['Attrition_Flag'] = data['Attrition_Flag'].replace(mapping)
print("Preprocessing: Target variable mapped.")

# Define categorical columns
cat_cols = ['Gender', 'Education_Level', 'Marital_Status', 'Income_Category', 'Card_Category']

# Apply LabelEncoder
encoders = {}
for col in cat_cols:
    le = LabelEncoder()
    data[col] = le.fit_transform(data[col])
    encoders[col] = le
    print(f"Preprocessing: '{col}' encoded.")

# --- 3. Feature and Target Split ---
X = data.drop("Attrition_Flag", axis=1)
y = data["Attrition_Flag"]

# Save feature columns
feature_cols = X.columns.tolist()
with open('columns.json', 'w') as f:
    json.dump(feature_cols, f)
print("Feature columns saved to 'columns.json'.")

# Save encoders
with open('encoders.pkl', 'wb') as f:
    pickle.dump(encoders, f)
print("Encoders saved to 'encoders.pkl'.")

# --- 4. Train-Test Split (for GridSearchCV) ---
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=16)
print(f"Data split into training ({X_train.shape[0]} samples) and test ({X_test.shape[0]} samples).")

# --- 5. Model Training (XGBoost with GridSearchCV) ---
print("Starting GridSearchCV for XGBoost...")
xgb_model = XGBClassifier(objective='binary:logistic', use_label_encoder=False, eval_metric='logloss')

# Parameters from your notebook (cell 43)
param_grid_xgb = {
    'n_estimators': [100, 300, 500],
    'max_depth': [3, 5, 7],
    'learning_rate': [0.01, 0.05, 0.1]
}

grid_search_xgb = GridSearchCV(estimator=xgb_model, param_grid=param_grid_xgb, cv=5, n_jobs=-1, verbose=2)
grid_search_xgb.fit(X_train, y_train)

# Get the best model and parameters
best_xgb_model = grid_search_xgb.best_estimator_
best_params = grid_search_xgb.best_params_

print(f"GridSearchCV finished. Best parameters: {best_params}")

# --- 6. Save Final Model & Parameters ---
# Save the best model
with open('model.pkl', 'wb') as f:
    pickle.dump(best_xgb_model, f)
print("Best model saved to 'model.pkl'.")

# Save the best parameters
with open('params.json', 'w') as f:
    json.dump(best_params, f, indent=4)
print("Best parameters saved to 'params.json'.")

print("Training script finished successfully.")