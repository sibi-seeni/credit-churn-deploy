# predict.py
import pickle
import json
import pandas as pd
from flask import Flask, request, jsonify

# --- 1. Load Artifacts ---
# Load the trained model
with open('model.pkl', 'rb') as f_model:
    model = pickle.load(f_model)
print("Model loaded.")

# Load the encoders
with open('encoders.pkl', 'rb') as f_enc:
    encoders = pickle.load(f_enc)
print("Encoders loaded.")

# Load the feature columns
with open('columns.json', 'r') as f_cols:
    feature_cols = json.load(f_cols)
print("Feature columns loaded.")

# Define categorical columns (must match training)
cat_cols = ['Gender', 'Education_Level', 'Marital_Status', 'Income_Category', 'Card_Category']

# --- 2. Initialize Flask App ---
app = Flask(__name__)

# --- 3. Define Prediction Endpoint ---
@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Get the JSON record from the request
        record = request.get_json()
        
        # Convert to DataFrame
        df = pd.DataFrame([record])

        # Preprocess the new data using loaded encoders
        for col in cat_cols:
            if col in df.columns:
                # Use the loaded encoder for this column
                le = encoders[col]
                df[col] = le.transform(df[col])
            else:
                return jsonify({"error": f"Missing column: {col}"}), 400

        # Ensure columns are in the same order as during training
        df = df[feature_cols]
        
        # Make prediction (probability of churn)
        churn_probability = model.predict_proba(df)[:, 1]
        
        # --- 4. Format and Return Response ---
        response = {
            # **FIX: Convert the NumPy float to a standard Python float**
            'churn_probability': float(churn_probability[0]) 
        }
        return jsonify(response)

    except Exception as e:
        return jsonify({"error": str(e)}), 400

# --- 5. Run the App ---
if __name__ == '__main__':
    # Runs on 0.0.0.0 to be accessible within Docker
    app.run(debug=True, host='0.0.0.0', port=5001)