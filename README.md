# Credit Card Churn Prediction Service: An XGBoost Deployment

This project implements a complete Machine Learning Classification pipeline to predict credit card customer churn. The solution includes data cleaning, feature engineering, model selection, and deployment via Flask and Docker on Google Cloud Run.

This work is highly relevant to my background, as my **previous experience in the financial industry** provided me with first-hand insight into the challenges of customer retention and the substantial costs associated with attrition. This context informed the feature engineering and model evaluation strategies, ensuring a pragmatic and business-focused solution.

## Problem Statement and Relevance

Credit card churn remains a persistent and significant challenge for financial institutions. Losing an existing customer is far more costly than retaining them, due to the expense of new customer acquisition, regulatory costs, and the loss of future revenue streams (e.g., interest, transaction fees).

The goal of this project is to build a highly accurate classification service that can identify customers at high risk of attrition early in their lifecycle. This enables the institution to deploy targeted, cost-effective retention campaigns.

## 📊 Model Selection and Performance

The methodology involved an extensive comparative analysis of multiple classification algorithms to determine the best model for production.

* **Model Comparison:** Several models, including Logistic Regression, Random Forest, Gradient Boosting, **XGBoost**, and **LightGBM**, were evaluated based on metrics critical for churn prediction, such as **AUC (Area Under the Curve)**, **Accuracy** and **F1-Score**.
* **Top Performers:** The ensemble methods, **XGBoost** and **LightGBM**, consistently demonstrated superior performance due to their robust handling of high-dimensional, mixed-type (categorical and numerical) data.
* **Final Model:** **XGBoost** was selected as the final production model due to its high accuracy (validation score $\approx 97.6\%$), stability, and proven ability to handle complex feature interactions efficiently. The model was optimized using `GridSearchCV` to find the best hyperparameters.

## 📂 Repository Structure

```text
├── notebook.ipynb                          # Jupyter Notebook for EDA, feature engineering, and model selection
├── train.py                                # Script to train the model and save artifacts
├── predict.py                              # Flask web application for serving predictions
├── Dockerfile                              # Configuration to containerize the application
├── requirements.txt                        # List of Python dependencies
├── README.md                               # Project documentation
├── .gitignore                              # Files to exclude from version control
└── credit_card_churn.csv                   # Dataset
````

## Setup and Installation

### Prerequisites

  * Python 3.10+
  * Docker (Desktop or Engine)
  * Google Cloud SDK (optional, for deployment)

### 1\. Clone the Repository

```bash
git clone https://github.com/sibi-seeni/credit-churn-deploy.git
cd credit-churn-deploy
```

### 2\. Install Dependencies

It is recommended to use a virtual environment.

```bash
# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows use: venv\Scripts\activate

# Install libraries
pip install -r requirements.txt
```

-----

## Training the Model 🧠

The training script loads the data, performs preprocessing (encoding categorical variables), handles class imbalance, and trains an XGBoost Classifier. It saves the trained model and necessary encoders for the inference service.

1.  Ensure `credit_card_churn.csv` is in the project directory.
2.  Run the training script:

<!-- end list -->

```bash
python train.py
```

**Output Artifacts:**

  * `model.pkl`: The trained XGBoost model.
  * `encoders.pkl`: Label encoders for categorical features.
  * `columns.json`: List of feature columns used during training.
  * `params.json`: Best hyperparameters found during search.

-----

## Running the Application Locally 💻

### Option 1: Running with Python (Flask)

You can run the service directly on your machine for testing.

```bash
python predict.py
```

The service will start on `http://0.0.0.0:5001`.

### Option 2: Running with Docker (Recommended)

Containerization ensures the application runs consistently across different environments.

1.  **Build the Docker Image:**

    ```bash
    docker build -t churn-service .
    ```

2.  **Run the Container:**

    ```bash
    docker run -p 5001:5001 churn-service
    ```

The application is now running inside a container and accessible at `http://localhost:5001`.

-----

## Cloud Deployment

This service has been deployed to **Google Cloud Run** and is publicly accessible.

  * **Deployment URL:** [https://churn-service-app-145522621364.us-central1.run.app](https://churn-service-app-145522621364.us-central1.run.app)

## API Usage

You can send a `POST` request to the `/predict` endpoint with customer data in JSON format to get the churn probability.

### Endpoint

`POST /predict`

### Request Body Example

```json
{
  "Customer_Age": 45,
  "Gender": "M",
  "Dependent_count": 3,
  "Education_Level": "High School",
  "Marital_Status": "Married",
  "Income_Category": "$60K - $80K",
  "Card_Category": "Blue",
  "Months_on_book": 39,
  "Total_Relationship_Count": 5,
  "Months_Inactive_12_mon": 1,
  "Contacts_Count_12_mon": 3,
  "Credit_Limit": 12691.0,
  "Total_Revolving_Bal": 777,
  "Avg_Open_To_Buy": 11914.0,
  "Total_Amt_Chng_Q4_Q1": 1.335,
  "Total_Trans_Amt": 1144,
  "Total_Trans_Ct": 42,
  "Total_Ct_Chng_Q4_Q1": 1.625,
  "Avg_Utilization_Ratio": 0.061
}
```

### Testing with cURL (Live Service)

```bash
curl -X POST https://churn-service-app-145522621364.us-central1.run.app/predict \
-H "Content-Type: application/json" \
-d '{"Customer_Age": 45, "Gender": "M", "Dependent_count": 3, "Education_Level": "High School", "Marital_Status": "Married", "Income_Category": "$60K - $80K", "Card_Category": "Blue", "Months_on_book": 39, "Total_Relationship_Count": 5, "Months_Inactive_12_mon": 1, "Contacts_Count_12_mon": 3, "Credit_Limit": 12691.0, "Total_Revolving_Bal": 777, "Avg_Open_To_Buy": 11914.0, "Total_Amt_Chng_Q4_Q1": 1.335, "Total_Trans_Amt": 1144, "Total_Trans_Ct": 42, "Total_Ct_Chng_Q4_Q1": 1.625, "Avg_Utilization_Ratio": 0.061}'
```

### Response Example

```json
{
  "churn_probability": 0.00024042316363193095
}
```

-----
**Note**: If you are using Windows (Powershell), use this cURL:

```bash
curl -X POST "https://churn-service-app-145522621364.us-central1.run.app/predict" `
  -H "Content-Type: application/json" `
  -d '{\"Customer_Age\": 45, \"Gender\": \"M\", \"Dependent_count\": 3, \"Education_Level\": \"High School\", \"Marital_Status\": \"Married\", \"Income_Category\": \"$60K - $80K\", \"Card_Category\": \"Blue\", \"Months_on_book\": 39, \"Total_Relationship_Count\": 5, \"Months_Inactive_12_mon\": 1, \"Contacts_Count_12_mon\": 3, \"Credit_Limit\": 12691.0, \"Total_Revolving_Bal\": 777, \"Avg_Open_To_Buy\": 11914.0, \"Total_Amt_Chng_Q4_Q1\": 1.335, \"Total_Trans_Amt\": 1144, \"Total_Trans_Ct\": 42, \"Total_Ct_Chng_Q4_Q1\": 1.625, \"Avg_Utilization_Ratio\": 0.061}'
```
