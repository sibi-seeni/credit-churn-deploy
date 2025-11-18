# Credit Card Churn Prediction Service

This project implements an end-to-end Machine Learning pipeline to predict credit card customer churn. It includes data preparation, model training with XGBoost, and a production-ready web service deployed using Docker and Google Cloud Run.

## 📋 Problem Description

Credit card churn is a significant concern for financial institutions. Losing customers results in lost revenue and market share. This project aims to predict whether a customer is likely to cancel their credit card ("churn") based on their transaction history, demographics, and interaction data.

By identifying at-risk customers early, banks can proactively engage them with retention strategies.

## 📂 Repository Structure

```text
├── credit-card-churn-classification.ipynb  # Jupyter Notebook for EDA, feature engineering, and model selection
├── train.py                                # Script to train the model and save artifacts
├── predict.py                              # Flask web application for serving predictions
├── Dockerfile                              # Configuration to containerize the application
├── requirements.txt                        # List of Python dependencies
├── README.md                               # Project documentation
├── .gitignore                              # Files to exclude from version control
└── data/
    └── credit_card_churn.csv               # Dataset (if committed) or instructions to download
````

## 🚀 Setup and Installation

### Prerequisites

  * Python 3.10+
  * Docker (Desktop or Engine)
  * Google Cloud SDK (optional, for deployment)

### 1\. Clone the Repository

```bash
git clone <YOUR_REPO_URL>
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

## 🧠 Training the Model

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

## 💻 Running the Application Locally

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

## ☁️ Deployment

This service has been deployed to **Google Cloud Run** and is publicly accessible.

  * **Deployment URL:** [https://churn-service-app-145522621364.us-central1.run.app](https://churn-service-app-145522621364.us-central1.run.app)

### Deployment Steps (Reference)

For reference, the following commands were used to deploy the Docker container to Google Cloud Run:

```bash
# 1. Build image for AMD64 architecture (required for Cloud Run if building on Mac M1/M2)
docker build --platform linux/amd64 -t churn-service .

# 2. Tag the image for Google Artifact Registry
docker tag churn-service us-central1-docker.pkg.dev/ml-camp-1/churn-service-repo/churn-model:latest

# 3. Push the image
docker push us-central1-docker.pkg.dev/ml-camp-1/churn-service-repo/churn-model:latest

# 4. Deploy to Cloud Run
gcloud run deploy churn-service-app \
  --image us-central1-docker.pkg.dev/ml-camp-1/churn-service-repo/churn-model:latest \
  --region us-central1 \
  --port 5001 \
  --allow-unauthenticated
```

-----

## 🔌 API Usage

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
curl -X POST [https://churn-service-app-145522621364.us-central1.run.app/predict](https://churn-service-app-145522621364.us-central1.run.app/predict) \
-H "Content-Type: application/json" \
-d '{"Customer_Age": 45, "Gender": "M", "Dependent_count": 3, "Education_Level": "High School", "Marital_Status": "Married", "Income_Category": "$60K - $80K", "Card_Category": "Blue", "Months_on_book": 39, "Total_Relationship_Count": 5, "Months_Inactive_12_mon": 1, "Contacts_Count_12_mon": 3, "Credit_Limit": 12691.0, "Total_Revolving_Bal": 777, "Avg_Open_To_Buy": 11914.0, "Total_Amt_Chng_Q4_Q1": 1.335, "Total_Trans_Amt": 1144, "Total_Trans_Ct": 42, "Total_Ct_Chng_Q4_Q1": 1.625, "Avg_Utilization_Ratio": 0.061}'
```

### Response Example

```json
{
  "churn_probability": 0.00024042316363193095
}
```
