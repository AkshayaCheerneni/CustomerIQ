# CustomerIQ — Customer Churn, CLV Prediction & Customer Segmentation

CustomerIQ is a machine learning project built to understand customer behavior and turn it into useful business insights.

The project focuses on three main questions:

* Which customers are likely to churn?
* What is the expected Customer Lifetime Value (CLV)?
* What different types of customers exist in the dataset?

I worked through the project from data cleaning and analysis to machine learning, clustering, API development, dashboard creation, testing, and GitHub Actions.

---

## What the Project Does

CustomerIQ takes customer data and processes it through several stages:

```text
Customer Data
      ↓
Data Cleaning & Analysis
      ↓
Feature Engineering
      ↓
Machine Learning
      ↓
Churn Prediction
CLV Prediction
Customer Segmentation
      ↓
FastAPI
      ↓
Dashboard & Business Insights
```

The project uses the same customer dataset across the analysis, machine learning models, clustering, API, and dashboard.

---

## Main Features

* Customer data cleaning and preprocessing
* Exploratory data analysis
* Missing-value and duplicate checks
* Statistical analysis
* Feature engineering
* Customer churn prediction
* Customer Lifetime Value prediction
* K-Means customer segmentation
* Cluster profiling
* Model comparison
* FastAPI REST API
* Swagger API testing
* Customer analytics dashboard
* Saved machine learning models
* Automated testing with Pytest
* Code quality checks with Ruff
* GitHub Actions CI

---

# 1. Customer Churn Prediction

I tested three classification models:

* Logistic Regression
* Decision Tree
* Random Forest

The models were evaluated using:

* Accuracy
* Precision
* Recall
* F1 Score

### Classification Results

| Model               | Accuracy | Precision | Recall | F1 Score |
| ------------------- | -------: | --------: | -----: | -------: |
| Logistic Regression |   0.7015 |    0.0480 | 0.5686 |   0.0885 |
| Decision Tree       |   0.9510 |    0.0392 | 0.0392 |   0.0392 |
| Random Forest       |   0.9745 |    0.0000 | 0.0000 |   0.0000 |

One important thing I found during the project is that **accuracy is not enough for this churn problem**.

The Decision Tree and Random Forest have higher accuracy, but they perform very poorly at actually identifying churned customers.

Logistic Regression has lower accuracy, but much better recall and F1 score.

For this reason, **Logistic Regression is the more useful classification model for the current dataset**.

---

# 2. Customer Lifetime Value Prediction

The second part of the project predicts Customer Lifetime Value.

I tested:

* Linear Regression
* Random Forest
* Decision Tree

The models were compared using:

* MAE
* RMSE
* R²

### Regression Results

| Model             |      MAE |      RMSE |     R² |
| ----------------- | -------: | --------: | -----: |
| Linear Regression | 5,560.48 |  7,317.30 | 0.9674 |
| Random Forest     | 6,196.95 |  7,899.21 | 0.9620 |
| Decision Tree     | 8,961.30 | 11,312.71 | 0.9220 |

**Linear Regression performed best on all three evaluation metrics.**

Its R² score was approximately **0.9674**, while its MAE and RMSE were also the lowest among the tested models.

So the current project uses **Linear Regression as the preferred CLV model based on the evaluation results**.

---

# 3. Customer Segmentation

For customer segmentation, I used **K-Means clustering**.

The clustering model uses these features:

* Tenure
* Monthly Spend
* Login Frequency
* Support Tickets
* Product Usage
* Customer Lifetime Value

The features were standardized using `StandardScaler` before applying K-Means.

The final clustering solution contains **4 customer segments**.

### Customer Segments

| Cluster   | Customers | Avg. Tenure | Avg. Monthly Spend |   Avg. CLV | Churn Rate |
| --------- | --------: | ----------: | -----------------: | ---------: | ---------: |
| Cluster 0 |     2,262 |       52.44 |           1,291.07 |  85,748.81 |      1.50% |
| Cluster 1 |     2,593 |       18.16 |           1,910.93 |  72,553.63 |      2.04% |
| Cluster 2 |     2,772 |       53.82 |           2,400.21 | 144,961.55 |      1.48% |
| Cluster 3 |     2,373 |       21.28 |           1,735.81 |  60,656.82 |      5.44% |

### Segment Observations

**Cluster 2** is the highest-value group. It has the highest average monthly spend and the highest average CLV.

**Cluster 3** has the highest churn rate at approximately **5.44%**. It also has lower product usage and login frequency compared with the other groups.

These segments make it easier to think about customers differently instead of treating the entire customer base as one group.

---

# 4. Feature Engineering

I created additional features from the original customer information to make the data more useful for analysis and machine learning.

Some of the engineered features are:

* Average Monthly Spend
* Usage per Month
* Support Ticket Rate
* Age Group
* Tenure Group

These features are used by the machine learning pipeline and help represent customer behavior in a more useful format.

---

# 5. Technology Used

### Python

* Python
* Pandas
* NumPy

### Machine Learning

* Scikit-learn
* Logistic Regression
* Decision Tree
* Random Forest
* Linear Regression
* K-Means
* StandardScaler
* ColumnTransformer
* Pipeline

### API

* FastAPI
* Uvicorn
* Pydantic
* Swagger/OpenAPI

### Model Saving

* Joblib

### Visualization

* Matplotlib
* Jupyter Notebook

### Development

* VS Code
* Jupyter Notebook
* Git
* GitHub
* GitHub Actions

### Testing

* Pytest
* Ruff

---

# 6. Project Structure

```text
CustomerIQ/
│
├── Data/
│   ├── Processed/
│   │   ├── classification_results.csv
│   │   ├── customerIQ_customer_churn_cleaned.csv
│   │   ├── customerIQ_customer_churn_features.csv
│   │   ├── final_model_summary.csv
│   │   └── regression_results.csv
│   │
│   └── Raw/
│       ├── customerIQ_customer_churn_dataset.csv
│       └── customerIQ_customer_churn_dataset.xlsx
│
├── Notebooks/
│   ├── 01_data_understanding.ipynb
│   ├── 01_etl_cleaning.ipynb
│   ├── business_insights.ipynb
│   ├── classification.ipynb
│   ├── clustering.ipynb
│   ├── dashboard.ipynb
│   ├── fastapi.ipynb
│   ├── feature_engineering.ipynb
│   ├── model_evaluation.ipynb
│   └── regression.ipynb
│
├── app/
│   └── main.py
│
├── models/
│   ├── churn_model.joblib
│   ├── cluster_scaler.joblib
│   ├── clv_model.joblib
│   └── kmeans_model.joblib
│
├── main.py
└── README.md
```

---

# 7. FastAPI

After completing the machine learning work, I connected the models to a FastAPI application.

The API provides endpoints for uploading data, making predictions, segmentation, viewing metrics, and training models.

### Available Endpoints

| Method | Endpoint         | Purpose                          |
| ------ | ---------------- | -------------------------------- |
| GET    | `/`              | Check whether the API is running |
| POST   | `/upload`        | Upload customer CSV data         |
| POST   | `/predict/churn` | Predict customer churn           |
| POST   | `/predict/clv`   | Predict Customer Lifetime Value  |
| POST   | `/segment`       | Assign a customer to a cluster   |
| GET    | `/metrics`       | Get model evaluation results     |
| POST   | `/train`         | Train the models                 |

---

# 8. Running the API

From the project folder:

```bash
python -m uvicorn app.main:app --reload
```

The API runs locally and can be tested through Swagger.

### Swagger

Open:

```text
http://127.0.0.1:8000/docs
```

Swagger provides an interactive page where each endpoint can be tested without writing separate API requests.

---

# 9. API Testing

I tested the main endpoints through Swagger.

### API Health Check

```json
{
  "message": "CustomerIQ API is running"
}
```

### CLV Prediction

A sample request returned:

```json
{
  "predicted_clv": 63320.4477
}
```

This means the model estimated the customer's lifetime value at approximately **63,320.45** in the dataset's monetary units.

### Customer Segmentation

A sample request returned:

```json
{
  "cluster": 1
}
```

The API successfully assigned the customer to Cluster 1.

The cluster returned for an individual customer should not be confused with the overall cluster analysis. For example, Cluster 2 is still the highest-value segment across the complete dataset.

### Metrics

The `/metrics` endpoint returns the classification and regression model evaluation results.

The regression results confirm that Linear Regression currently performs best for CLV prediction.

### Model Training

The `/train` endpoint successfully retrains the churn and CLV models and saves the updated model files.

---

# 10. Saved Models

The trained models are stored using Joblib:

```text
models/
├── churn_model.joblib
├── clv_model.joblib
├── cluster_scaler.joblib
└── kmeans_model.joblib
```

The FastAPI application loads these models when predictions are requested.

---

# 11. Dashboard

I created a dashboard notebook to bring the model results and customer segments together.

The dashboard includes:

* Classification model comparison
* Regression model comparison
* Customer count by cluster
* Average CLV by cluster
* Churn rate by cluster
* Customer segment profiles
* Business observations

The dashboard makes the model results easier to understand without looking directly at the raw evaluation tables.

---

# 12. Business Insights

### Highest-Value Customers

Cluster 2 has the highest average CLV at approximately **144,961.55**.

This makes it the most valuable customer group in the current segmentation.

### Highest-Risk Customers

Cluster 3 has the highest churn rate at approximately **5.44%**.

This group also has lower product usage and login frequency, which makes it an important group to investigate from a retention perspective.

### Churn Model

The classification results showed an important problem with the dataset: the churn classes are highly imbalanced.

Because of this, a model with very high accuracy can still perform badly when identifying actual churn customers.

For the current results, Logistic Regression is more useful than simply selecting the model with the highest accuracy.

### CLV Model

Linear Regression produced the lowest MAE and RMSE and the highest R² score among the tested regression models.

---

# 13. Testing

I added automated tests for the FastAPI application using Pytest.

The project also uses Ruff for code quality checks.

Local testing completed successfully:

```text
1 passed
```

Ruff also completed successfully with no linting errors.

---

# 14. GitHub Actions

The project includes a GitHub Actions workflow that runs automatically when changes are pushed to the `main` branch.

The CI process:

```text
Git Push
   ↓
GitHub Actions
   ↓
Install Dependencies
   ↓
Ruff
   ↓
Pytest
   ↓
PASS / FAIL
```

This makes sure that changes to the project are checked automatically instead of relying only on manual testing.

---

# 15. How to Run the Project

### Clone the repository

```bash
git clone <https://github.com/AkshayaCheerneni/CustomerIQ.git>
```

### Open the project

```bash
cd CustomerIQ
```

### Install dependencies

```bash
pip install pandas numpy scikit-learn matplotlib fastapi uvicorn pydantic joblib jupyter pytest ruff httpx python-multipart
```

### Run the notebooks

The notebooks can be opened using Jupyter Notebook or VS Code.

A typical workflow is:

```text
Data Understanding
       ↓
ETL / Cleaning
       ↓
Feature Engineering
       ↓
Classification
       ↓
Regression
       ↓
Clustering
       ↓
Business Insights
       ↓
FastAPI
       ↓
Dashboard
```

### Start FastAPI

```bash
python -m uvicorn app.main:app --reload
```

Then open:

```text
http://127.0.0.1:8000/docs
```

---

# 16. Limitations

There are a few limitations in the current version of the project:

* The churn dataset is imbalanced.
* Accuracy alone is not a good measure for the churn model.
* The clustering results depend on the selected features and number of clusters.
* The API is currently intended for local development.
* The dashboard is notebook-based.
* Model performance may change when the models are used with new data.

---

# Project Summary

CustomerIQ started as a customer dataset and was developed into a complete machine learning application.

The project covers:

```text
Data Analysis
      ↓
Data Cleaning
      ↓
Feature Engineering
      ↓
Machine Learning
      ↓
Model Evaluation
      ↓
Customer Segmentation
      ↓
FastAPI
      ↓
Dashboard
      ↓
Testing
      ↓
GitHub Actions
```

The final project demonstrates practical experience with **data analysis, machine learning, customer segmentation, API development, testing, and basic CI/CD**.

---

## Author

**Akshaya Cheerneni**

Machine Learning + Data Analysis Project

---

## License

This project is intended for educational and portfolio purposes.
