# 🛒 Product Grouping & Next Month Revenue Prediction 📈

This project provides an end-to-end pipeline for grouping products based on sales/return patterns and predicting next month's revenue for each product using machine learning. It includes data cleaning, feature engineering, clustering (KMeans, DBSCAN), regression modeling (Random Forest, XGBoost), and a Flask API for easy integration.

## ✨ Features

- **Product Grouping:** Unsupervised clustering of products using KMeans and DBSCAN.
- **Revenue Prediction:** Predicts next month's revenue for each product using advanced regression models.
- **API Deployment:** Flask API for both clustering and regression predictions.
- **Modular Codebase:** Clean separation of data processing, modeling, and API logic.
- **Documentation:** Workflow and business interpretation included.

## 📁 Repo structure (as in your workspace)

```
.
├── clustering_results/        # elbow + PCA cluster plots (png)
├── dataset/
│   ├── processed/             # aggregated & product-level CSVs
│   └── raw/                   # raw CSVs and cleaned CSV
├── models/                    # saved scalers & model .joblib/.pkl files
├── regression_results/        # model metrics + plots
├── templates/                 # forms.html (Flask UI)
├── venv/                      # optional virtual environment
├── .gitignore                 # Files/folders to ignore in git
├── app.py                     # Flask API + UI route (/ui)
├── business_interpretation.py # cluster -> business text mapping
├── clustering.py              # clustering pipeline (elbow, KMeans, DBSCAN)
├── forms.py                   # Flask-WTF form for UI
├── LICENSE                    # Project license
├── prepare_regression_data.py # aggregation & lag features script
├── processing.py              # cleaning & feature engineering script
├── README.md                  # this file
├── regression.py              # regression training & evaluation
├── requirements.txt           # Python dependencies
└── utils.py                   # helpers used by app and tests
```

## 🚀 Quick start (4 steps)

1. **Clone the repo:**
   ```bash
   git clone https://github.com/Zahraaxikmah123/prodgroup-revenuepred-ml.git
   cd prodgroup-revenuepred-ml
   ```
2. **Create & activate venv:**
   ```bash
   python3 -m venv venv
   source venv/bin/activate
   ```
3. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   ```
4. **Run the API and open UI:**
   ```bash
   python app.py
   # Open http://localhost:8000/ui
   ```

## Key endpoints
- GET /                → API index (lists expected payloads)
- GET /ui              → Web form (manual input + predict)
- POST /predict_group  → clustering (kmeans or dbscan)
- POST /predict_revenue→ revenue prediction (random_forest or xgboost)
- POST /predict_all    → both clustering + revenue

## Examples

- POST /predict_group?model=kmeans
  - Request JSON:
    ```json
    {
      "NetRevenue": 12345.67,
      "NetQuantity": 200,
      "NumTransactions": 150,
      "NumUniqueCustomers": 120
    }
    ```
  - Example response (KMeans):
    ```json
    {
      "nooca_modelka": "KMeans",
      "xogta_badeecada": { ... },
      "kooxda_nambarkeeda": 1,
      "kooxda": "High value, high frequency",
      "sharaxaad": "Products with consistent high revenue and frequent purchases"
    }
    ```

- POST /predict_revenue?model=xgboost
  - Request JSON:
    ```json
    {
      "NetRevenue": 12345.67,
      "NetRevenue_LastMonth": 11000.00,
      "NetRevenue_MA3": 10500.00,
      "Month": 9,
      "ProductFrequency": 20
    }
    ```
  - Example response:
    ```json
    {
      "nooca_modelka": "xgboost",
      "xogta_badeecada": { ... },
      "next_month_revenue_prediction": "$12,987.45"
    }
    ```

- POST /predict_all?group_model=kmeans&rev_model=xgboost
  - Body: combine required fields from both endpoints (see `/predict_all` docstring in `app.py`).

## Where results are
- models/                → load these for inference
- clustering_results/    → visual cluster plots
- regression_results/    → metrics & actual vs predicted plots
- dataset/processed/     → product-level, aggregated datasets

## Troubleshooting
- If you get errors about missing model files, ensure `models/` contains the following filenames used by `app.py`:
- `kmeans_model.joblib`, `dbscan_model.joblib`, `random_forest_regressor.joblib`, `xgboost_regressor.joblib`, `clustering_scaler.pkl`, `regression_scaler.pkl`.
- Do not use the hard-coded `app.secret_key` in production. Replace with an environment-provided secret.

## Notes
- Ensure model files in `models/` match names used in `app.py` and `utils.py`.
- For local UI fetch requests during development you can disable CSRF (not recommended for production):
  ```python
  app.config['WTF_CSRF_ENABLED'] = False
  ```

That's it — simple structure, quick run, and UI at /ui for manual testing. ✅

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

---