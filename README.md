# 🛒 Product Grouping & Next Month Revenue Prediction 📈

This project provides an end-to-end pipeline for grouping products based on sales/return patterns and predicting next month's revenue for each product using machine learning. It includes data cleaning, feature engineering, clustering (KMeans, DBSCAN), regression modeling (Random Forest, XGBoost), and a Flask API for easy integration.

## ✨ Features

- **Product Grouping:** Unsupervised clustering of products using KMeans and DBSCAN.
- **Revenue Prediction:** Predicts next month's revenue for each product using advanced regression models.
- **API Deployment:** Flask API for both clustering and regression predictions.
- **Modular Codebase:** Clean separation of data processing, modeling, and API logic.
- **Documentation:** Well-documented workflow and business interpretation.

## 📁 Project Structure

```
.
├── app.py                  # Flask API for clustering & regression
├── src/                    # Source code (processing, utils, business logic)
├── models/                 # Saved models and scalers
├── data/                   # Raw and processed data
├── regression_results/     # Model results and plots
├── docs/                   # Documentation and notes
├── requirements.txt        # Python dependencies
├── README.md               # Project overview and usage
├── .gitignore              # Files/folders to ignore in git
└── LICENSE                 # Project license
```

## 🚀 Getting Started

1. **Clone the repo:**
   ```bash
   git clone https://github.com/Zahraaxikmah123/prodgroup-revenuepred-ml.git
   cd prodgroup-revenuepred-ml
   ```

2. **Create and activate a virtual environment:**
   ```bash
   python3 -m venv venv
   source venv/bin/activate
   ```

3. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

4. **Run the API:**
   ```bash
   python app.py
   ```

5. **Use the API endpoints** for clustering and revenue prediction.

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

---