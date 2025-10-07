import numpy as np
import joblib

# Features used for clustering (must match training order)
CLUSTER_FEATURES = ["NetQuantity", "NetRevenue", "NumTransactions", "NumUniqueCustomers"]

# Load the saved scalers for clustering and regression
SCALER = joblib.load("models/clustering_scaler.pkl")
reg_scaler = joblib.load("models/regression_scaler.pkl")

def prepare_features_from_json(record: dict) -> np.ndarray:
    """
    Extract and scale features from user JSON input for clustering models.
    Returns a 2D numpy array ready for model prediction.
    """
    # Extract features in correct order, fill missing with 0.0
    values = [float(record.get(f, 0.0)) for f in CLUSTER_FEATURES]
    X = np.array([values])
    # Scale features
    X_scaled = SCALER.transform(X)
    return X_scaled

def prepare_regression_features_from_json(record: dict) -> np.ndarray:
    """
    Prepare and scale features from user JSON input for regression models.
    Returns a 2D numpy array ready for model prediction.
    """
    features = [
    'NetRevenue', 'NetRevenue_LastMonth', 'NetRevenue_MA3', 'Month', 'ProductFrequency'
]

    values = [float(record.get(f, 0.0)) for f in features]
    arr = np.array([values], dtype=float)
    arr_scaled = reg_scaler.transform(arr)
    return arr_scaled