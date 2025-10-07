import pandas as pd
import numpy as np
import os
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from xgboost import XGBRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.preprocessing import StandardScaler
import joblib

# ==================================================
# 1. Load Aggregated Dataset
# ==================================================
agg = pd.read_csv('dataset/processed/product_revenue_dataset.csv')

features = [
    'NetRevenue', 'NetRevenue_LastMonth', 'NetRevenue_MA3', 'Month', 'ProductFrequency'
]
X = agg[features]
y = agg['NextMonthRevenue']

# ==================================================
# 1a. Train-Test Split
# ==================================================
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# ==================================================
# 2. Feature Scaling (StandardScaler)
# ==================================================
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# ==================================================
# 3. Train Models
# ==================================================
# Linear Regression
lr = LinearRegression()
lr.fit(X_train_scaled, y_train)
lr_pred = lr.predict(X_test_scaled)

# Random Forest
rf_params = {
    'n_estimators': [100, 200],
    'max_depth': [10, 20, None],
    'min_samples_split': [2, 5],
}
rf_grid = GridSearchCV(RandomForestRegressor(random_state=42), rf_params, cv=3, scoring='r2', n_jobs=-1)
rf_grid.fit(X_train_scaled, y_train)
rf_best = rf_grid.best_estimator_
rf_pred = rf_best.predict(X_test_scaled)

# XGBoost
xgb_params = {
    'n_estimators': [100, 200],
    'learning_rate': [0.05, 0.1],
    'max_depth': [3, 5, 7],
}
xgb_grid = GridSearchCV(XGBRegressor(objective='reg:squarederror', random_state=42), xgb_params, cv=3, scoring='r2', n_jobs=-1)
xgb_grid.fit(X_train_scaled, y_train)
xgb_best = xgb_grid.best_estimator_
xgb_pred = xgb_best.predict(X_test_scaled)

# save models and scaler
joblib.dump(rf_best, "models/random_forest_regressor.joblib")
joblib.dump(xgb_best, "models/xgboost_regressor.joblib")
joblib.dump(scaler, "models/regression_scaler.pkl")
print("✅ Models and scaler saved in 'models/' folder.")

# ==================================================
# 4. Evaluate Models
# ==================================================
def evaluate_model(name, y_true, y_pred):
    r2 = r2_score(y_true, y_pred)
    mae = mean_absolute_error(y_true, y_pred)
    mse = mean_squared_error(y_true, y_pred)
    rmse = np.sqrt(mse)
    return {'Model': name, 'R2': r2, 'MAE': mae, 'MSE': mse, 'RMSE': rmse}

results = []
results.append(evaluate_model('Linear Regression', y_test, lr_pred))
results.append(evaluate_model('Random Forest', y_test, rf_pred))
results.append(evaluate_model('XGBoost', y_test, xgb_pred))

results_df = pd.DataFrame(results)
results_df.to_csv("regression_results/model_performance.csv", index=False)
print("\nModel Performance Summary:")
print(results_df)

# ==================================================
# 5. Visualizations
# ==================================================
plt.figure(figsize=(8,6))
sns.barplot(data=results_df, x='Model', y='R2')
plt.title("Model Comparison (R² Score)")
plt.savefig("regression_results/model_r2_comparison.png")
plt.close()

feat_importances = pd.Series(rf_best.feature_importances_, index=features)
feat_importances.nlargest(10).plot(kind='barh', title='Feature Importance (Random Forest)')
plt.tight_layout()
plt.savefig("regression_results/feature_importance_rf.png")
plt.close()

for name, preds in [
    ("Linear Regression", lr_pred),
    ("Random Forest", rf_pred),
    ("XGBoost", xgb_pred)
]:
    plt.figure(figsize=(6,6))
    sns.scatterplot(x=y_test, y=preds, alpha=0.5)
    plt.xlabel("Actual Revenue")
    plt.ylabel("Predicted Revenue")
    plt.title(f"Actual vs Predicted ({name})")
    plt.tight_layout()
    plt.savefig(f"regression_results/actual_vs_predicted_{name.replace(' ','_')}.png")
    plt.close()

# ==================================================
# 6. Sanity Check - Single Prediction Example
# ==================================================
i = 9
x_one_df = X_test_scaled[[i]]
y_true = y_test.iloc[i]
print(f"\nSingle-row prediction sanity check:")
print(f"  Actual: {y_true:,.0f}")
print(f"  LR     : {lr.predict(x_one_df)[0]:.0f}")
print(f"  RF     : {rf_best.predict(x_one_df)[0]:.0f}")
print(f"  XGB    : {xgb_best.predict(x_one_df)[0]:.0f}")

# print("\n✅ All results (plots + metrics) saved in 'regression_results/' folder.")