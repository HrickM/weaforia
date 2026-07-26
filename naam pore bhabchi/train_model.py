"""
================================================================================
 CLIMATE VISIBILITY PREDICTION — MODEL TRAINING PIPELINE
================================================================================
Run this in VS Code (or any local Python environment) to train, evaluate and
export the visibility prediction model used by the backend API.

Folder expectations:
    training/
        train_model.py   <- this file
        cleaned_data.csv <- your dataset (already provided)
    model/                <- created automatically, holds the exported model

Usage:
    python train_model.py

Outputs (written to ../model/):
    visibility_model.joblib   -> trained model + scaler + feature list (single bundle)
    metrics.json               -> evaluation metrics for every model tried
    eda_correlation_heatmap.png
    eda_feature_distributions.png
    eda_visibility_vs_humidity.png
    feature_importance.png
================================================================================
"""

import json
import os
import warnings

import joblib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

try:
    from xgboost import XGBRegressor
    HAS_XGBOOST = True
except ImportError:
    HAS_XGBOOST = False
    print("[warn] xgboost not installed — run `pip install xgboost` to include it. "
          "Continuing with Linear Regression + Random Forest only.")

warnings.filterwarnings("ignore")
sns.set_theme(style="darkgrid")

# --------------------------------------------------------------------------
# 0. PATHS
# --------------------------------------------------------------------------
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_PATH = os.path.join(BASE_DIR, "cleaned_data.csv")
MODEL_DIR = os.path.join(BASE_DIR, "model")
os.makedirs(MODEL_DIR, exist_ok=True)


# --------------------------------------------------------------------------
# 1. LOAD DATA
# --------------------------------------------------------------------------
print("Loading dataset...")
df = pd.read_csv(DATA_PATH, parse_dates=["DATE"])
print(f"Shape: {df.shape}")
print(df.head())
print(df.describe())
print("Nulls per column:\n", df.isnull().sum())


# --------------------------------------------------------------------------
# 2. FEATURE ENGINEERING
# --------------------------------------------------------------------------
# Domain-informed derived features (also referenced in the project's slides:
# dew point spread / fog likelihood / air density).
print("\nEngineering features...")

df["Month"] = df["DATE"].dt.month
df["Hour"] = df["DATE"].dt.hour

# Dew point spread: the smaller the gap between air temp and dew point,
# the closer the air is to saturation -> higher fog / low-visibility risk.
df["DewPointSpread"] = df["DRYBULBTEMPF"] - df["DewPointTempF"]

# A simple fog-likelihood proxy: high humidity + tiny dew point spread.
df["FogLikelihood"] = (df["RelativeHumidity"] / 100) * (1 / (df["DewPointSpread"].abs() + 1))

# Approximate air density (kg/m^3) from pressure and temperature
# (ideal gas law, pressure in inHg -> Pa, temp in F -> K).
pressure_pa = df["StationPressure"] * 3386.39
temp_k = (df["DRYBULBTEMPF"] - 32) * 5 / 9 + 273.15
df["AirDensity"] = pressure_pa / (287.05 * temp_k)

# Cyclical encodings for month/hour so the model understands seasonality
# without treating December (12) and January (1) as far apart.
df["Month_sin"] = np.sin(2 * np.pi * df["Month"] / 12)
df["Month_cos"] = np.cos(2 * np.pi * df["Month"] / 12)
df["Hour_sin"] = np.sin(2 * np.pi * df["Hour"] / 24)
df["Hour_cos"] = np.cos(2 * np.pi * df["Hour"] / 24)

FEATURES = [
    "DRYBULBTEMPF", "WETBULBTEMPF", "DewPointTempF", "RelativeHumidity",
    "WindSpeed", "WindDirection", "StationPressure", "SeaLevelPressure",
    "Precip", "DewPointSpread", "FogLikelihood", "AirDensity",
    "Month_sin", "Month_cos", "Hour_sin", "Hour_cos",
]
TARGET = "VISIBILITY"

df = df.replace([np.inf, -np.inf], np.nan).dropna(subset=FEATURES + [TARGET])
print(f"Shape after feature engineering & cleanup: {df.shape}")


# --------------------------------------------------------------------------
# 3. EXPLORATORY DATA ANALYSIS (saved as PNGs — open them after running)
# --------------------------------------------------------------------------
print("\nGenerating EDA plots...")

# 3a. Correlation heatmap
plt.figure(figsize=(11, 9))
corr = df[FEATURES + [TARGET]].corr()
sns.heatmap(corr, annot=True, fmt=".2f", cmap="coolwarm", center=0, square=True)
plt.title("Feature Correlation Heatmap")
plt.tight_layout()
plt.savefig(os.path.join(MODEL_DIR, "eda_correlation_heatmap.png"), dpi=150)
plt.close()

# 3b. Distribution of key raw features
fig, axes = plt.subplots(2, 3, figsize=(16, 9))
for ax, col in zip(
    axes.flat,
    ["VISIBILITY", "RelativeHumidity", "DRYBULBTEMPF", "WindSpeed", "SeaLevelPressure", "Precip"],
):
    sns.histplot(df[col], kde=True, ax=ax, color="#3A6EA5")
    ax.set_title(col)
plt.tight_layout()
plt.savefig(os.path.join(MODEL_DIR, "eda_feature_distributions.png"), dpi=150)
plt.close()

# 3c. Visibility vs humidity (the strongest single predictor)
plt.figure(figsize=(9, 6))
sample = df.sample(min(5000, len(df)), random_state=42)
sns.scatterplot(data=sample, x="RelativeHumidity", y="VISIBILITY", hue="Precip", palette="viridis", alpha=0.6)
plt.title("Visibility vs Relative Humidity (colored by Precipitation)")
plt.tight_layout()
plt.savefig(os.path.join(MODEL_DIR, "eda_visibility_vs_humidity.png"), dpi=150)
plt.close()

print("EDA plots saved to model/ folder.")


# --------------------------------------------------------------------------
# 4. TRAIN / TEST SPLIT + SCALING
# --------------------------------------------------------------------------
X = df[FEATURES]
y = df[TARGET]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)


# --------------------------------------------------------------------------
# 5. TRAIN MULTIPLE MODELS & COMPARE
# --------------------------------------------------------------------------
def evaluate(model_name, y_true, y_pred):
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    mae = mean_absolute_error(y_true, y_pred)
    r2 = r2_score(y_true, y_pred)
    print(f"  {model_name:<20} RMSE={rmse:.3f}  MAE={mae:.3f}  R2={r2:.3f}")
    return {"rmse": rmse, "mae": mae, "r2": r2}


results = {}
trained_models = {}

print("\nTraining models...")

# Linear Regression (baseline, uses scaled features)
lr = LinearRegression()
lr.fit(X_train_scaled, y_train)
results["LinearRegression"] = evaluate("LinearRegression", y_test, lr.predict(X_test_scaled))
trained_models["LinearRegression"] = lr

# Random Forest (tree-based, robust to feature scale — use raw features)
rf = RandomForestRegressor(
    n_estimators=300, max_depth=18, min_samples_leaf=3, n_jobs=-1, random_state=42
)
rf.fit(X_train, y_train)
results["RandomForest"] = evaluate("RandomForest", y_test, rf.predict(X_test))
trained_models["RandomForest"] = rf

# XGBoost (usually the strongest performer for this kind of tabular data)
if HAS_XGBOOST:
    xgb = XGBRegressor(
        n_estimators=400, max_depth=7, learning_rate=0.05,
        subsample=0.9, colsample_bytree=0.9, random_state=42, n_jobs=-1,
    )
    xgb.fit(X_train, y_train)
    results["XGBoost"] = evaluate("XGBoost", y_test, xgb.predict(X_test))
    trained_models["XGBoost"] = xgb


# --------------------------------------------------------------------------
# 6. SELECT BEST MODEL (lowest RMSE)
# --------------------------------------------------------------------------
best_name = min(results, key=lambda k: results[k]["rmse"])
best_model = trained_models[best_name]
uses_scaler = best_name == "LinearRegression"
print(f"\nBest model: {best_name} (RMSE={results[best_name]['rmse']:.3f}, R2={results[best_name]['r2']:.3f})")


# --------------------------------------------------------------------------
# 7. FEATURE IMPORTANCE (tree models only)
# --------------------------------------------------------------------------
if best_name in ("RandomForest", "XGBoost"):
    importances = pd.Series(best_model.feature_importances_, index=FEATURES).sort_values()
    plt.figure(figsize=(9, 7))
    importances.plot(kind="barh", color="#D97757")
    plt.title(f"Feature Importance — {best_name}")
    plt.tight_layout()
    plt.savefig(os.path.join(MODEL_DIR, "feature_importance.png"), dpi=150)
    plt.close()
    print("Feature importance plot saved.")


# --------------------------------------------------------------------------
# 8. EXPORT MODEL BUNDLE + METRICS
# --------------------------------------------------------------------------
bundle = {
    "model": best_model,
    "model_name": best_name,
    "scaler": scaler,
    "uses_scaler": uses_scaler,
    "features": FEATURES,
    "target": TARGET,
    "metrics": results[best_name],
}
joblib.dump(bundle, os.path.join(MODEL_DIR, "visibility_model.joblib"))

with open(os.path.join(MODEL_DIR, "metrics.json"), "w") as f:
    json.dump(results, f, indent=2)

print(f"\nModel bundle saved to: {os.path.join(MODEL_DIR, 'visibility_model.joblib')}")
print(f"Metrics saved to:      {os.path.join(MODEL_DIR, 'metrics.json')}")
print("\nDone. Point the backend's MODEL_PATH at the .joblib file above.")
