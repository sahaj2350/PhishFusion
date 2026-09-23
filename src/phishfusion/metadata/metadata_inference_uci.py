# =====================================================
# IMPORTS
# =====================================================
import os
import pandas as pd
import numpy as np
import joblib

from scipy.io import arff

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    classification_report
)

# =====================================================
# PATHS
# =====================================================
BASE = r"C:\Users\sketc\Downloads\4th Year Research\Multimodal\Metadata"

ARFF_FILE = os.path.join(BASE, "phishing+websites", "Training Dataset.arff")

MODEL_DIR = "saved_models"

# =====================================================
# LOAD MODELS
# =====================================================
rf = joblib.load(os.path.join(MODEL_DIR, "rf_metadata.pkl"))
xgb = joblib.load(os.path.join(MODEL_DIR, "xgb_metadata.pkl"))
encoders = joblib.load(os.path.join(MODEL_DIR, "feature_encoders.pkl"))

print("[INFO] Models loaded")

# =====================================================
# LOAD ARFF DATA
# =====================================================
data, meta = arff.loadarff(ARFF_FILE)
df = pd.DataFrame(data)

print("[INFO] Raw dataset shape:", df.shape)

# =====================================================
# FIX BYTE STRINGS
# =====================================================
for col in df.columns:
    if df[col].dtype == object:
        df[col] = df[col].apply(
            lambda x: x.decode("utf-8") if isinstance(x, bytes) else x
        )

# =====================================================
# LABEL HANDLING
# =====================================================
# last column is label
target_col = df.columns[-1]

y = df[target_col]
X = df.drop(columns=[target_col])

# convert labels: -1 → 0, 1 → 1
y = y.astype(int)
y = y.replace(-1, 0)

print("[INFO] Label distribution:", np.bincount(y))

# =====================================================
# ENCODING (if needed)
# =====================================================
for col in encoders:
    if col in X.columns:
        le = encoders[col]
        X[col] = X[col].astype(str)

        X[col] = X[col].map(lambda x: x if x in le.classes_ else "missing")

        if "missing" not in le.classes_:
            le.classes_ = np.append(le.classes_, "missing")

        X[col] = le.transform(X[col])

# =====================================================
# 🔥 FEATURE ALIGNMENT (CRITICAL)
# =====================================================
train_features = rf.feature_names_in_

# Align columns properly
X = X.reindex(columns=train_features, fill_value=0)

# Ensure numeric
X = X.apply(pd.to_numeric, errors="coerce").fillna(0)

print("[INFO] Feature alignment done")
print("[INFO] Final shape:", X.shape)

# =====================================================
# ENSEMBLE PREDICTION
# =====================================================
w_rf = 0.55
w_xgb = 0.45

rf_prob = rf.predict_proba(X)[:, 1]
xgb_prob = xgb.predict_proba(X)[:, 1]

final_prob = w_rf * rf_prob + w_xgb * xgb_prob

# lower threshold for cross-dataset
y_pred = (final_prob > 0.3).astype(int)

# =====================================================
# METRICS
# =====================================================
print("\n===== UCI TEST RESULTS =====")
print("Accuracy :", accuracy_score(y, y_pred))
print("Precision:", precision_score(y, y_pred, zero_division=0))
print("Recall   :", recall_score(y, y_pred, zero_division=0))
print("F1 Score :", f1_score(y, y_pred, zero_division=0))

print("\nConfusion Matrix:")
print(confusion_matrix(y, y_pred))

print("\nClassification Report:")
print(classification_report(y, y_pred, zero_division=0))

# =====================================================
# SAVE RESULTS
# =====================================================
OUTPUT_DIR = os.path.join(BASE, "results")
os.makedirs(OUTPUT_DIR, exist_ok=True)

df_out = pd.DataFrame({
    "actual": y,
    "predicted": y_pred,
    "probability": final_prob
})

csv_path = os.path.join(OUTPUT_DIR, "uci_metadata_predictions.csv")
df_out.to_csv(csv_path, index=False)

print(f"\n[INFO] Results saved → {csv_path}")