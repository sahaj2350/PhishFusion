# =====================================================
# IMPORTS
# =====================================================
import os
import pandas as pd
import numpy as np
import joblib

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

TEST_FILE = os.path.join(BASE, "archive", "CEAS_08.csv")
MODEL_DIR = "saved_models"

# =====================================================
# LOAD MODELS
# =====================================================
rf = joblib.load(os.path.join(MODEL_DIR, "rf_metadata.pkl"))
xgb = joblib.load(os.path.join(MODEL_DIR, "xgb_metadata.pkl"))
encoders = joblib.load(os.path.join(MODEL_DIR, "feature_encoders.pkl"))

label_encoder = None
if os.path.exists(os.path.join(MODEL_DIR, "label_encoder.pkl")):
    label_encoder = joblib.load(os.path.join(MODEL_DIR, "label_encoder.pkl"))

print("[INFO] Models loaded")

# =====================================================
# LOAD DATA
# =====================================================
df = pd.read_csv(TEST_FILE, low_memory=False)
print("[INFO] Test data shape:", df.shape)

# =====================================================
# CLEANING
# =====================================================
df.replace([np.inf, -np.inf], np.nan, inplace=True)

num_cols = df.select_dtypes(include=["int64", "float64"]).columns
cat_cols = df.select_dtypes(include=["object", "string"]).columns

df[num_cols] = df[num_cols].fillna(0)
df[cat_cols] = df[cat_cols].fillna("missing")

# =====================================================
# LABEL
# =====================================================
target_col = [c for c in df.columns if c.lower() in ["label","class","result"]][0]

y = df[target_col]
X = df.drop(columns=[target_col])

# safe label handling
if pd.api.types.is_numeric_dtype(y):
    y = y.astype(int)
else:
    y = y.astype(str)
    if label_encoder:
        known = set(label_encoder.classes_)
        y = y.map(lambda x: x if x in known else "unknown")

        if "unknown" not in label_encoder.classes_:
            label_encoder.classes_ = np.append(label_encoder.classes_, "unknown")

        y = label_encoder.transform(y)

# =====================================================
# ENCODING
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
# 🔥 PROPER FEATURE ALIGNMENT (FIXED)
# =====================================================
train_features = rf.feature_names_in_

# reindex → fastest + correct
X = X.reindex(columns=train_features, fill_value=0)

# ensure numeric
X = X.apply(pd.to_numeric, errors="coerce").fillna(0)

print("[INFO] Feature alignment done")

# =====================================================
# ENSEMBLE
# =====================================================
w_rf, w_xgb = 0.55, 0.45

rf_prob = rf.predict_proba(X)[:, 1]
xgb_prob = xgb.predict_proba(X)[:, 1]

final_prob = w_rf * rf_prob + w_xgb * xgb_prob

# ⚠️ adjust threshold (important for this dataset)
y_pred = (final_prob > 0.3).astype(int)

# =====================================================
# METRICS
# =====================================================
print("\n===== TEST RESULTS =====")
print("Accuracy :", accuracy_score(y, y_pred))
print("Precision:", precision_score(y, y_pred, zero_division=0))
print("Recall   :", recall_score(y, y_pred, zero_division=0))
print("F1 Score :", f1_score(y, y_pred, zero_division=0))

print("\nConfusion Matrix:")
print(confusion_matrix(y, y_pred))

print("\nClassification Report:")
print(classification_report(y, y_pred, zero_division=0))

# =====================================================
# SAVE (CSV ONLY — SAFE)
# =====================================================
OUTPUT_DIR = os.path.join(BASE, "results")
os.makedirs(OUTPUT_DIR, exist_ok=True)

df_out = pd.DataFrame({
    "actual": y,
    "predicted": y_pred,
    "probability": final_prob
})

csv_path = os.path.join(OUTPUT_DIR, "metadata_predictions.csv")
df_out.to_csv(csv_path, index=False)

print(f"\n[INFO] Results saved → {csv_path}")