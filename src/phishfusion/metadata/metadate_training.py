# =====================================================
# IMPORTS
# =====================================================
import os
import pandas as pd
import numpy as np
import joblib

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix
from xgboost import XGBClassifier

# =====================================================
# DEVICE SETUP (GPU / CPU)
# =====================================================
import torch

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
print(f"[INFO] Using device: {DEVICE}")

# =====================================================
# PATHS
# =====================================================
BASE = r"C:\Users\sketc\Downloads\4th Year Research\Multimodal\Metadata"

TRAIN_FILES = [
    os.path.join(BASE, "Phishing-Dataset-master", "dataset_full.csv"),
    os.path.join(BASE, "Phishing-Dataset-master", "dataset_small.csv"),
    os.path.join(BASE, "phiusiil+phishing+url+dataset", "PhiUSIIL_Phishing_URL_Dataset.csv"),
    os.path.join(BASE, "archive", "phishing_email.csv"),
    os.path.join(BASE, "archive", "Nigerian_Fraud.csv"),
    os.path.join(BASE, "archive", "SpamAssasin.csv"),
    os.path.join(BASE, "archive", "Ling.csv"),
    os.path.join(BASE, "archive", "Nazario.csv"),
    os.path.join(BASE, "archive", "Enron.csv")
]

# =====================================================
# LOAD DATA
# =====================================================
def load_and_merge(files):
    dfs = []
    for f in files:
        try:
            df = pd.read_csv(f, low_memory=False)
            dfs.append(df)
            print(f"[LOADED] {os.path.basename(f)} → {df.shape}")
        except Exception as e:
            print(f"[SKIPPED] {f} → {e}")
    return pd.concat(dfs, ignore_index=True)

df = load_and_merge(TRAIN_FILES)
print("\nMerged dataset shape:", df.shape)

# =====================================================
# CLEANING
# =====================================================
df = df.drop_duplicates()

df.replace([np.inf, -np.inf], np.nan, inplace=True)

# Fill numeric and categorical separately
num_cols = df.select_dtypes(include=["int64", "float64"]).columns
cat_cols = df.select_dtypes(include=["object", "string"]).columns

df[num_cols] = df[num_cols].fillna(0)
df[cat_cols] = df[cat_cols].fillna("missing")

# =====================================================
# LABEL DETECTION
# =====================================================
possible_labels = ["label", "Label", "class", "Class", "result", "Result"]

target_col = None
for col in possible_labels:
    if col in df.columns:
        target_col = col
        break

if target_col is None:
    raise ValueError("❌ No label column found")

print(f"[INFO] Using label column: {target_col}")

y = df[target_col]
X = df.drop(columns=[target_col])

# =====================================================
# LABEL ENCODING (FORCE NUMERIC)
# =====================================================
y = y.astype(str)
y_encoder = LabelEncoder()
y = y_encoder.fit_transform(y)

os.makedirs("saved_models", exist_ok=True)
joblib.dump(y_encoder, "saved_models/label_encoder.pkl")

# =====================================================
# FEATURE ENCODING (ROBUST FIX)
# =====================================================
encoders = {}

for col in X.columns:
    if not pd.api.types.is_numeric_dtype(X[col]):
        le = LabelEncoder()
        X[col] = le.fit_transform(X[col].astype(str))
        encoders[col] = le

joblib.dump(encoders, "saved_models/feature_encoders.pkl")

print(f"[INFO] Encoded {len(encoders)} categorical features")

# =====================================================
# FINAL SAFETY CHECK (CRITICAL)
# =====================================================
# Ensure everything is numeric
X = X.apply(pd.to_numeric, errors='coerce')
X = X.fillna(0)

print("[INFO] All features converted to numeric")

# =====================================================
# CLASS IMBALANCE
# =====================================================
y = y.astype(int)  # FIX for bincount warning

class_counts = np.bincount(y)
neg, pos = class_counts[0], class_counts[1]

scale_pos_weight = neg / pos if pos != 0 else 1

print(f"[INFO] Class Distribution → 0:{neg}, 1:{pos}")
print(f"[INFO] XGBoost scale_pos_weight: {scale_pos_weight:.2f}")

# =====================================================
# SPLIT
# =====================================================
X_train, X_val, y_train, y_val = train_test_split(
    X, y,
    test_size=0.2,
    stratify=y,
    random_state=42
)

# =====================================================
# MODELS
# =====================================================
rf = RandomForestClassifier(
    n_estimators=300,
    max_depth=20,
    class_weight="balanced",
    n_jobs=-1,
    random_state=42
)

xgb = XGBClassifier(
    n_estimators=400,
    max_depth=8,
    learning_rate=0.05,
    scale_pos_weight=scale_pos_weight,
    eval_metric="logloss",
    n_jobs=-1,
    random_state=42
)

# =====================================================
# TRAINING
# =====================================================
print("\nTraining Random Forest...")
rf.fit(X_train, y_train)

print("Training XGBoost...")
xgb.fit(X_train, y_train)

print(xgb.get_params())

print("Final X shape:", X.shape)
print("Unique labels:", np.unique(y, return_counts=True))
# =====================================================
# VALIDATION
# =====================================================
print("\n===== VALIDATION RESULTS =====")

rf_pred = rf.predict(X_val)
xgb_pred = xgb.predict(X_val)

print("\n--- Random Forest ---")
print(classification_report(y_val, rf_pred))
print(confusion_matrix(y_val, rf_pred))

print("\n--- XGBoost ---")
print(classification_report(y_val, xgb_pred))
print(confusion_matrix(y_val, xgb_pred))

# =====================================================
# SAVE MODELS
# =====================================================
joblib.dump(rf, "saved_models/rf_metadata.pkl")
joblib.dump(xgb, "saved_models/xgb_metadata.pkl")

print("\n[INFO] Models saved successfully")

# =====================================================
# FEATURE IMPORTANCE
# =====================================================
importances = xgb.feature_importances_

feature_importance_df = pd.DataFrame({
    "feature": X.columns,
    "importance": importances
}).sort_values(by="importance", ascending=False)

feature_importance_df.to_csv("saved_models/feature_importance.csv", index=False)

print("\n[INFO] Feature importance saved")