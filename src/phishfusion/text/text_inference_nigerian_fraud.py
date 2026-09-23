# phishfusion_inference_nigerian_fraud.py
import os
import re
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from typing import List
from transformers import BertTokenizer, BertModel, DistilBertTokenizer, DistilBertModel
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report, confusion_matrix

# ---------------------------
# CONFIG
# ---------------------------
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
BERT_NAME = "bert-base-uncased"
DISTIL_NAME = "distilbert-base-uncased"
MAX_LEN = 128
BATCH_SIZE = 8

WEIGHT_BERT = 0.45
WEIGHT_DISTIL = 0.45
WEIGHT_META = 0.10

BERT_CKPT = "bert_best.pt"
DISTIL_CKPT = "distil_best.pt"
META_CKPT = "meta_best.pt"

SCALER_MEAN_FILE = "scaler_mean.npy"
SCALER_SCALE_FILE = "scaler_scale.npy"

# ---------------------------
# TEXT CLEANING
# ---------------------------
def clean_text(text: str) -> str:
    if not isinstance(text, str):
        return ""
    text = text.lower()
    text = re.sub(r"http\S+|www\.\S+", " ", text)
    text = re.sub(r"\b[\w\.-]+@[\w\.-]+\.\w+\b", " ", text)
    text = re.sub(r"[^a-z0-9\s]", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text

url_re = re.compile(r"(https?://[^\s]+|www\.[^\s]+)")

# ---------------------------
# METADATA FEATURES
# ---------------------------
def extract_metadata_features(texts: List[str]) -> np.ndarray:
    features = []
    for t in texts:
        ln = len(t)
        digits = sum(c.isdigit() for c in t)
        urls = len(url_re.findall(t))
        punct = sum(1 for c in t if c in "!?.,;:")
        uppercase_ratio = sum(1 for c in t if c.isupper()) / (ln + 1)
        domain_tokens = sum(tok in t for tok in [".com", ".net", ".org", ".ru", ".cn", ".info"])
        features.append([ln, digits, urls, punct, uppercase_ratio, domain_tokens])
    return np.array(features, dtype=np.float32)

# ---------------------------
# MODELS
# ---------------------------
class BertCNNLSTM(nn.Module):
    def __init__(self):
        super().__init__()
        self.bert = BertModel.from_pretrained(BERT_NAME)
        self.conv = nn.Conv1d(768, 128, kernel_size=3, padding=1)
        self.lstm = nn.LSTM(128, 128, batch_first=True)
        self.fc = nn.Linear(128, 2)
        self.dropout = nn.Dropout(0.3)

    def forward(self, input_ids, attention_mask):
        out = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        x = out.last_hidden_state.permute(0, 2, 1)
        x = torch.relu(self.conv(x)).permute(0, 2, 1)
        h, _ = self.lstm(x)
        return self.fc(self.dropout(h[:, -1]))

class DistilCNNBiLSTM(nn.Module):
    def __init__(self):
        super().__init__()
        self.distil = DistilBertModel.from_pretrained(DISTIL_NAME)
        self.conv = nn.Conv1d(768, 128, kernel_size=3, padding=1)
        self.lstm = nn.LSTM(128, 128, batch_first=True, bidirectional=True)
        self.fc = nn.Linear(256, 2)
        self.dropout = nn.Dropout(0.3)

    def forward(self, input_ids, attention_mask):
        out = self.distil(input_ids=input_ids, attention_mask=attention_mask)
        x = out.last_hidden_state.permute(0, 2, 1)
        x = torch.relu(self.conv(x)).permute(0, 2, 1)
        h, _ = self.lstm(x)
        return self.fc(self.dropout(h[:, -1]))

class MetadataMLP(nn.Module):
    def __init__(self, input_dim):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, 64),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(64, 2)
        )

    def forward(self, x):
        return self.net(x)

# ---------------------------
# SAFE LOAD
# ---------------------------
def safe_load(path):
    try:
        return torch.load(path, map_location=DEVICE, weights_only=True)
    except TypeError:
        return torch.load(path, map_location=DEVICE)

# ---------------------------
# BATCH PREDICTION
# ---------------------------
def predict_text(model, tokenizer, texts):
    model.eval()
    probs = []
    with torch.no_grad():
        for i in range(0, len(texts), BATCH_SIZE):
            enc = tokenizer(
                texts[i:i+BATCH_SIZE],
                padding="max_length",
                truncation=True,
                max_length=MAX_LEN,
                return_tensors="pt"
            ).to(DEVICE)
            logits = model(enc["input_ids"], enc["attention_mask"])
            probs.append(torch.softmax(logits, dim=1).cpu().numpy())
    return np.vstack(probs)

def predict_meta(model, feats):
    model.eval()
    probs = []
    with torch.no_grad():
        for i in range(0, len(feats), BATCH_SIZE):
            x = torch.tensor(feats[i:i+BATCH_SIZE], dtype=torch.float32).to(DEVICE)
            probs.append(torch.softmax(model(x), dim=1).cpu().numpy())
    return np.vstack(probs)

# ---------------------------
# MAIN
# ---------------------------
def main():
    if not os.path.exists("Nigerian_Fraud.csv"):
        raise SystemExit("Nigerian_Fraud.csv not found")

    df = pd.read_csv("Nigerian_Fraud.csv")

    # ✅ CRITICAL FIX: cast EVERYTHING to string
    df["subject"] = df["subject"].astype(str)
    df["body"] = df["body"].astype(str)
    df["urls"] = df["urls"].astype(str)

    df["full_text"] = (
        df["subject"].fillna("") + " " +
        df["body"].fillna("") + " " +
        df["urls"].fillna("")
    )

    texts = [clean_text(t) for t in df["full_text"]]

    # Labels
    raw_labels = df["label"].astype(str).str.lower()
    labels = [
        1 if any(x in lbl for x in ["fraud", "scam", "phish", "1"]) else 0
        for lbl in raw_labels
    ]

    # Metadata
    meta_feats = extract_metadata_features(texts)
    scaler = StandardScaler()

    if os.path.exists(SCALER_MEAN_FILE) and os.path.exists(SCALER_SCALE_FILE):
        scaler.mean_ = np.load(SCALER_MEAN_FILE)
        scaler.scale_ = np.load(SCALER_SCALE_FILE)
        scaler.var_ = scaler.scale_ ** 2
        scaler.n_features_in_ = meta_feats.shape[1]
        meta_feats = scaler.transform(meta_feats)
    else:
        meta_feats = scaler.fit_transform(meta_feats)

    # Models
    bert_tok = BertTokenizer.from_pretrained(BERT_NAME)
    distil_tok = DistilBertTokenizer.from_pretrained(DISTIL_NAME)

    m1 = BertCNNLSTM().to(DEVICE)
    m2 = DistilCNNBiLSTM().to(DEVICE)
    m3 = MetadataMLP(meta_feats.shape[1]).to(DEVICE)

    m1.load_state_dict(safe_load(BERT_CKPT))
    m2.load_state_dict(safe_load(DISTIL_CKPT))
    m3.load_state_dict(safe_load(META_CKPT))

    # Predictions
    p1 = predict_text(m1, bert_tok, texts)
    p2 = predict_text(m2, distil_tok, texts)
    p3 = predict_meta(m3, meta_feats)

    ensemble = WEIGHT_BERT*p1 + WEIGHT_DISTIL*p2 + WEIGHT_META*p3
    preds = np.argmax(ensemble, axis=1)

    print(classification_report(labels, preds, target_names=["ham", "phish"]))
    print(confusion_matrix(labels, preds))

    df["pred_label"] = np.where(preds == 1, "phish", "ham")
    df["phish_confidence"] = ensemble[:, 1]
    df.to_csv("Nigerian_Fraud_predictions.csv", index=False)

    print("✅ Saved → Nigerian_Fraud_predictions.csv")

if __name__ == "__main__":
    main()
