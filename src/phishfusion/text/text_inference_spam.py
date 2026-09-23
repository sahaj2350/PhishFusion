# phishfusion_inference.py
import os
import re
import math
import glob
import random
from typing import List
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from transformers import BertTokenizer, BertModel, DistilBertTokenizer, DistilBertModel
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report, confusion_matrix

# ---------------------------
# CONFIG (tweak if needed)
# ---------------------------
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
BERT_NAME = "bert-base-uncased"
DISTIL_NAME = "distilbert-base-uncased"
MAX_LEN = 128
BATCH_SIZE = 8
WEIGHT_BERT = 0.45
WEIGHT_DISTIL = 0.45
WEIGHT_META = 0.10

# Model checkpoint names
BERT_CKPT = "bert_best.pt"
DISTIL_CKPT = "distil_best.pt"
META_CKPT = "meta_best.pt"

# Optional scaler files (if you saved scaler.mean_ and scaler.scale_ during training)
SCALER_MEAN_FILE = "scaler_mean.npy"
SCALER_SCALE_FILE = "scaler_scale.npy"

# ---------------------------
# UTIL: cleaning + metadata extraction (same as training)
# ---------------------------
def clean_text(text: str) -> str:
    if not isinstance(text, str):
        return ""
    t = text.lower()
    t = re.sub(r"http\S+|www\.\S+", " ", t)
    t = re.sub(r"\b[\w\.-]+@[\w\.-]+\.\w+\b", " ", t)
    t = re.sub(r"[^a-z0-9\s]", " ", t)
    t = re.sub(r"\s+", " ", t).strip()
    return t

url_re = re.compile(r"(https?://[^\s]+|www\.[^\s]+)")

def extract_metadata_features(texts: List[str]) -> np.ndarray:
    feats = []
    for t in texts:
        ln = len(t)
        digits = sum(c.isdigit() for c in t)
        urls = len(url_re.findall(t))
        punct = sum(1 for c in t if c in "!?.,;:")
        uppercase_ratio = (sum(1 for c in t if c.isupper()) / (ln + 1))
        domain_tokens = sum(tok in t for tok in [".com", ".net", ".org", ".ru", ".cn", ".info"])
        feats.append([ln, digits, urls, punct, uppercase_ratio, domain_tokens])
    return np.array(feats, dtype=np.float32)

# ---------------------------
# MODEL CLASSES (match training)
# ---------------------------
class BertCNNLSTM(nn.Module):
    def __init__(self, bert_name=BERT_NAME, cnn_filters=128, lstm_hidden=128, num_labels=2, dropout=0.3):
        super().__init__()
        self.bert = BertModel.from_pretrained(bert_name)
        emb_dim = self.bert.config.hidden_size
        # conv expects in_channels=emb_dim
        self.conv = nn.Conv1d(in_channels=emb_dim, out_channels=cnn_filters, kernel_size=3, padding=1)
        self.lstm = nn.LSTM(input_size=cnn_filters, hidden_size=lstm_hidden, batch_first=True, bidirectional=False)
        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(lstm_hidden, num_labels)
    def forward(self, input_ids=None, attention_mask=None):
        out = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        seq = out.last_hidden_state          # [B, L, emb_dim]
        x = seq.permute(0, 2, 1)            # [B, emb_dim, L]
        x = torch.relu(self.conv(x))        # [B, cnn_filters, L]
        x = x.permute(0, 2, 1)              # [B, L, cnn_filters]
        h, _ = self.lstm(x)                 # [B, L, hidden]
        last = h[:, -1, :]                  # [B, hidden]
        logits = self.fc(self.dropout(last))
        return logits

class DistilCNNBiLSTM(nn.Module):
    def __init__(self, distil_name=DISTIL_NAME, cnn_filters=128, lstm_hidden=128, num_labels=2, dropout=0.3):
        super().__init__()
        self.distil = DistilBertModel.from_pretrained(distil_name)
        emb_dim = self.distil.config.hidden_size
        self.conv = nn.Conv1d(in_channels=emb_dim, out_channels=cnn_filters, kernel_size=3, padding=1)
        self.lstm = nn.LSTM(input_size=cnn_filters, hidden_size=lstm_hidden, batch_first=True, bidirectional=True)
        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(lstm_hidden*2, num_labels)
    def forward(self, input_ids=None, attention_mask=None):
        out = self.distil(input_ids=input_ids, attention_mask=attention_mask)
        seq = out.last_hidden_state
        x = seq.permute(0, 2, 1)
        x = torch.relu(self.conv(x))
        x = x.permute(0, 2, 1)
        h, _ = self.lstm(x)
        last = h[:, -1, :]
        logits = self.fc(self.dropout(last))
        return logits

class MetadataMLP(nn.Module):
    def __init__(self, input_dim, hidden=64, num_labels=2, dropout=0.2):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden, num_labels)
        )
    def forward(self, features):
        return self.net(features)

# ---------------------------
# SAFE torch.load wrapper
# ---------------------------
def safe_torch_load(path, map_location):
    """
    Try to load with weights_only=True (if supported), otherwise fallback.
    """
    try:
        # newer PyTorch param to limit unpickling
        return torch.load(path, map_location=map_location, weights_only=True)
    except TypeError:
        # older PyTorch: no weights_only arg
        return torch.load(path, map_location=map_location)
    except Exception as e:
        # fallback: try normal load (user accepts warning)
        print("Warning while using weights_only=True:", e)
        return torch.load(path, map_location=map_location)

# ---------------------------
# INFERENCE helpers
# ---------------------------
def batch_predict_text_model(model, tokenizer, texts, batch_size=BATCH_SIZE, max_len=MAX_LEN):
    model.eval()
    all_probs = []
    with torch.no_grad():
        for i in range(0, len(texts), batch_size):
            batch = texts[i:i+batch_size]
            enc = tokenizer(batch, padding="max_length", truncation=True, max_length=max_len, return_tensors="pt")
            input_ids = enc["input_ids"].to(DEVICE)
            attention_mask = enc["attention_mask"].to(DEVICE)
            logits = model(input_ids=input_ids, attention_mask=attention_mask)
            probs = torch.softmax(logits, dim=1).cpu().numpy()
            all_probs.append(probs)
    if all_probs:
        return np.vstack(all_probs)
    else:
        return np.zeros((0,2))

def batch_predict_meta(model, feats, batch_size=BATCH_SIZE):
    model.eval()
    all_probs = []
    with torch.no_grad():
        for i in range(0, len(feats), batch_size):
            b = feats[i:i+batch_size]
            t = torch.tensor(b, dtype=torch.float32).to(DEVICE)
            logits = model(t)
            probs = torch.softmax(logits, dim=1).cpu().numpy()
            all_probs.append(probs)
    if all_probs:
        return np.vstack(all_probs)
    else:
        return np.zeros((0,2))

# ---------------------------
# MAIN
# ---------------------------
def main():
    # load test data
    if not os.path.exists("spam.csv"):
        raise SystemExit("spam.csv not found in current directory.")

    df = pd.read_csv("spam.csv", encoding="latin1", low_memory=False)
    # common Kaggle spam columns are 'v1' (label) and 'v2' (text)
    if "text" in df.columns:
        text_col = "text"
    elif "message" in df.columns:
        text_col = "message"
    elif "v2" in df.columns:
        text_col = "v2"
    else:
        # fallback to first column that looks like text
        text_col = df.columns[0]

    if "label" in df.columns:
        label_col = "label"
    elif "v1" in df.columns:
        label_col = "v1"
    else:
        # if no label present, we still run inference; create dummy labels
        label_col = None

    texts_raw = df[text_col].astype(str).tolist()
    texts = [clean_text(t) for t in texts_raw]

    # labels -> numeric mapping similar to training mapping
    if label_col:
        raw_labels = df[label_col].astype(str).str.lower().tolist()
        labels = [1 if ("spam" in x or "phish" in x or x.strip() == "1") else 0 for x in raw_labels]
    else:
        labels = None

    # metadata features
    meta_feats = extract_metadata_features(texts)   # shape [N, 6]

    # scaler: try to load saved mean/scale, otherwise fit on current meta_feats
    scaler = StandardScaler()
    if os.path.exists(SCALER_MEAN_FILE) and os.path.exists(SCALER_SCALE_FILE):
        try:
            mean = np.load(SCALER_MEAN_FILE)
            scale = np.load(SCALER_SCALE_FILE)
            scaler.mean_ = mean
            scaler.scale_ = scale
            scaler.var_ = scale ** 2
            scaler.n_features_in_ = meta_feats.shape[1]
            meta_feats_scaled = scaler.transform(meta_feats)
            print("Loaded saved scaler mean/scale files.")
        except Exception as e:
            print("Could not load scaler files cleanly, fitting scaler on inference set. Error:", e)
            scaler = StandardScaler().fit(meta_feats)
            meta_feats_scaled = scaler.transform(meta_feats)
    else:
        # No saved scaler available -> fit on test set (not perfect, but workable)
        print("No saved scaler found — fitting StandardScaler on inference set (if you have a saved scaler from training, put scaler_mean.npy & scaler_scale.npy).")
        scaler = StandardScaler().fit(meta_feats)
        meta_feats_scaled = scaler.transform(meta_feats)

    # ---------------------------
    # Load tokenizers & models
    # ---------------------------
    print("Loading tokenizers...")
    bert_tokenizer = BertTokenizer.from_pretrained(BERT_NAME)
    distil_tokenizer = DistilBertTokenizer.from_pretrained(DISTIL_NAME)

    print("Building model objects (architectures must match training)...")
    model_bert = BertCNNLSTM().to(DEVICE)
    model_distil = DistilCNNBiLSTM().to(DEVICE)
    model_meta = MetadataMLP(input_dim=meta_feats.shape[1]).to(DEVICE)

    # load checkpoints
    if not os.path.exists(BERT_CKPT) or not os.path.exists(DISTIL_CKPT) or not os.path.exists(META_CKPT):
        raise SystemExit("One or more checkpoint files are missing (bert_best.pt, distil_best.pt, meta_best.pt)")

    print("Loading weights...")
    model_bert.load_state_dict(safe_torch_load(BERT_CKPT, map_location=DEVICE))
    model_distil.load_state_dict(safe_torch_load(DISTIL_CKPT, map_location=DEVICE))
    model_meta.load_state_dict(safe_torch_load(META_CKPT, map_location=DEVICE))

    model_bert.eval()
    model_distil.eval()
    model_meta.eval()

    # ---------------------------
    # Predictions
    # ---------------------------
    print("Running BERT predictions...")
    probs_bert = batch_predict_text_model(model_bert, bert_tokenizer, texts, batch_size=BATCH_SIZE, max_len=MAX_LEN)
    print("Running DistilBERT predictions...")
    probs_distil = batch_predict_text_model(model_distil, distil_tokenizer, texts, batch_size=BATCH_SIZE, max_len=MAX_LEN)
    print("Running Metadata MLP predictions...")
    probs_meta = batch_predict_meta(model_meta, meta_feats_scaled, batch_size=BATCH_SIZE)

    # Safety: if dimensions mismatch (e.g., model returns zero rows), handle gracefully
    n = len(texts)
    if probs_bert.shape[0] != n:
        probs_bert = np.zeros((n,2))
    if probs_distil.shape[0] != n:
        probs_distil = np.zeros((n,2))
    if probs_meta.shape[0] != n:
        probs_meta = np.zeros((n,2))

    # ensemble (soft voting weighted)
    ensemble_probs = WEIGHT_BERT * probs_bert + WEIGHT_DISTIL * probs_distil + WEIGHT_META * probs_meta
    ensemble_preds = np.argmax(ensemble_probs, axis=1)
    confidences_phish = ensemble_probs[:, 1]

    # ---------------------------
    # Evaluation / Save results
    # ---------------------------
    if labels is not None:
        print("\n=== Classification Report ===")
        print(classification_report(labels, ensemble_preds, target_names=["ham", "phish"], zero_division=0))
        print("Confusion matrix:\n", confusion_matrix(labels, ensemble_preds))
    else:
        print("No ground-truth labels found; skipping classification report.")

    # Save predictions
    out_df = df.copy()
    out_df["text_clean"] = texts
    out_df["pred_label_num"] = ensemble_preds
    out_df["pred_label"] = out_df["pred_label_num"].map({0: "ham", 1: "phish"})
    out_df["phish_confidence"] = confidences_phish
    out_file = "spam_predictions.csv"
    out_df.to_csv(out_file, index=False, encoding="utf-8")
    print(f"\nSaved predictions to {out_file}")

if __name__ == "__main__":
    main()
