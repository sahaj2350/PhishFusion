# phishfusion_full_pipeline.py
"""
PhishFusion - full multimodal phishing pipeline (text + metadata + ensemble)

Features:
 - Loads many CSV/text datasets (SpamAssassin, Enron, Nazario, Phishtank, Zefang-Liu)
 - Cleans, dedupes, and filters short texts
 - Lightweight augmentation for phishing class (WordNet synonyms + char-swap)
 - Extracts metadata features (length, digits, url counts, punctuation, uppercase ratio)
 - Builds BERT-CNN-LSTM and DistilBERT-CNN-BiLSTM text models
 - Builds a small metadata MLP
 - Trains each model separately with class-weighted loss
 - Soft-voting ensemble of the three models (weighted average of probabilities)
 - Evaluation metrics: accuracy, precision, recall, F1
"""

import os
import re
import glob
import random
import math
from typing import List, Tuple, Dict
import pandas as pd
import numpy as np
import torch
from torch import nn
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split
from sklearn.utils.class_weight import compute_class_weight
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from transformers import BertTokenizer, BertModel, DistilBertTokenizer, DistilBertModel, get_linear_schedule_with_warmup
from torch.optim import AdamW

# Optional: nltk wordnet for synonyms (lightweight usage)
import nltk
from nltk.corpus import wordnet
try:
    wordnet.ensure_loaded()
except Exception:
    nltk.download("wordnet")
    nltk.download("omw-1.4")

# ---------------------------
# CONFIG
# ---------------------------
SEED = 42
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# Dataset paths - edit if needed
PATH_SPAM_CSV = "spam.csv"
PATH_NAZARIO = "Nazario_5 - Copy.csv"
PATH_ENRON = "emails.csv"
PATH_PHISHTANK = "phishtank.csv"
PATH_ZEFANG_HF = "hf://datasets/zefang-liu/phishing-email-dataset/Phishing_Email.csv"
SPAMASSASSIN_ROOT = "./SpamAssassin"

# Model / training hyperparams
BERT_NAME = "bert-base-uncased"
DISTIL_NAME = "distilbert-base-uncased"
MAX_LEN = 128
BATCH_SIZE = 8          # reduce if GPU OOM
EPOCHS = 3              # increase for final runs
LR = 2e-5
AUGMENT_TARGET_MODE = "balance"  # "balance" or "fixed_total"
TARGET_TOTAL_ROWS = 750_000      # used only if AUGMENT_TARGET_MODE == "fixed_total"

# Ensemble weights (sum to 1)
WEIGHT_BERT = 0.45
WEIGHT_DISTIL = 0.45
WEIGHT_META = 0.10

# Early stopping patience (epochs with no improvement on val f1)
PATIENCE = 2

# ---------------------------
# HELPERS: loading + cleaning
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

def load_csv_generic(path: str, label_default=None) -> pd.DataFrame:
    if not os.path.exists(path):
        return None
    try:
        df = pd.read_csv(path, encoding="latin1", low_memory=False)
    except Exception as e:
        print(f"⚠️ Could not read {path}: {e}")
        return None
    cols_lower = [c.lower().strip() for c in df.columns]
    text_candidates = ["message", "text", "body", "content", "email_text", "message_text"]
    text_col = None
    for cand in text_candidates:
        if cand in cols_lower:
            text_col = df.columns[cols_lower.index(cand)]
            break
    if text_col is None:
        text_col = df.columns[0]
    df = df.rename(columns={text_col: "text"})
    df["text"] = df["text"].astype(str)
    label_candidates = ["label", "category", "class", "y", "is_spam"]
    label_col = None
    for cand in label_candidates:
        if cand in cols_lower:
            label_col = df.columns[cols_lower.index(cand)]
            break
    if label_col is not None:
        lbls = df[label_col].astype(str).str.lower()
        df["label"] = lbls.map(lambda x: 1 if ("spam" in x or "phish" in x or x.strip() == "1") else 0)
    elif label_default is not None:
        df["label"] = label_default
    else:
        print(f"⚠️ {path} has no label column and no default — skipping.")
        return None
    return df[["text", "label"]]

def load_spamassassin(root_dir: str) -> pd.DataFrame:
    if not os.path.exists(root_dir):
        return None
    parts = []
    mapping = [("easy_ham", 0), ("easy_ham_2", 0), ("hard_ham", 0), ("spam", 1), ("spam_2", 1)]
    for folder, lbl in mapping:
        fpath = os.path.join(root_dir, folder)
        if not os.path.isdir(fpath):
            continue
        texts = []
        for fp in glob.glob(os.path.join(fpath, "**", "*.*"), recursive=True):
            if not os.path.isfile(fp):
                continue
            try:
                with open(fp, "r", encoding="latin1", errors="ignore") as fh:
                    texts.append(fh.read())
            except Exception:
                continue
        if texts:
            parts.append(pd.DataFrame({"text": texts, "label": lbl}))
            print(f"✅ Loaded {len(texts)} files from SpamAssassin folder: {folder}")
    return pd.concat(parts, ignore_index=True) if parts else None

# ---------------------------
# LIGHTWEIGHT AUGMENTATION (no heavy nlpaug/nltk tagger)
# - synonym replacement using wordnet (take first synset lemma)
# - random char swap in word
# ---------------------------
def synonym_replace(sentence: str, replace_prob: float = 0.1) -> str:
    words = sentence.split()
    out = []
    for w in words:
        if random.random() < replace_prob:
            syns = wordnet.synsets(w)
            if syns:
                # pick first lemma that's not the same as original
                lemmas = [l.name().replace("_", " ") for l in syns[0].lemmas() if l.name().lower() != w.lower()]
                if lemmas:
                    out.append(random.choice(lemmas))
                    continue
        out.append(w)
    return " ".join(out)

def char_swap(sentence: str, swap_prob: float = 0.05) -> str:
    # swap two adjacent characters in random words with swap_prob
    def swap_chars_in_word(w):
        if len(w) < 3 or random.random() > swap_prob:
            return w
        i = random.randint(0, len(w)-2)
        lst = list(w)
        lst[i], lst[i+1] = lst[i+1], lst[i]
        return "".join(lst)
    return " ".join(swap_chars_in_word(w) for w in sentence.split())

def augment_texts_minority(texts: List[str], target_count: int) -> List[str]:
    """Generate augmented texts from given list until target_count reached.
        Returns list of augmented texts (not including originals)."""
    augmented = []
    if not texts:
        return augmented
    i = 0
    while len(augmented) < max(0, target_count - len(texts)):
        base = texts[i % len(texts)]
        s = synonym_replace(base, replace_prob=0.12)    # tune
        s = char_swap(s, swap_prob=0.06)
        # ensure it's not a duplicate of base or previous
        if s and s != base and s not in augmented:
            augmented.append(s)
        i += 1
        # safety break (shouldn't happen but avoid infinite loop)
        if i > target_count * 10:
            break
    return augmented

# ---------------------------
# METADATA FEATURES
# - message length
# - number of digits
# - count of urls (http/www)
# - punctuation count
# - uppercase letter ratio
# - domain token (if url present) — simple extraction
# ---------------------------
url_re = re.compile(r"(https?://[^\s]+|www\.[^\s]+)")

def extract_metadata_features(texts: List[str]) -> np.ndarray:
    feats = []
    for t in texts:
        ln = len(t)
        digits = sum(c.isdigit() for c in t)
        urls = len(url_re.findall(t))
        punct = sum(1 for c in t if c in "!?.,;:")
        uppercase_ratio = (sum(1 for c in t if c.isupper()) / (ln + 1))  # after cleaning may be near 0
        # basic domain token count: count occurrences of common domain tokens
        domain_tokens = sum(tok in t for tok in [".com", ".net", ".org", ".ru", ".cn", ".info"])
        feats.append([ln, digits, urls, punct, uppercase_ratio, domain_tokens])
    arr = np.array(feats, dtype=np.float32)
    # scale later with StandardScaler
    return arr

# ---------------------------
# DATASET & DATALOADER classes
# ---------------------------
class TextDataset(Dataset):
    def __init__(self, texts: List[str], labels: List[int], tokenizer, max_len=128):
        self.texts = texts
        self.labels = labels
        self.tokenizer = tokenizer
        self.max_len = max_len
    def __len__(self): return len(self.texts)
    def __getitem__(self, idx):
        t = self.texts[idx]
        enc = self.tokenizer(t, truncation=True, padding="max_length", max_length=self.max_len, return_tensors="pt")
        item = {k: v.squeeze(0) for k, v in enc.items()}
        item["labels"] = torch.tensor(int(self.labels[idx]), dtype=torch.long)
        return item

class MetaDataset(Dataset):
    def __init__(self, feats: np.ndarray, labels: List[int]):
        self.X = feats
        self.y = np.array(labels, dtype=np.int64)
    def __len__(self): return len(self.X)
    def __getitem__(self, idx):
        return {"features": torch.tensor(self.X[idx], dtype=torch.float32),
                "labels": torch.tensor(self.y[idx], dtype=torch.long)}

# ---------------------------
# MODEL DEFINITIONS
# (text: encode -> conv -> LSTM -> classifier)
# (meta: simple MLP)
# ---------------------------
class BertCNNLSTM(nn.Module):
    def __init__(self, bert_name=BERT_NAME, cnn_filters=128, lstm_hidden=128, num_labels=2, dropout=0.3):
        super().__init__()
        self.bert = BertModel.from_pretrained(bert_name)
        emb_dim = self.bert.config.hidden_size
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
# TRAIN / EVAL UTILITIES
# ---------------------------
def compute_metrics_from_labels(preds: List[int], labels: List[int]) -> Dict[str, float]:
    return {
        "accuracy": accuracy_score(labels, preds),
        "precision": precision_score(labels, preds, zero_division=0),
        "recall": recall_score(labels, preds, zero_division=0),
        "f1": f1_score(labels, preds, zero_division=0)
    }

def train_epoch_text(model: nn.Module, dataloader: DataLoader, optimizer, scheduler, class_weights_tensor=None):
    model.train()
    total_loss = 0.0
    for batch in dataloader:
        optimizer.zero_grad()
        input_ids = batch["input_ids"].to(DEVICE)
        attention_mask = batch["attention_mask"].to(DEVICE)
        labels = batch["labels"].to(DEVICE)
        logits = model(input_ids=input_ids, attention_mask=attention_mask)
        if class_weights_tensor is not None:
            loss_fct = nn.CrossEntropyLoss(weight=class_weights_tensor.to(DEVICE))
        else:
            loss_fct = nn.CrossEntropyLoss()
        loss = loss_fct(logits.view(-1, logits.size(-1)), labels.view(-1))
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        if scheduler is not None:
            scheduler.step()
        total_loss += loss.item()
    return total_loss / len(dataloader)

def eval_text(model: nn.Module, dataloader: DataLoader) -> Tuple[List[int], List[int], np.ndarray]:
    model.eval()
    all_preds = []
    all_labels = []
    all_probs = []
    with torch.no_grad():
        for batch in dataloader:
            input_ids = batch["input_ids"].to(DEVICE)
            attention_mask = batch["attention_mask"].to(DEVICE)
            labels = batch["labels"].to(DEVICE)
            logits = model(input_ids=input_ids, attention_mask=attention_mask)
            probs = torch.softmax(logits, dim=1).cpu().numpy()
            preds = np.argmax(probs, axis=1)
            all_preds.extend(preds.tolist())
            all_labels.extend(labels.cpu().numpy().tolist())
            all_probs.append(probs)
    probs_arr = np.vstack(all_probs) if all_probs else np.zeros((0,2))
    return all_preds, all_labels, probs_arr

def train_epoch_meta(model: nn.Module, dataloader: DataLoader, optimizer, class_weights_tensor=None):
    model.train()
    total_loss = 0.0
    for batch in dataloader:
        optimizer.zero_grad()
        feats = batch["features"].to(DEVICE)
        labels = batch["labels"].to(DEVICE)
        logits = model(feats)
        if class_weights_tensor is not None:
            loss_fct = nn.CrossEntropyLoss(weight=class_weights_tensor.to(DEVICE))
        else:
            loss_fct = nn.CrossEntropyLoss()
        loss = loss_fct(logits.view(-1, logits.size(-1)), labels.view(-1))
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
    return total_loss / len(dataloader)

def eval_meta(model: nn.Module, dataloader: DataLoader) -> Tuple[List[int], List[int], np.ndarray]:
    model.eval()
    all_preds = []
    all_labels = []
    all_probs = []
    with torch.no_grad():
        for batch in dataloader:
            feats = batch["features"].to(DEVICE)
            labels = batch["labels"].to(DEVICE)
            logits = model(feats)
            probs = torch.softmax(logits, dim=1).cpu().numpy()
            preds = np.argmax(probs, axis=1)
            all_preds.extend(preds.tolist())
            all_labels.extend(labels.cpu().numpy().tolist())
            all_probs.append(probs)
    probs_arr = np.vstack(all_probs) if all_probs else np.zeros((0,2))
    return all_preds, all_labels, probs_arr

# ---------------------------
# MAIN PIPELINE
# ---------------------------
def main():
    print("Device:", DEVICE)
    # -----------------------
    # 1) Load datasets
    # -----------------------
    frames = []
    for (path, default) in [
        (PATH_SPAM_CSV, None),
        (PATH_NAZARIO, 1),
        (PATH_ENRON, 0),
        (PATH_PHISHTANK, 1),
    ]:
        df = load_csv_generic(path, label_default=default)
        if df is not None:
            frames.append(df)
            print(f"✅ Loaded {path} -> {len(df)} rows")

    # Zefang-Liu attempt via pandas (hf:// may or may not work in local env)
    try:
        if PATH_ZEFANG_HF:
            print("🔎 Trying to read Zefang-Liu via pandas from PATH_ZEFANG_HF...")
            dfz = pd.read_csv(PATH_ZEFANG_HF, encoding="latin1", low_memory=False)
            if "Email Text" in dfz.columns:
                dfz = dfz.rename(columns={"Email Text": "text", "Email Type": "label"})
            if "text" in dfz.columns and "label" in dfz.columns:
                dfz["label"] = dfz["label"].astype(str).map(lambda x: 1 if "phishing" in x.lower() else 0)
                frames.append(dfz[["text", "label"]].dropna())
                print(f"✅ Loaded Zefang-Liu via pandas: {len(dfz)} rows")
    except Exception as e:
        print("⚠️ Could not load Zefang-Liu via pandas:", e)

    # SpamAssassin
    sa = load_spamassassin(SPAMASSASSIN_ROOT)
    if sa is not None:
        frames.append(sa)
        print(f"✅ Added SpamAssassin combined: {len(sa)} rows")

    if not frames:
        raise SystemExit("No datasets found. Check paths.")

    df_all = pd.concat(frames, ignore_index=True).dropna(subset=["text"])
    print("Raw merged rows:", len(df_all))

    # clean and dedupe
    df_all["text"] = df_all["text"].map(clean_text)
    df_all = df_all[df_all["text"].str.len() > 5].copy()
    before = len(df_all)
    df_all = df_all.drop_duplicates(subset=["text"]).reset_index(drop=True)
    after = len(df_all)
    print(f"After clean & dedupe: {after} (removed {before-after}) ; label dist:\n{df_all['label'].value_counts()}")

    # -----------------------
    # 2) Augmentation of minority (phish)
    # -----------------------
    n_total = len(df_all)
    n_phish = int(df_all["label"].sum())
    n_ham = n_total - n_phish
    print(f"Current total: {n_total}, phish: {n_phish}, ham: {n_ham}")

    if AUGMENT_TARGET_MODE == "balance":
        target_phish = max(n_phish, n_ham)
    else:
        # fixed total target
        target_phish = int(min(n_phish + (TARGET_TOTAL_ROWS - n_total), n_phish + (TARGET_TOTAL_ROWS - n_total)))
        target_phish = max(target_phish, n_phish)

    print("Target phishing count after augmentation:", target_phish)
    if n_phish < target_phish:
        phish_texts = df_all.loc[df_all["label"] == 1, "text"].tolist()
        aug_texts = augment_texts_minority(phish_texts, target_phish)
        print("Augmented phishing samples created:", len(aug_texts))
        aug_df = pd.DataFrame({"text": aug_texts, "label": 1})
        df_all = pd.concat([df_all, aug_df], ignore_index=True).drop_duplicates(subset=["text"]).reset_index(drop=True)
    else:
        print("No augmentation needed.")

    print("After augmentation & dedupe: total:", len(df_all), "label dist:\n", df_all["label"].value_counts())

    # -----------------------
    # 3) Extract metadata & scale
    # -----------------------
    texts = df_all["text"].tolist()
    labels = df_all["label"].astype(int).tolist()
    meta_feats = extract_metadata_features(texts)
    scaler = StandardScaler().fit(meta_feats)
    meta_feats_scaled = scaler.transform(meta_feats)

    # -----------------------
    # 4) Train/Val split (stratified)
    # -----------------------
    train_idx, val_idx = train_test_split(list(range(len(texts))), test_size=0.15, stratify=labels, random_state=SEED)
    train_texts = [texts[i] for i in train_idx]
    val_texts = [texts[i] for i in val_idx]
    train_labels = [labels[i] for i in train_idx]
    val_labels = [labels[i] for i in val_idx]
    train_meta = meta_feats_scaled[train_idx]
    val_meta = meta_feats_scaled[val_idx]

    # -----------------------
    # 5) Tokenizers & Datasets
    # -----------------------
    tokenizer_bert = BertTokenizer.from_pretrained(BERT_NAME)
    tokenizer_distil = DistilBertTokenizer.from_pretrained(DISTIL_NAME)

    train_dataset_bert = TextDataset(train_texts, train_labels, tokenizer_bert, MAX_LEN)
    val_dataset_bert = TextDataset(val_texts, val_labels, tokenizer_bert, MAX_LEN)
    train_dataset_distil = TextDataset(train_texts, train_labels, tokenizer_distil, MAX_LEN)
    val_dataset_distil = TextDataset(val_texts, val_labels, tokenizer_distil, MAX_LEN)
    train_dataset_meta = MetaDataset(train_meta, train_labels)
    val_dataset_meta = MetaDataset(val_meta, val_labels)

    train_loader_bert = DataLoader(train_dataset_bert, batch_size=BATCH_SIZE, shuffle=True)
    val_loader_bert = DataLoader(val_dataset_bert, batch_size=BATCH_SIZE)
    train_loader_distil = DataLoader(train_dataset_distil, batch_size=BATCH_SIZE, shuffle=True)
    val_loader_distil = DataLoader(val_dataset_distil, batch_size=BATCH_SIZE)
    train_loader_meta = DataLoader(train_dataset_meta, batch_size=BATCH_SIZE, shuffle=True)
    val_loader_meta = DataLoader(val_dataset_meta, batch_size=BATCH_SIZE)

    # -----------------------
    # 6) Class weights
    # -----------------------
    class_weights = compute_class_weight(class_weight="balanced", classes=np.array([0,1]), y=train_labels)
    class_weights_tensor = torch.tensor(class_weights, dtype=torch.float32)

    # -----------------------
    # 7) Model init
    # -----------------------
    model_bert = BertCNNLSTM().to(DEVICE)
    model_distil = DistilCNNBiLSTM().to(DEVICE)
    model_meta = MetadataMLP(input_dim=train_meta.shape[1]).to(DEVICE)

    optimizer_bert = AdamW(model_bert.parameters(), lr=LR)
    optimizer_distil = AdamW(model_distil.parameters(), lr=LR)
    optimizer_meta = torch.optim.Adam(model_meta.parameters(), lr=1e-3)

    total_steps_bert = len(train_loader_bert) * EPOCHS
    scheduler_bert = get_linear_schedule_with_warmup(optimizer_bert, num_warmup_steps=0, num_training_steps=total_steps_bert)
    total_steps_distil = len(train_loader_distil) * EPOCHS
    scheduler_distil = get_linear_schedule_with_warmup(optimizer_distil, num_warmup_steps=0, num_training_steps=total_steps_distil)

    # -----------------------
    # 8) Training loops with early stopping
    # -----------------------
    def train_model_text(model, train_loader, val_loader, optimizer, scheduler, name="model"):
        best_f1 = 0.0
        patience_counter = 0
        for epoch in range(EPOCHS):
            loss = train_epoch_text(model, train_loader, optimizer, scheduler, class_weights_tensor)
            preds, labels, _ = eval_text(model, val_loader)
            metrics = compute_metrics_from_labels(preds, labels)
            print(f"[{name}] Epoch {epoch+1}/{EPOCHS}, Loss: {loss:.4f}, Val F1: {metrics['f1']:.4f}")
            if metrics["f1"] > best_f1:
                best_f1 = metrics["f1"]
                torch.save(model.state_dict(), f"{name}_best.pt")
                patience_counter = 0
            else:
                patience_counter += 1
                if patience_counter >= PATIENCE:
                    print(f"[{name}] Early stopping at epoch {epoch+1}")
                    break
        model.load_state_dict(torch.load(f"{name}_best.pt"))

    def train_model_meta(model, train_loader, val_loader, optimizer, name="meta"):
        best_f1 = 0.0
        patience_counter = 0
        for epoch in range(EPOCHS):
            loss = train_epoch_meta(model, train_loader, optimizer, class_weights_tensor)
            preds, labels, _ = eval_meta(model, val_loader)
            metrics = compute_metrics_from_labels(preds, labels)
            print(f"[{name}] Epoch {epoch+1}/{EPOCHS}, Loss: {loss:.4f}, Val F1: {metrics['f1']:.4f}")
            if metrics["f1"] > best_f1:
                best_f1 = metrics["f1"]
                torch.save(model.state_dict(), f"{name}_best.pt")
                patience_counter = 0
            else:
                patience_counter += 1
                if patience_counter >= PATIENCE:
                    print(f"[{name}] Early stopping at epoch {epoch+1}")
                    break
        model.load_state_dict(torch.load(f"{name}_best.pt"))

    print("=== Training BERT model ===")
    train_model_text(model_bert, train_loader_bert, val_loader_bert, optimizer_bert, scheduler_bert, "bert")
    print("=== Training DistilBERT model ===")
    train_model_text(model_distil, train_loader_distil, val_loader_distil, optimizer_distil, scheduler_distil, "distil")
    print("=== Training metadata MLP ===")
    train_model_meta(model_meta, train_loader_meta, val_loader_meta, optimizer_meta, "meta")

    # -----------------------
    # 9) Ensemble Evaluation
    # -----------------------
    _, _, probs_bert = eval_text(model_bert, val_loader_bert)
    _, _, probs_distil = eval_text(model_distil, val_loader_distil)
    _, _, probs_meta = eval_meta(model_meta, val_loader_meta)

    # soft voting weighted average
    ensemble_probs = WEIGHT_BERT*probs_bert + WEIGHT_DISTIL*probs_distil + WEIGHT_META*probs_meta
    ensemble_preds = np.argmax(ensemble_probs, axis=1)
    metrics_ens = compute_metrics_from_labels(ensemble_preds.tolist(), val_labels)
    print("=== Ensemble metrics ===")
    print(metrics_ens)

if __name__ == "__main__":
    main()
