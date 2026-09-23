# =====================================================
# IMPORTS
# =====================================================
import os
import random
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, models
from torchvision.models import ResNet50_Weights
from transformers import ViTModel
from PIL import Image
from collections import Counter
from sklearn.metrics import precision_score, recall_score, f1_score
from tqdm import tqdm
from multiprocessing import freeze_support

# =====================================================
# CONFIG
# =====================================================
ROOT = r"C:\Users\sketc\Documents\4th Year Research\Multimodal\Image Modality\Images"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
BATCH_SIZE = 32
EPOCHS = 20
NUM_WORKERS = 4        # set to 0 if still debugging
SEED = 42
PATIENCE = 3

# =====================================================
# REPRODUCIBILITY
# =====================================================
def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

set_seed(SEED)

# =====================================================
# IMAGE EXTENSIONS
# =====================================================
IMG_EXTENSIONS = (".png", ".jpg", ".jpeg", ".bmp", ".webp")

def is_image(path):
    return path.lower().endswith(IMG_EXTENSIONS)

# =====================================================
# DATASET
# =====================================================
class PhishImageDataset(Dataset):
    def __init__(self, root_dir, transform=None, split=None):
        self.samples = []
        self.transform = transform

        self._load_flat(os.path.join(root_dir, "genuine_site_0"), 0)
        self._load_flat(os.path.join(root_dir, "phishing_site_1"), 1)

        self._load_nested(os.path.join(root_dir, "phishing"), 1)
        self._load_nested(os.path.join(root_dir, "not-phishing"), 0)

        iris_root = os.path.join(root_dir, "phishIRIS_DL_Dataset")
        if split:
            self._load_phishiris(os.path.join(iris_root, split), 1)

        print(f"[INFO] Loaded {len(self.samples)} images")

    def _load_flat(self, folder, label):
        if not os.path.exists(folder):
            return
        for f in os.listdir(folder):
            p = os.path.join(folder, f)
            if os.path.isfile(p) and is_image(p):
                self.samples.append((p, label))

    def _load_nested(self, folder, label):
        if not os.path.exists(folder):
            return
        for root, _, files in os.walk(folder):
            if os.path.basename(root) == "screenshots":
                for f in files:
                    p = os.path.join(root, f)
                    if is_image(p):
                        self.samples.append((p, label))

    def _load_phishiris(self, folder, label):
        if not os.path.exists(folder):
            return
        for brand in os.listdir(folder):
            bpath = os.path.join(folder, brand)
            if os.path.isdir(bpath):
                for f in os.listdir(bpath):
                    p = os.path.join(bpath, f)
                    if is_image(p):
                        self.samples.append((p, label))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        path, label = self.samples[idx]
        img = Image.open(path).convert("RGB")
        if self.transform:
            img = self.transform(img)
        return img, label

# =====================================================
# TRANSFORMS
# =====================================================
train_tfms = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.RandomHorizontalFlip(),
    transforms.ColorJitter(0.2, 0.2, 0.2, 0.1),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406],
                         [0.229, 0.224, 0.225])
])

val_tfms = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406],
                         [0.229, 0.224, 0.225])
])

# =====================================================
# MODELS
# =====================================================
class CNNClassifier(nn.Module):
    def __init__(self):
        super().__init__()
        self.backbone = models.resnet50(
            weights=ResNet50_Weights.DEFAULT
        )
        in_features = self.backbone.fc.in_features
        self.backbone.fc = nn.Sequential(
            nn.Linear(in_features, 256),
            nn.ReLU(),
            nn.Dropout(0.4),
            nn.Linear(256, 2)
        )

    def forward(self, x):
        return self.backbone(x)

class ViTClassifier(nn.Module):
    def __init__(self):
        super().__init__()
        self.vit = ViTModel.from_pretrained(
            "google/vit-base-patch16-224-in21k"
        )
        self.classifier = nn.Sequential(
            nn.Linear(self.vit.config.hidden_size, 256),
            nn.ReLU(),
            nn.Dropout(0.4),
            nn.Linear(256, 2)
        )

    def forward(self, x):
        out = self.vit(pixel_values=x)
        cls = out.last_hidden_state[:, 0]
        return self.classifier(cls)

# =====================================================
# METRICS
# =====================================================
def compute_metrics(y_true, y_pred):
    return {
        "precision": precision_score(y_true, y_pred),
        "recall": recall_score(y_true, y_pred),
        "f1": f1_score(y_true, y_pred)
    }

# =====================================================
# EARLY STOPPING
# =====================================================
class EarlyStopping:
    def __init__(self, patience):
        self.patience = patience
        self.counter = 0
        self.best = None
        self.stop = False

    def step(self, score):
        if self.best is None or score > self.best:
            self.best = score
            self.counter = 0
        else:
            self.counter += 1
            if self.counter >= self.patience:
                self.stop = True

# =====================================================
# TRAIN / VALIDATE
# =====================================================
def train_epoch(model, loader, optimizer, criterion):
    model.train()
    total = 0
    for x, y in tqdm(loader, leave=False):
        x, y = x.to(DEVICE), y.to(DEVICE)
        optimizer.zero_grad()
        loss = criterion(model(x), y)
        loss.backward()
        optimizer.step()
        total += loss.item()
    return total / len(loader)

@torch.no_grad()
def validate(model, loader, criterion):
    model.eval()
    total = 0
    y_true, y_pred = [], []
    for x, y in loader:
        x, y = x.to(DEVICE), y.to(DEVICE)
        out = model(x)
        loss = criterion(out, y)
        preds = torch.argmax(out, dim=1)
        total += loss.item()
        y_true.extend(y.cpu().numpy())
        y_pred.extend(preds.cpu().numpy())
    return total / len(loader), compute_metrics(y_true, y_pred)

# =====================================================
# MAIN (WINDOWS SAFE)
# =====================================================
if __name__ == "__main__":
    freeze_support()

    # ---------------- DATA ----------------
    train_ds = PhishImageDataset(ROOT, train_tfms, split="train")
    val_ds   = PhishImageDataset(ROOT, val_tfms, split="val")

    train_loader = DataLoader(
        train_ds, BATCH_SIZE, True,
        num_workers=NUM_WORKERS, pin_memory=True
    )
    val_loader = DataLoader(
        val_ds, BATCH_SIZE, False,
        num_workers=NUM_WORKERS, pin_memory=True
    )

    # ---------------- CLASS WEIGHTS ----------------
    labels = [l for _, l in train_ds.samples]
    count = Counter(labels)
    weights = torch.tensor([1/count[0], 1/count[1]]).to(DEVICE)
    criterion = nn.CrossEntropyLoss(weight=weights)

    # ---------------- TRAIN CNN ----------------
    cnn = CNNClassifier().to(DEVICE)
    cnn_opt = torch.optim.AdamW(cnn.parameters(), lr=1e-4)
    early = EarlyStopping(PATIENCE)
    cnn_best_f1 = 0.0

    for e in range(EPOCHS):
        train_epoch(cnn, train_loader, cnn_opt, criterion)
        _, m = validate(cnn, val_loader, criterion)
        print(f"[CNN] Epoch {e+1} | F1={m['f1']:.4f}")
        if m["f1"] > cnn_best_f1:
            cnn_best_f1 = m["f1"]
            torch.save(cnn.state_dict(), "cnn_best.pt")
        early.step(m["f1"])
        if early.stop:
            print("⏹ CNN early stopping")
            break

    # ---------------- TRAIN ViT ----------------
    vit = ViTClassifier().to(DEVICE)
    vit_opt = torch.optim.AdamW(vit.parameters(), lr=3e-5)
    early = EarlyStopping(PATIENCE)
    vit_best_f1 = 0.0

    for e in range(EPOCHS):
        train_epoch(vit, train_loader, vit_opt, criterion)
        _, m = validate(vit, val_loader, criterion)
        print(f"[ViT] Epoch {e+1} | F1={m['f1']:.4f}")
        if m["f1"] > vit_best_f1:
            vit_best_f1 = m["f1"]
            torch.save(vit.state_dict(), "vit_best.pt")
        early.step(m["f1"])
        if early.stop:
            print("⏹ ViT early stopping")
            break

    # ---------------- ENSEMBLE ----------------
    total_f1 = cnn_best_f1 + vit_best_f1
    w_cnn = cnn_best_f1 / total_f1
    w_vit = vit_best_f1 / total_f1

    print(f"\nAdaptive Weights → CNN={w_cnn:.3f}, ViT={w_vit:.3f}")

    @torch.no_grad()
    def ensemble_predict():
        cnn.eval()
        vit.eval()
        y_true, y_pred = [], []
        for x, y in val_loader:
            x = x.to(DEVICE)
            p = w_cnn * torch.softmax(cnn(x), 1) + \
                w_vit * torch.softmax(vit(x), 1)
            preds = torch.argmax(p, 1)
            y_pred.extend(preds.cpu().numpy())
            y_true.extend(y.numpy())
        return compute_metrics(y_true, y_pred)

    print("\nENSEMBLE METRICS:", ensemble_predict())
