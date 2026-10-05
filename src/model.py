"""
A small CNN for classifying the synthetic steel-surface-defect images
from generate_data.py into one of four classes: none, scratch, pitting,
patches.

Real, trained PyTorch code with a full training loop. Runs on CPU.
"""

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.metrics import (
    accuracy_score,
    precision_recall_fscore_support,
    confusion_matrix,
)

from generate_data import CLASSES, IMG_SIZE

LABEL_TO_IDX = {label: i for i, label in enumerate(CLASSES)}
IDX_TO_LABEL = {i: label for label, i in LABEL_TO_IDX.items()}


class DefectCNN(nn.Module):
    """A small CNN: two conv+pool blocks, then two FC layers."""

    def __init__(self, num_classes=len(CLASSES)):
        super().__init__()
        self.conv1 = nn.Conv2d(1, 16, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(16, 32, kernel_size=3, padding=1)
        self.pool = nn.MaxPool2d(2, 2)
        # IMG_SIZE=64 -> after two 2x2 pools -> 16x16
        reduced = IMG_SIZE // 4
        self.fc1 = nn.Linear(32 * reduced * reduced, 64)
        self.fc2 = nn.Linear(64, num_classes)
        self.dropout = nn.Dropout(0.25)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = torch.flatten(x, 1)
        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.fc2(x)
        return x


def images_to_tensor(images):
    """(N, H, W) float64 array in [0,255] -> normalized (N, 1, H, W) float32 tensor."""
    arr = images.astype(np.float32) / 255.0
    arr = (arr - 0.5) / 0.5  # roughly [-1, 1]
    return torch.from_numpy(arr).unsqueeze(1)


def labels_to_tensor(labels):
    return torch.tensor([LABEL_TO_IDX[l] for l in labels], dtype=torch.long)


def train_model(train_images, train_labels, val_images, val_labels,
                 epochs=15, lr=1e-3, batch_size=16, seed=0, verbose=False):
    """Train DefectCNN on the given data. Returns (model, history) where
    history is a list of per-epoch dicts with train_loss, val_loss, val_accuracy."""
    torch.manual_seed(seed)
    model = DefectCNN()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    criterion = nn.CrossEntropyLoss()

    X_train = images_to_tensor(train_images)
    y_train = labels_to_tensor(train_labels)
    X_val = images_to_tensor(val_images)
    y_val = labels_to_tensor(val_labels)

    n = X_train.shape[0]
    history = []

    for epoch in range(epochs):
        model.train()
        perm = torch.randperm(n)
        epoch_loss = 0.0
        for start in range(0, n, batch_size):
            idx = perm[start:start + batch_size]
            xb, yb = X_train[idx], y_train[idx]
            optimizer.zero_grad()
            out = model(xb)
            loss = criterion(out, yb)
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item() * xb.shape[0]
        epoch_loss /= n

        model.eval()
        with torch.no_grad():
            val_out = model(X_val)
            val_loss = criterion(val_out, y_val).item()
            val_preds = val_out.argmax(dim=1)
            val_acc = (val_preds == y_val).float().mean().item()

        history.append({"epoch": epoch, "train_loss": epoch_loss,
                         "val_loss": val_loss, "val_accuracy": val_acc})
        if verbose:
            print(f"epoch {epoch}: train_loss={epoch_loss:.4f} "
                  f"val_loss={val_loss:.4f} val_acc={val_acc:.4f}")

    return model, history


def predict(model, images):
    """Return predicted label strings for a batch of raw images."""
    model.eval()
    X = images_to_tensor(images)
    with torch.no_grad():
        out = model(X)
        preds = out.argmax(dim=1).tolist()
    return [IDX_TO_LABEL[p] for p in preds]


def evaluate(model, images, labels):
    """Compute accuracy, per-class precision/recall/F1, and confusion matrix
    on the given labeled set. Returns a dict of results."""
    preds = predict(model, images)
    acc = accuracy_score(labels, preds)
    precision, recall, f1, support = precision_recall_fscore_support(
        labels, preds, labels=CLASSES, zero_division=0
    )
    cm = confusion_matrix(labels, preds, labels=CLASSES)
    per_class = {
        cls: {
            "precision": float(precision[i]),
            "recall": float(recall[i]),
            "f1": float(f1[i]),
            "support": int(support[i]),
        }
        for i, cls in enumerate(CLASSES)
    }
    return {
        "accuracy": float(acc),
        "per_class": per_class,
        "confusion_matrix": cm.tolist(),
        "labels_order": list(CLASSES),
    }
