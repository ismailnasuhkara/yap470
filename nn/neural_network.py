import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader, random_split
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score,
    f1_score, roc_auc_score, confusion_matrix, roc_curve
)
from torch.optim.lr_scheduler import ReduceLROnPlateau
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from torch.utils.data import Subset



class NeuralNetwork:
    def __init__(self):
        self.model = None
        self.use_l1 = False
        self.l1_lambda = 0.0
        self.input_dim = None

    # ---------- MODEL ----------
    def build_model(self, input_dim, dropout=True, regularize=True):
        self.input_dim = input_dim
        self.use_l1 = regularize
        self.l1_lambda = 1e-5 if regularize else 0.0

        if dropout:
            self.model = nn.Sequential(
                nn.Linear(self.input_dim, 32),
                nn.ReLU(),
                nn.Dropout(0.2),
                nn.Linear(32, 16),
                nn.ReLU(),
                nn.Dropout(0.2),
                nn.Linear(16, 1)
            )
        else:
            self.model = nn.Sequential(
                nn.Linear(input_dim, 32),
                nn.ReLU(),
                nn.Linear(32, 16),
                nn.ReLU(),
                nn.Linear(16, 1)
            )

    # ---------- DATA ----------
    def load_data(self, path, batch_size=32):
        if self.model is None:
            raise ValueError("Model must be built before loading data.")

        df = pd.read_csv(path).dropna()
        data = df.values.astype("float32")

        X = data[:, :-1]
        y = data[:, -1].reshape(-1, 1)

        # safety check
        if X.shape[1] != self.input_dim:
            raise ValueError(
                f"Input dim mismatch: model expects {self.input_dim}, "
                f"data has {X.shape[1]}"
            )

        X = torch.tensor(X)
        y = torch.tensor(y)

        dataset = TensorDataset(X, y)

        n = len(dataset)
        train_size = int(0.7 * n)
        val_size   = int(0.15 * n)
        test_size  = n - train_size - val_size

        train_ds, val_ds, test_ds = random_split(
            dataset, [train_size, val_size, test_size]
        )

        self.train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
        self.val_loader   = DataLoader(val_ds,   batch_size=batch_size)
        self.test_loader  = DataLoader(test_ds,  batch_size=batch_size)

    # ---------- TRAIN ----------
    def train(self, epochs=30):
        if self.model is None:
            raise ValueError("Model not built.")

        criterion = nn.BCEWithLogitsLoss()
        optimizer = optim.Adam(self.model.parameters(), lr=3e-4)

        scheduler = ReduceLROnPlateau(
            optimizer,
            mode="max",
            factor=0.5,
            patience=3
        )

        train_acc_hist, val_acc_hist = [], []
        train_f1_hist,  val_f1_hist  = [], []

        for epoch in range(epochs):
            # ----- training -----
            self.model.train()
            correct, total = 0, 0
            train_preds, train_labels = [], []

            for Xb, yb in self.train_loader:
                optimizer.zero_grad()
                logits = self.model(Xb)
                loss = criterion(logits, yb)

                if self.use_l1:
                    l1_penalty = sum(p.abs().sum() for p in self.model.parameters())
                    loss = loss + self.l1_lambda * l1_penalty

                loss.backward()
                optimizer.step()

                preds = (torch.sigmoid(logits) >= 0.5).float()
                correct += (preds == yb).sum().item()
                total += yb.size(0)

                train_preds.append(preds)
                train_labels.append(yb)

            train_acc = correct / total
            train_f1 = f1_score(
                torch.cat(train_labels).cpu().numpy(),
                torch.cat(train_preds).cpu().numpy(),
                zero_division=0
            )

            # ----- validation -----
            self.model.eval()
            correct, total = 0, 0
            val_preds, val_labels = [], []

            with torch.no_grad():
                for Xb, yb in self.val_loader:
                    logits = self.model(Xb)
                    preds = (torch.sigmoid(logits) >= 0.5).float()

                    correct += (preds == yb).sum().item()
                    total += yb.size(0)

                    val_preds.append(preds)
                    val_labels.append(yb)

            val_acc = correct / total
            val_f1 = f1_score(
                torch.cat(val_labels).cpu().numpy(),
                torch.cat(val_preds).cpu().numpy(),
                zero_division=0
            )

            scheduler.step(val_f1)

            train_acc_hist.append(train_acc)
            val_acc_hist.append(val_acc)
            train_f1_hist.append(train_f1)
            val_f1_hist.append(val_f1)

            print(
                f"Epoch {epoch+1:02d}/{epochs} | "
                f"Train Acc: {train_acc*100:.2f}% | "
                f"Val Acc: {val_acc*100:.2f}% | "
                f"Train F1: {train_f1:.4f} | "
                f"Val F1: {val_f1:.4f}"
            )

        # ----- learning curves -----
        epochs_range = range(1, epochs + 1)

        plt.figure(figsize=(12, 5))

        plt.subplot(1, 2, 1)
        plt.plot(epochs_range, train_acc_hist, label="Train Accuracy")
        plt.plot(epochs_range, val_acc_hist, label="Validation Accuracy")
        plt.legend()
        plt.grid(alpha=0.3)

        plt.subplot(1, 2, 2)
        plt.plot(epochs_range, train_f1_hist, label="Train F1")
        plt.plot(epochs_range, val_f1_hist, label="Validation F1")
        plt.legend()
        plt.grid(alpha=0.3)

        plt.tight_layout()
        plt.show()

    # ---------- EVAL ----------
    def evaluate(self):
        if self.model is None:
            raise ValueError("Model not built.")

        self.model.eval()
        probs, preds, labels = [], [], []

        with torch.no_grad():
            for Xb, yb in self.test_loader:
                logits = self.model(Xb)
                prob = torch.sigmoid(logits)

                probs.append(prob)
                preds.append((prob >= 0.5).float())
                labels.append(yb)

        probs  = torch.cat(probs).cpu().numpy()
        preds  = torch.cat(preds).cpu().numpy()
        labels = torch.cat(labels).cpu().numpy()

        acc  = accuracy_score(labels, preds)
        prec = precision_score(labels, preds, zero_division=0)
        rec  = recall_score(labels, preds, zero_division=0)
        f1   = f1_score(labels, preds, zero_division=0)
        auc  = roc_auc_score(labels, probs)

        tn, fp, fn, tp = confusion_matrix(labels, preds).ravel()

        print(f"Accuracy: {acc:.4f}")
        print(f"Precision: {prec:.4f}")
        print(f"Recall: {rec:.4f}")
        print(f"F1: {f1:.4f}")
        print(f"ROC-AUC: {auc:.4f}")
        print(f"TP={tp}, FP={fp}, FN={fn}, TN={tn}")
        
        fpr, tpr, _ = roc_curve(labels, probs)
        plt.figure()
        plt.plot(fpr, tpr, label=f"ROC curve (AUC = {auc:.4f})")
        plt.plot([0, 1], [0, 1], linestyle="--")
        plt.xlabel("False Positive Rate")
        plt.ylabel("True Positive Rate")
        plt.title("ROC Curve")
        plt.legend(loc="lower right")
        plt.grid(True)
        plt.show()

