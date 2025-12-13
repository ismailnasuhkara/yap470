import pandas as pd
import torch
from torch.utils.data import TensorDataset, DataLoader, random_split
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import precision_score, recall_score, f1_score, confusion_matrix

class NeuralNetwork:
    def __init__(self):
        self.model = nn.Sequential(
            nn.Linear(22, 32),
            nn.ReLU(),
            nn.Linear(32, 16),
            nn.ReLU(),
            nn.Linear(16, 1)
        )

        self.dataset_path = "../dataset/diabetes_binary_5050split_health_indicators_BRFSS2023.csv"
    
    def load_data(self):
        dataset = pd.read_csv(self.dataset_path).values.astype("float32")

        X = torch.tensor(dataset[:, 1:])
        y = torch.tensor(dataset[:, 0]).unsqueeze(1)

        num_samples = len(X)
        indices = torch.randperm(num_samples)

        train_size = int(0.8 * num_samples)
        train_idx = indices[:train_size]
        test_idx  = indices[train_size:]

        X_train, y_train = X[train_idx], y[train_idx]
        X_test,  y_test  = X[test_idx],  y[test_idx]

        X_mean = X_train.mean(dim=0, keepdim=True)
        X_std  = X_train.std(dim=0, keepdim=True)

        X_train = (X_train - X_mean) / X_std
        X_test  = (X_test - X_mean) / X_std

        val_size = int(0.2 * train_size)
        train_size = train_size - val_size

        full_train_ds = TensorDataset(X_train, y_train)
        train_ds, val_ds = random_split(full_train_ds, [train_size, val_size])

        test_ds = TensorDataset(X_test, y_test)

        self.train_loader = DataLoader(train_ds, batch_size=32, shuffle=True)
        self.val_loader   = DataLoader(val_ds,   batch_size=32)
        self.test_loader  = DataLoader(test_ds,  batch_size=32)

    def train(self, epochs=5, learning_rate=0.001, patience=5):
        criterion = nn.BCEWithLogitsLoss()
        optimizer = optim.Adam(self.model.parameters(), lr=learning_rate)

        best_val_loss = float('inf')
        wait = 0
        best_model_state = None

        for epoch in range(epochs):
            self.model.train()
            total_loss = 0.0
            correct = 0
            total = 0

            for X_batch, y_batch in self.train_loader:
                optimizer.zero_grad()
                logits = self.model(X_batch)
                loss = criterion(logits, y_batch)
                loss.backward()
                optimizer.step()

                total_loss += loss.item() * X_batch.size(0)

                preds = (torch.sigmoid(logits) >= 0.5).float()
                correct += (preds == y_batch).sum().item()
                total += y_batch.size(0)

            train_acc = correct / total * 100
            avg_loss = total_loss / total

            self.model.eval()
            val_loss = 0.0
            with torch.no_grad():
                for X_batch, y_batch in self.val_loader:
                    logits = self.model(X_batch)
                    loss = criterion(logits, y_batch)
                    val_loss += loss.item() * X_batch.size(0)

            val_loss /= len(self.val_loader.dataset)

            print(f"Epoch {epoch+1}/{epochs} | Train Loss: {avg_loss:.4f} | "
                  f"Train Acc: {train_acc:.2f}% | Val Loss: {val_loss:.4f}")

            if val_loss < best_val_loss:
                best_val_loss = val_loss
                wait = 0
                best_model_state = self.model.state_dict()
            else:
                wait += 1
                if wait >= patience:
                    print("Early stopping triggered.")
                    break

        if best_model_state is not None:
            self.model.load_state_dict(best_model_state)

    def evaluate(self):
        self.model.eval()
        all_preds = []
        all_labels = []

        with torch.no_grad():
            for X_batch, y_batch in self.test_loader:
                logits = self.model(X_batch)
                probs = torch.sigmoid(logits)
                preds = (probs >= 0.5).float()

                all_preds.append(preds)
                all_labels.append(y_batch)

        all_preds = torch.cat(all_preds).cpu().numpy()
        all_labels = torch.cat(all_labels).cpu().numpy()

        accuracy = (all_preds == all_labels).mean()
        precision = precision_score(all_labels, all_preds)
        recall = recall_score(all_labels, all_preds)
        f1 = f1_score(all_labels, all_preds)

        print("\n----- TEST METRICS -----")
        print("Accuracy :", accuracy)
        print("Precision:", precision)
        print("Recall   :", recall)
        print("F1 Score :", f1)

        cm = confusion_matrix(all_labels, all_preds)
        plt.figure(figsize=(5,4))
        sns.heatmap(cm, annot=True, fmt="d", cmap="Blues")
        plt.xlabel("Predicted")
        plt.ylabel("Actual")
        plt.title("Confusion Matrix")
        plt.show()