import pandas as pd
import torch
from torch.utils.data import TensorDataset, DataLoader, random_split
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix, roc_auc_score, roc_curve
from torch.optim.lr_scheduler import ReduceLROnPlateau, StepLR

class NeuralNetwork:
    def __init__(self):
        self.model = nn.Sequential(
            nn.Linear(20, 32),
            nn.Dropout(0.3),
            nn.ReLU(),
            nn.Linear(32, 16),
            nn.Dropout(0.3),
            nn.ReLU(),
            nn.Linear(16, 16),
            nn.Dropout(0.3),
            nn.ReLU(),
            nn.Linear(16, 1)
        )
    
    def load_data(self, path):
        dataset = pd.read_csv(path).values.astype("float32")

        X = torch.tensor(dataset[:, 1:-2])
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
        optimizer = optim.Adam(self.model.parameters(), lr=learning_rate, weight_decay=0.0001)

        scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=3)


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

            scheduler.step(val_loss)

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

    def evaluate(self, thresholds=[0.3, 0.4, 0.5, 0.6, 0.7]):
        self.model.eval()
        all_probs = []
        all_labels = []

        with torch.no_grad():
            for X_batch, y_batch in self.test_loader:
                logits = self.model(X_batch)
                probs = torch.sigmoid(logits)

                all_probs.append(probs)
                all_labels.append(y_batch)

        all_probs = torch.cat(all_probs).cpu().numpy().ravel()
        all_labels = torch.cat(all_labels).cpu().numpy().ravel()

        results = []
        for threshold in thresholds:
            preds = (all_probs >= threshold).astype(int)

            accuracy = accuracy_score(all_labels, preds)
            precision = precision_score(all_labels, preds, zero_division=0)
            recall = recall_score(all_labels, preds, zero_division=0)
            f1 = f1_score(all_labels, preds, zero_division=0)

            tn, fp, fn, tp = confusion_matrix(all_labels, preds).ravel()
            specificity = tn / (tn + fp) if (tn + fp) > 0 else 0
            
            results.append({
                'Threshold': threshold,
                'Accuracy': accuracy,
                'Precision': precision,
                'Recall': recall,
                'F1-Score': f1,
                'Specificity': specificity,
                'TP': tp,
                'TN': tn,
                'FP': fp,
                'FN': fn
            })
        
        results_df = pd.DataFrame(results)
        
        roc_auc = roc_auc_score(all_labels, all_probs)

        print("\n" + "="*80)
        print("EVALUATION RESULTS ON TEST SET")
        print("="*80)
        print(f"\nROC-AUC Score: {roc_auc:.4f}\n")
        print(results_df.to_string(index=False))
        print("="*80 + "\n")
        
        return results_df, all_probs, all_labels, roc_auc
    
    def plot_threshold_analysis(self, results_df, all_probs=None, all_labels=None):
        plot_roc = all_probs is not None and all_labels is not None
        
        if plot_roc:
            fig, axes = plt.subplots(2, 3, figsize=(18, 10))
        else:
            fig, axes = plt.subplots(2, 2, figsize=(14, 10))
            axes = axes.flatten()
        
        # Plot 1: Main metrics
        ax1 = axes[0, 0] if plot_roc else axes[0]
        metrics = ['Accuracy', 'Precision', 'Recall', 'F1-Score']
        for metric in metrics:
            ax1.plot(results_df['Threshold'], results_df[metric], marker='o', label=metric)
        ax1.set_xlabel('Threshold')
        ax1.set_ylabel('Score')
        ax1.set_title('Performance Metrics vs Threshold')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        # Plot 2: Precision-Recall tradeoff
        ax2 = axes[0, 1] if plot_roc else axes[1]
        ax2.plot(results_df['Threshold'], results_df['Precision'], marker='o', label='Precision', color='blue')
        ax2.plot(results_df['Threshold'], results_df['Recall'], marker='s', label='Recall', color='red')
        ax2.plot(results_df['Threshold'], results_df['Specificity'], marker='^', label='Specificity', color='green')
        ax2.set_xlabel('Threshold')
        ax2.set_ylabel('Score')
        ax2.set_title('Precision-Recall-Specificity Tradeoff')
        ax2.legend()
        ax2.grid(True, alpha=0.3)
        
        # Plot 3: Confusion matrix components
        ax3 = axes[1, 0] if plot_roc else axes[2]
        width = 0.02
        x = results_df['Threshold']
        ax3.bar(x - 1.5*width, results_df['TP'], width, label='True Positive', color='green')
        ax3.bar(x - 0.5*width, results_df['TN'], width, label='True Negative', color='lightgreen')
        ax3.bar(x + 0.5*width, results_df['FP'], width, label='False Positive', color='orange')
        ax3.bar(x + 1.5*width, results_df['FN'], width, label='False Negative', color='red')
        ax3.set_xlabel('Threshold')
        ax3.set_ylabel('Count')
        ax3.set_title('Confusion Matrix Components')
        ax3.legend()
        ax3.grid(True, alpha=0.3, axis='y')
        
        # Plot 4: F1-Score highlighting best threshold
        ax4 = axes[1, 1] if plot_roc else axes[3]
        ax4.plot(results_df['Threshold'], results_df['F1-Score'], marker='o', linewidth=2, color='purple')
        best_idx = results_df['F1-Score'].idxmax()
        best_threshold = results_df.loc[best_idx, 'Threshold']
        best_f1 = results_df.loc[best_idx, 'F1-Score']
        ax4.scatter([best_threshold], [best_f1], color='red', s=200, zorder=5, label=f'Best: {best_threshold}')
        ax4.set_xlabel('Threshold')
        ax4.set_ylabel('F1-Score')
        ax4.set_title('F1-Score vs Threshold (Best Threshold Highlighted)')
        ax4.legend()
        ax4.grid(True, alpha=0.3)
        
        # Plot 5: ROC Curve
        if plot_roc:
            ax5 = axes[0, 2]
            fpr, tpr, thresholds_roc = roc_curve(all_labels, all_probs)
            roc_auc = roc_auc_score(all_labels, all_probs)
            
            ax5.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (AUC = {roc_auc:.4f})')
            ax5.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--', label='Random Classifier')
            ax5.set_xlim([0.0, 1.0])
            ax5.set_ylim([0.0, 1.05])
            ax5.set_xlabel('False Positive Rate')
            ax5.set_ylabel('True Positive Rate')
            ax5.set_title('ROC Curve')
            ax5.legend(loc="lower right")
            ax5.grid(True, alpha=0.3)
            
            # Plot 6: Threshold vs TPR/FPR
            ax6 = axes[1, 2]
            step = max(1, len(thresholds_roc) // 50)
            ax6.plot(thresholds_roc[::step], tpr[::step], marker='o', label='True Positive Rate', markersize=4)
            ax6.plot(thresholds_roc[::step], fpr[::step], marker='s', label='False Positive Rate', markersize=4)
            ax6.set_xlabel('Threshold')
            ax6.set_ylabel('Rate')
            ax6.set_title('TPR and FPR vs Threshold')
            ax6.legend()
            ax6.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.show()