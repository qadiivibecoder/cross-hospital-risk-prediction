"""
=====================================================
Neural Network Model for Healthcare Risk Prediction
=====================================================
PyTorch neural network with configurable architecture.
Supports different input sizes for each dataset type.
=====================================================
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import numpy as np
from sklearn.metrics import (accuracy_score, precision_score, recall_score,
                             f1_score, roc_auc_score, confusion_matrix)
import time


class HealthcareRiskModel(nn.Module):
    """
    Neural network for binary classification of health risks.
    Architecture: Input -> 128 -> 64 -> 32 -> 1 (with dropout & batch norm)
    """
    def __init__(self, input_dim):
        super(HealthcareRiskModel, self).__init__()
        self.network = nn.Sequential(
            nn.Linear(input_dim, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(128, 64),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(32, 1),
            nn.Sigmoid()
        )

    def forward(self, x):
        return self.network(x).squeeze()


def get_input_dim(dataset_type):
    """Return the number of input features for each dataset type."""
    dims = {'heart': 13, 'diabetes': 8, 'kidney': 19}
    return dims.get(dataset_type, 13)


def train_local_model(model, X_train, y_train, X_test, y_test,
                      epochs=5, lr=0.001, batch_size=32):
    """
    Train a model on local hospital data.
    Returns training history with metrics for each epoch.
    """
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = model.to(device)

    # Create data loaders
    train_dataset = TensorDataset(
        torch.FloatTensor(X_train).to(device),
        torch.FloatTensor(y_train).to(device)
    )
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

    X_test_t = torch.FloatTensor(X_test).to(device)
    y_test_t = torch.FloatTensor(y_test).to(device)

    criterion = nn.BCELoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.9)

    history = {'accuracy': [], 'loss': [], 'val_accuracy': [], 'val_loss': [],
               'precision': [], 'recall': [], 'f1': [], 'auc': []}

    start_time = time.time()

    for epoch in range(epochs):
        # Training phase
        model.train()
        epoch_loss = 0
        for batch_X, batch_y in train_loader:
            optimizer.zero_grad()
            outputs = model(batch_X)
            loss = criterion(outputs, batch_y)
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()
        
        avg_loss = epoch_loss / len(train_loader)
        scheduler.step()

        # Evaluation phase
        model.eval()
        with torch.no_grad():
            val_outputs = model(X_test_t)
            val_loss = criterion(val_outputs, y_test_t).item()
            predictions = (val_outputs.cpu().numpy() > 0.5).astype(int)
            y_true = y_test_t.cpu().numpy().astype(int)
            probs = val_outputs.cpu().numpy()

        acc = accuracy_score(y_true, predictions)
        prec = precision_score(y_true, predictions, zero_division=0)
        rec = recall_score(y_true, predictions, zero_division=0)
        f1 = f1_score(y_true, predictions, zero_division=0)
        try:
            auc = roc_auc_score(y_true, probs)
        except ValueError:
            auc = 0.5

        history['accuracy'].append(acc)
        history['loss'].append(avg_loss)
        history['val_accuracy'].append(acc)
        history['val_loss'].append(val_loss)
        history['precision'].append(prec)
        history['recall'].append(rec)
        history['f1'].append(f1)
        history['auc'].append(auc)

        print(f"   Epoch {epoch+1}/{epochs} | Loss: {avg_loss:.4f} | "
              f"Acc: {acc:.4f} | F1: {f1:.4f} | AUC: {auc:.4f}")

    training_time = time.time() - start_time
    cm = confusion_matrix(y_true, predictions)

    final_metrics = {
        'accuracy': history['accuracy'][-1],
        'loss': history['loss'][-1],
        'precision': history['precision'][-1],
        'recall': history['recall'][-1],
        'f1': history['f1'][-1],
        'auc': history['auc'][-1],
        'confusion_matrix': cm.tolist(),
        'training_time': training_time,
        'history': history
    }
    return model, final_metrics


def get_model_weights(model):
    """Extract model weights as a dictionary of numpy arrays."""
    return {k: v.cpu().numpy().copy() for k, v in model.state_dict().items()}


def set_model_weights(model, weights):
    """Set model weights from a dictionary of numpy arrays."""
    state_dict = {k: torch.tensor(v) for k, v in weights.items()}
    model.load_state_dict(state_dict)
    return model


def predict_risk(model, features, device='cpu'):
    """Make a risk prediction for a single patient."""
    model.eval()
    model = model.to(device)
    with torch.no_grad():
        x = torch.FloatTensor(features).unsqueeze(0).to(device)
        prob = model(x).item()
    if prob < 0.3:
        level = "low"
    elif prob < 0.7:
        level = "medium"
    else:
        level = "high"
    return {'risk_score': round(prob, 4), 'risk_level': level, 'confidence': round(abs(prob - 0.5) * 2, 4)}
