"""
trainer.py

Training utilities for UAH Driver Risk Prediction.
"""

from xml.parsers.expat import model

import torch
import torch.nn as nn

from torch.utils.data import Dataset
from torch.utils.data import DataLoader

class UAHDataset(Dataset):

    def __init__(self, X, y):

        self.X = torch.FloatTensor(X)
        self.y = torch.LongTensor(y)

    def __len__(self):

        return len(self.X)

    def __getitem__(self, idx):

        return self.X[idx], self.y[idx]
    
def create_dataloader(
    X,
    y,
    batch_size=32,
    shuffle=True,
):
    """
    Create a PyTorch DataLoader.
    """

    dataset = UAHDataset(X, y)

    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        drop_last=False,
    )

    return loader

def train_one_epoch(
    model,
    dataloader,
    criterion,
    optimizer,
    device,
):
    """
    Train the model for one epoch.
    """

    model.train()

    total_loss = 0.0
    correct = 0
    total = 0

    for batch_x, batch_y in dataloader:

        batch_x = batch_x.to(device)
        batch_y = batch_y.to(device)

        optimizer.zero_grad()

        outputs = model(batch_x)

        loss = criterion(outputs, batch_y)

        loss.backward()
        
        torch.nn.utils.clip_grad_norm_(
            model.parameters(),
            max_norm=1.0,
        )
        optimizer.step()

        total_loss += loss.item()

        predictions = outputs.argmax(dim=1)

        correct += (predictions == batch_y).sum().item()

        total += batch_y.size(0)

    avg_loss = total_loss / len(dataloader)
    accuracy = correct / total

    return avg_loss, accuracy

def validate_one_epoch(
    model,
    dataloader,
    criterion,
    device,
):
    """
    Validate the model for one epoch.
    """

    model.eval()

    total_loss = 0.0
    correct = 0
    total = 0

    with torch.no_grad():

        for batch_x, batch_y in dataloader:

            batch_x = batch_x.to(device)
            batch_y = batch_y.to(device)

            outputs = model(batch_x)

            loss = criterion(outputs, batch_y)

            total_loss += loss.item()

            predictions = outputs.argmax(dim=1)

            correct += (predictions == batch_y).sum().item()

            total += batch_y.size(0)

    avg_loss = total_loss / len(dataloader)
    accuracy = correct / total

    return avg_loss, accuracy

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
)

def evaluate_model(
    model,
    dataloader,
    device,
):
    """
    Return predictions and labels for evaluation.
    """

    model.eval()

    all_predictions = []
    all_labels = []

    with torch.no_grad():

        for batch_x, batch_y in dataloader:

            batch_x = batch_x.to(device)

            outputs = model(batch_x)

            predictions = outputs.argmax(dim=1)

            all_predictions.extend(predictions.cpu().numpy())
            all_labels.extend(batch_y.numpy())

    return all_labels, all_predictions

def fit(
    model,
    train_loader,
    val_loader,
    criterion,
    optimizer,
    device,
    epochs,
    patience=3,
):
    """
    Train the model with Early Stopping.

    Returns
    -------
    history : dict
        Training history.
    """

    history = {
        "train_loss": [],
        "train_acc": [],
        "val_loss": [],
        "val_acc": [],
        "best_val_acc": 0.0,
    }

    best_val_acc = 0.0
    patience_counter = 0

    for epoch in range(epochs):

        train_loss, train_acc = train_one_epoch(
            model,
            train_loader,
            criterion,
            optimizer,
            device,
        )

        val_loss, val_acc = validate_one_epoch(
            model,
            val_loader,
            criterion,
            device,
        )

        history["train_loss"].append(train_loss)
        history["train_acc"].append(train_acc)
        history["val_loss"].append(val_loss)
        history["val_acc"].append(val_acc)

        if val_acc > best_val_acc:

            best_val_acc = val_acc
            history["best_val_acc"] = best_val_acc

            patience_counter = 0

            # Best model
            torch.save(
                model.state_dict(),
                "best_model.pth",
            )

        else:

            patience_counter += 1

        print(
            f"Epoch {epoch + 1}/{epochs} | "
            f"Train Loss: {train_loss:.4f} | "
            f"Train Acc: {train_acc:.4f} | "
            f"Val Loss: {val_loss:.4f} | "
            f"Val Acc: {val_acc:.4f}"
        )

        if patience_counter >= patience:

            print()
            print("=" * 60)
            print(f"Early stopping at epoch {epoch + 1}")
            print(f"Best Validation Accuracy : {best_val_acc:.4f}")
            print("=" * 60)
            break

    # Load best model before returning
    model.load_state_dict(torch.load("best_model.pth"))
    
    return history