"""
fusion_trainer.py

Training utilities for the multimodal
LSTM + ResNet50 fusion model.
"""

import copy
import torch


def fit(
    model,
    train_loader,
    val_loader,
    criterion,
    optimizer,
    device,
    epochs=20,
    patience=3,
):

    history = {
        "train_loss": [],
        "train_acc": [],
        "val_loss": [],
        "val_acc": [],
    }

    best_acc = 0.0
    best_weights = copy.deepcopy(model.state_dict())

    early_stop_counter = 0

    for epoch in range(epochs):

        # =====================================================
        # TRAIN
        # =====================================================

        model.train()

        train_loss = 0
        train_correct = 0
        train_total = 0

        for sensor, video, labels in train_loader:

            sensor = sensor.to(device)
            video = video.to(device)
            labels = labels.to(device)

            optimizer.zero_grad()

            outputs = model(
                sensor,
                video,
            )

            loss = criterion(
                outputs,
                labels,
            )

            loss.backward()

            optimizer.step()

            train_loss += loss.item()

            predictions = outputs.argmax(dim=1)

            train_correct += (
                predictions == labels
            ).sum().item()

            train_total += labels.size(0)

        train_loss /= len(train_loader)

        train_acc = train_correct / train_total

        # =====================================================
        # VALIDATION
        # =====================================================

        model.eval()

        val_loss = 0
        val_correct = 0
        val_total = 0

        with torch.no_grad():

            for sensor, video, labels in val_loader:

                sensor = sensor.to(device)
                video = video.to(device)
                labels = labels.to(device)

                outputs = model(
                    sensor,
                    video,
                )

                loss = criterion(
                    outputs,
                    labels,
                )

                val_loss += loss.item()

                predictions = outputs.argmax(dim=1)

                val_correct += (
                    predictions == labels
                ).sum().item()

                val_total += labels.size(0)

        val_loss /= len(val_loader)

        val_acc = val_correct / val_total

        history["train_loss"].append(train_loss)
        history["train_acc"].append(train_acc)
        history["val_loss"].append(val_loss)
        history["val_acc"].append(val_acc)

        print(
            f"Epoch {epoch+1}/{epochs} | "
            f"Train Loss: {train_loss:.4f} | "
            f"Train Acc: {train_acc:.4f} | "
            f"Val Loss: {val_loss:.4f} | "
            f"Val Acc: {val_acc:.4f}"
        )

        # =====================================================
        # EARLY STOPPING
        # =====================================================

        if val_acc > best_acc:

            best_acc = val_acc

            best_weights = copy.deepcopy(
                model.state_dict()
            )

            early_stop_counter = 0

        else:

            early_stop_counter += 1

        if early_stop_counter >= patience:

            print()
            print("=" * 60)
            print(f"Early stopping at epoch {epoch+1}")
            print(f"Best Validation Accuracy: {best_acc:.4f}")
            print("=" * 60)

            break

    model.load_state_dict(best_weights)

    return history