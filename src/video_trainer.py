"""
video_trainer.py

Training utilities for the video-only classifier.
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

    counter = 0

    for epoch in range(epochs):

        ############################
        # TRAIN
        ############################

        model.train()

        train_loss = 0
        train_correct = 0
        train_total = 0

        for videos, labels in train_loader:

            videos = videos.to(device)
            labels = labels.to(device)

            optimizer.zero_grad()

            outputs = model(videos)

            loss = criterion(outputs, labels)

            loss.backward()

            optimizer.step()

            train_loss += loss.item()

            preds = outputs.argmax(1)

            train_correct += (preds == labels).sum().item()

            train_total += labels.size(0)

        train_loss /= len(train_loader)
        train_acc = train_correct / train_total

        ############################
        # VALIDATION
        ############################

        model.eval()

        val_loss = 0
        val_correct = 0
        val_total = 0

        with torch.no_grad():

            for videos, labels in val_loader:

                videos = videos.to(device)
                labels = labels.to(device)

                outputs = model(videos)

                loss = criterion(outputs, labels)

                val_loss += loss.item()

                preds = outputs.argmax(1)

                val_correct += (preds == labels).sum().item()

                val_total += labels.size(0)

        val_loss /= len(val_loader)
        val_acc = val_correct / val_total

        history["train_loss"].append(train_loss)
        history["train_acc"].append(train_acc)
        history["val_loss"].append(val_loss)
        history["val_acc"].append(val_acc)

        print(
            f"Epoch {epoch+1}/{epochs} | "
            f"Train Loss {train_loss:.4f} | "
            f"Train Acc {train_acc:.4f} | "
            f"Val Loss {val_loss:.4f} | "
            f"Val Acc {val_acc:.4f}"
        )

        if val_acc > best_acc:

            best_acc = val_acc
            best_weights = copy.deepcopy(model.state_dict())
            counter = 0

        else:

            counter += 1

        if counter >= patience:

            print("\nEarly stopping!")
            print(f"Best Val Acc: {best_acc:.4f}")

            break

    model.load_state_dict(best_weights)

    return history