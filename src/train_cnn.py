
import json
import random
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
import matplotlib.pyplot as plt

from src.preprocessing import load_data, preprocess_data
from src.cnn_model import FraudDetectionCNN


PROJECT_ROOT = Path(__file__).resolve().parents[1]
MODELS_PATH = PROJECT_ROOT / "models"
METRICS_PATH = PROJECT_ROOT / "results" / "metrics"
FIGURES_PATH = PROJECT_ROOT / "results" / "figures"

BATCH_SIZE = 512
LEARNING_RATE = 0.001
MAX_EPOCHS = 50
PATIENCE = 7
RANDOM_STATE = 42


def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def to_tensor(data, dtype):
    return torch.as_tensor(
        np.asarray(data, copy=True),
        dtype=dtype
    )


def main():
    set_seed(RANDOM_STATE)

    MODELS_PATH.mkdir(parents=True, exist_ok=True)
    METRICS_PATH.mkdir(parents=True, exist_ok=True)
    FIGURES_PATH.mkdir(parents=True, exist_ok=True)

    device = torch.device(
        "cuda" if torch.cuda.is_available() else "cpu"
    )

    print("Using device:", device)
    print("\nLoading dataset...")

    df = load_data()

    (
        X_train, X_val, X_test,
        y_train, y_val, y_test,
        scaler
    ) = preprocess_data(df)

    X_train = to_tensor(X_train, torch.float32)
    X_val = to_tensor(X_val, torch.float32)

    y_train = to_tensor(y_train, torch.float32).reshape(-1, 1)
    y_val = to_tensor(y_val, torch.float32).reshape(-1, 1)

    print("\nDataset split:")
    print("Training:", len(X_train))
    print("Validation:", len(X_val))

    train_dataset = TensorDataset(X_train, y_train)

    train_loader = DataLoader(
        train_dataset,
        batch_size=BATCH_SIZE,
        shuffle=True
    )

    model = FraudDetectionCNN(
        input_size=X_train.shape[1],
        dropout_rate=0.3
    ).to(device)

    positive_count = y_train.sum().item()
    negative_count = len(y_train) - positive_count

    positive_weight = negative_count / positive_count

    print("\nPositive class weight:", positive_weight)

    criterion = nn.BCEWithLogitsLoss(
        pos_weight=torch.tensor(
            [positive_weight],
            dtype=torch.float32,
            device=device
        )
    )

    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=LEARNING_RATE,
        weight_decay=1e-5
    )

    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode="min",
        factor=0.5,
        patience=3
    )

    best_val_loss = float("inf")
    best_epoch = 0
    patience_counter = 0

    history = {
        "train_loss": [],
        "val_loss": []
    }

    checkpoint_path = MODELS_PATH / "cnn_best_model.pth"

    for epoch in range(MAX_EPOCHS):

        model.train()
        train_loss = 0.0

        for inputs, labels in train_loader:
            inputs = inputs.to(device)
            labels = labels.to(device)

            optimizer.zero_grad()

            outputs = model(inputs)
            loss = criterion(outputs, labels)

            loss.backward()
            optimizer.step()

            train_loss += loss.item() * inputs.size(0)

        train_loss /= len(train_loader.dataset)

        model.eval()
        val_loss = 0.0

        with torch.no_grad():
            val_inputs = X_val.to(device)
            val_labels = y_val.to(device)

            val_outputs = model(val_inputs)
            loss = criterion(val_outputs, val_labels)

            val_loss = loss.item()

        scheduler.step(val_loss)

        history["train_loss"].append(train_loss)
        history["val_loss"].append(val_loss)

        print(
            f"Epoch [{epoch + 1}/{MAX_EPOCHS}] "
            f"Train Loss: {train_loss:.5f} "
            f"Validation Loss: {val_loss:.5f}"
        )

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_epoch = epoch + 1
            patience_counter = 0

            torch.save(
                {
                    "model_state_dict": model.state_dict(),
                    "epoch": best_epoch,
                    "best_val_loss": best_val_loss,
                    "input_size": X_train.shape[1]
                },
                checkpoint_path
            )

        else:
            patience_counter += 1

        if patience_counter >= PATIENCE:
            print("\nEarly stopping triggered.")
            break

    history_path = METRICS_PATH / "cnn_training_history.json"

    with open(history_path, "w") as file:
        json.dump(history, file, indent=4)

    plt.figure(figsize=(9, 5))
    plt.plot(history["train_loss"], label="Training Loss")
    plt.plot(history["val_loss"], label="Validation Loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title("1D-CNN Training and Validation Loss")
    plt.legend()
    plt.grid(alpha=0.3)
    plt.tight_layout()

    plt.savefig(
        FIGURES_PATH / "cnn_training_loss.png",
        dpi=300
    )
    plt.close()

    print("\nCNN training completed.")
    print("Best epoch:", best_epoch)
    print("Best validation loss:", best_val_loss)
    print("Checkpoint saved:", checkpoint_path)


if __name__ == "__main__":
    main()
