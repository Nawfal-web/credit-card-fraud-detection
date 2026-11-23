
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

from src.preprocessing import load_data, preprocess_data
from src.model import FraudDetectionMLP


PROJECT_ROOT = Path(__file__).resolve().parents[1]
MODEL_PATH = PROJECT_ROOT / "models" / "best_model.pth"

DEVICE = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)


def create_data_loader(X, y, batch_size=512, shuffle=False):

    X_tensor = torch.tensor(
        X.to_numpy(dtype=np.float32),
        dtype=torch.float32
    )

    y_tensor = torch.tensor(
        y.to_numpy(dtype=np.float32).reshape(-1, 1),
        dtype=torch.float32
    )

    dataset = TensorDataset(X_tensor, y_tensor)

    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle
    )


def train_model(model, train_loader, val_loader, pos_weight,
                epochs=50, patience=7):

    criterion = nn.BCEWithLogitsLoss(
        pos_weight=pos_weight
    )

    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=0.001,
        weight_decay=1e-5
    )

    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode="min",
        factor=0.5,
        patience=3
    )

    best_val_loss = float("inf")
    patience_counter = 0

    history = {
        "train_loss": [],
        "val_loss": []
    }

    MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)

    for epoch in range(epochs):

        # Training phase
        model.train()
        train_loss = 0.0

        for X_batch, y_batch in train_loader:

            X_batch = X_batch.to(DEVICE)
            y_batch = y_batch.to(DEVICE)

            optimizer.zero_grad()

            outputs = model(X_batch)

            loss = criterion(outputs, y_batch)

            loss.backward()
            optimizer.step()

            train_loss += loss.item() * X_batch.size(0)

        train_loss /= len(train_loader.dataset)

        # Validation phase
        model.eval()
        val_loss = 0.0

        with torch.no_grad():

            for X_batch, y_batch in val_loader:

                X_batch = X_batch.to(DEVICE)
                y_batch = y_batch.to(DEVICE)

                outputs = model(X_batch)

                loss = criterion(outputs, y_batch)

                val_loss += loss.item() * X_batch.size(0)

        val_loss /= len(val_loader.dataset)

        history["train_loss"].append(train_loss)
        history["val_loss"].append(val_loss)

        scheduler.step(val_loss)

        print(
            f"Epoch [{epoch + 1}/{epochs}] "
            f"Train Loss: {train_loss:.5f} "
            f"Validation Loss: {val_loss:.5f}"
        )

        # Save the best model
        if val_loss < best_val_loss:

            best_val_loss = val_loss
            patience_counter = 0

            torch.save(
                {
                    "model_state_dict": model.state_dict(),
                    "best_val_loss": best_val_loss,
                    "epoch": epoch + 1
                },
                MODEL_PATH
            )

        else:
            patience_counter += 1

        if patience_counter >= patience:
            print("\nEarly stopping triggered.")
            break

    print(f"\nBest model saved to: {MODEL_PATH}")

    return history


def main():

    print("Using device:", DEVICE)

    df = load_data()

    (
        X_train,
        X_val,
        X_test,
        y_train,
        y_val,
        y_test,
        scaler
    ) = preprocess_data(df)

    train_loader = create_data_loader(
        X_train, y_train, shuffle=True
    )

    val_loader = create_data_loader(
        X_val, y_val
    )

    # Calculate positive-class weight from training data only
    negative_count = int((y_train == 0).sum())
    positive_count = int((y_train == 1).sum())

    positive_weight = negative_count / positive_count

    pos_weight = torch.tensor(
        [positive_weight],
        dtype=torch.float32,
        device=DEVICE
    )

    print("Positive class weight:", positive_weight)

    model = FraudDetectionMLP(
        input_size=X_train.shape[1],
        dropout_rate=0.3
    ).to(DEVICE)

    history = train_model(
        model,
        train_loader,
        val_loader,
        pos_weight
    )

    print("\nTraining completed successfully.")


if __name__ == "__main__":
    main()
