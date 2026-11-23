
from pathlib import Path

import numpy as np
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler


# Project paths
PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_PATH = PROJECT_ROOT / "data" / "creditcard.csv"


def load_data():
    """Load the credit card transaction dataset."""
    if not DATA_PATH.exists():
        raise FileNotFoundError(
            f"Dataset not found: {DATA_PATH}"
        )

    df = pd.read_csv(DATA_PATH)

    print("\nDataset loaded successfully.")
    print(f"Dataset shape: {df.shape}")

    return df


def explore_data(df):
    """Display basic information about the dataset."""
    print("\nFirst five rows:")
    print(df.head())

    print("\nDataset information:")
    print(df.info())

    print("\nMissing values:")
    print(df.isnull().sum().sum())

    print("\nClass distribution:")
    print(df["Class"].value_counts())

    print("\nFraud percentage:")
    print(df["Class"].mean() * 100)

    return df


def preprocess_data(df):
    """
    Split the dataset and scale numerical features.

    The scaler is fitted only on training data to prevent
    information leakage.
    """
    X = df.drop(columns=["Class"]).copy()
    y = df["Class"].copy()

    # 70% training, 15% validation, 15% testing
    X_train, X_temp, y_train, y_temp = train_test_split(
        X,
        y,
        test_size=0.30,
        random_state=42,
        stratify=y
    )

    X_val, X_test, y_val, y_test = train_test_split(
        X_temp,
        y_temp,
        test_size=0.50,
        random_state=42,
        stratify=y_temp
    )

    # Scale Time and Amount only
    scaler = StandardScaler()

    columns_to_scale = ["Time", "Amount"]

    X_train = X_train.copy()
    X_val = X_val.copy()
    X_test = X_test.copy()

    X_train[columns_to_scale] = scaler.fit_transform(
        X_train[columns_to_scale]
    )

    X_val[columns_to_scale] = scaler.transform(
        X_val[columns_to_scale]
    )

    X_test[columns_to_scale] = scaler.transform(
        X_test[columns_to_scale]
    )

    # Convert to float32 for PyTorch compatibility
    X_train = X_train.astype(np.float32)
    X_val = X_val.astype(np.float32)
    X_test = X_test.astype(np.float32)

    print("\nData preprocessing completed.")
    print(f"Training samples: {len(X_train)}")
    print(f"Validation samples: {len(X_val)}")
    print(f"Testing samples: {len(X_test)}")

    print("\nFraud cases in each split:")
    print(f"Training: {int(y_train.sum())}")
    print(f"Validation: {int(y_val.sum())}")
    print(f"Testing: {int(y_test.sum())}")

    return (
        X_train,
        X_val,
        X_test,
        y_train,
        y_val,
        y_test,
        scaler
    )


if __name__ == "__main__":
    dataset = load_data()
    explore_data(dataset)

    splits = preprocess_data(dataset)
