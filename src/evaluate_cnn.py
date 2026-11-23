
import json
from pathlib import Path

import numpy as np
import torch
import matplotlib.pyplot as plt

from sklearn.metrics import (
    precision_recall_curve,
    confusion_matrix,
    classification_report,
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    average_precision_score,
    ConfusionMatrixDisplay
)

from src.preprocessing import load_data, preprocess_data
from src.cnn_model import FraudDetectionCNN


PROJECT_ROOT = Path(__file__).resolve().parents[1]
MODELS_PATH = PROJECT_ROOT / "models"
METRICS_PATH = PROJECT_ROOT / "results" / "metrics"
FIGURES_PATH = PROJECT_ROOT / "results" / "figures"


def select_best_threshold(y_true, probabilities):
    precision, recall, thresholds = precision_recall_curve(
        y_true,
        probabilities
    )

    f1_scores = (
        2 * precision[:-1] * recall[:-1]
        / (precision[:-1] + recall[:-1] + 1e-10)
    )

    best_index = np.argmax(f1_scores)

    return float(thresholds[best_index]), float(f1_scores[best_index])


def main():
    device = torch.device(
        "cuda" if torch.cuda.is_available() else "cpu"
    )

    print("Loading dataset...")

    df = load_data()

    (
        X_train, X_val, X_test,
        y_train, y_val, y_test,
        scaler
    ) = preprocess_data(df)

    X_val = torch.as_tensor(
        np.asarray(X_val),
        dtype=torch.float32
    ).to(device)

    X_test = torch.as_tensor(
        np.asarray(X_test),
        dtype=torch.float32
    ).to(device)

    y_val = np.asarray(y_val).reshape(-1)
    y_test = np.asarray(y_test).reshape(-1)

    checkpoint_path = MODELS_PATH / "cnn_best_model.pth"

    if not checkpoint_path.exists():
        raise FileNotFoundError(
            "CNN checkpoint not found. Run train_cnn.py first."
        )

    print("\nLoading CNN checkpoint...")

    checkpoint = torch.load(
        checkpoint_path,
        map_location=device,
        weights_only=True
    )

    model = FraudDetectionCNN(
        input_size=checkpoint["input_size"],
        dropout_rate=0.3
    ).to(device)

    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()

    print("Best checkpoint epoch:", checkpoint["epoch"])

    with torch.no_grad():
        val_logits = model(X_val).squeeze(1)
        test_logits = model(X_test).squeeze(1)

        val_probabilities = torch.sigmoid(
            val_logits
        ).cpu().numpy()

        test_probabilities = torch.sigmoid(
            test_logits
        ).cpu().numpy()

    threshold, validation_f1 = select_best_threshold(
        y_val,
        val_probabilities
    )

    print("\nValidation threshold selection:")
    print("Best threshold:", threshold)
    print("Validation F1:", validation_f1)

    predictions = (
        test_probabilities >= threshold
    ).astype(int)

    metrics = {
        "model": "1D-CNN",
        "best_epoch": int(checkpoint["epoch"]),
        "threshold": threshold,
        "validation_f1": validation_f1,
        "accuracy": float(
            accuracy_score(y_test, predictions)
        ),
        "precision": float(
            precision_score(y_test, predictions, zero_division=0)
        ),
        "recall": float(
            recall_score(y_test, predictions, zero_division=0)
        ),
        "f1_score": float(
            f1_score(y_test, predictions, zero_division=0)
        ),
        "roc_auc": float(
            roc_auc_score(y_test, test_probabilities)
        ),
        "pr_auc": float(
            average_precision_score(y_test, test_probabilities)
        ),
        "confusion_matrix": confusion_matrix(
            y_test, predictions
        ).tolist()
    }

    print("\nCNN TEST RESULTS")

    for key, value in metrics.items():
        if key != "confusion_matrix":
            print(f"{key}: {value}")

    print("\nConfusion matrix:")
    print(confusion_matrix(y_test, predictions))

    print("\nClassification report:")
    print(
        classification_report(
            y_test,
            predictions,
            target_names=["Legitimate", "Fraud"],
            zero_division=0
        )
    )

    with open(
        METRICS_PATH / "cnn_results.json",
        "w"
    ) as file:
        json.dump(metrics, file, indent=4)

    ConfusionMatrixDisplay.from_predictions(
        y_test,
        predictions,
        display_labels=["Legitimate", "Fraud"],
        cmap="Blues"
    )

    plt.title("1D-CNN Confusion Matrix")
    plt.tight_layout()
    plt.savefig(
        FIGURES_PATH / "cnn_confusion_matrix.png",
        dpi=300
    )
    plt.close()

    precision, recall, _ = precision_recall_curve(
        y_test,
        test_probabilities
    )

    plt.figure(figsize=(8, 5))
    plt.plot(recall, precision)
    plt.xlabel("Recall")
    plt.ylabel("Precision")
    plt.title("1D-CNN Precision-Recall Curve")
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(
        FIGURES_PATH / "cnn_precision_recall_curve.png",
        dpi=300
    )
    plt.close()

    print("\nCNN evaluation completed.")
    print("Results saved:", METRICS_PATH / "cnn_results.json")


if __name__ == "__main__":
    main()
