
import json
from pathlib import Path

import numpy as np
import torch
import matplotlib.pyplot as plt

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    average_precision_score,
    confusion_matrix,
    classification_report,
    precision_recall_curve,
    ConfusionMatrixDisplay,
    PrecisionRecallDisplay,
    RocCurveDisplay
)

from src.preprocessing import load_data, preprocess_data
from src.model import FraudDetectionMLP


PROJECT_ROOT = Path(__file__).resolve().parents[1]

MODEL_PATH = PROJECT_ROOT / "models" / "best_model.pth"
RESULTS_PATH = PROJECT_ROOT / "results"
FIGURES_PATH = RESULTS_PATH / "figures"
METRICS_PATH = RESULTS_PATH / "metrics"

DEVICE = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)


def predict_probabilities(model, X):
    model.eval()

    X_tensor = torch.tensor(
        X.to_numpy(dtype=np.float32),
        dtype=torch.float32,
        device=DEVICE
    )

    with torch.no_grad():
        logits = model(X_tensor)
        probabilities = torch.sigmoid(logits)

    return probabilities.cpu().numpy().ravel()


def select_threshold(y_true, probabilities):
    precision, recall, thresholds = precision_recall_curve(
        y_true, probabilities
    )

    f1_scores = (
        2 * precision[:-1] * recall[:-1]
        / (precision[:-1] + recall[:-1] + 1e-12)
    )

    best_index = np.argmax(f1_scores)
    best_threshold = float(thresholds[best_index])

    print("\nValidation threshold selection:")
    print(f"Best threshold: {best_threshold:.6f}")
    print(f"Validation F1: {f1_scores[best_index]:.4f}")

    return best_threshold


def calculate_metrics(y_true, probabilities, threshold):

    predictions = (probabilities >= threshold).astype(int)

    return {
        "threshold": float(threshold),
        "accuracy": float(accuracy_score(y_true, predictions)),
        "precision": float(
            precision_score(y_true, predictions, zero_division=0)
        ),
        "recall": float(
            recall_score(y_true, predictions, zero_division=0)
        ),
        "f1_score": float(
            f1_score(y_true, predictions, zero_division=0)
        ),
        "roc_auc": float(
            roc_auc_score(y_true, probabilities)
        ),
        "pr_auc": float(
            average_precision_score(y_true, probabilities)
        ),
        "confusion_matrix": confusion_matrix(
            y_true, predictions
        ).tolist()
    }


def evaluate_model(y_true, probabilities, threshold):

    metrics = calculate_metrics(
        y_true, probabilities, threshold
    )

    predictions = (
        probabilities >= threshold
    ).astype(int)

    print("\nClassification metrics:")

    for name, value in metrics.items():
        if name != "confusion_matrix":
            print(f"{name}: {value:.4f}")

    print("\nConfusion matrix:")
    print(np.array(metrics["confusion_matrix"]))

    print("\nClassification report:")
    print(
        classification_report(
            y_true,
            predictions,
            target_names=["Legitimate", "Fraud"],
            zero_division=0
        )
    )

    return metrics, predictions


def analyze_thresholds(y_true, probabilities, selected_threshold):

    thresholds_to_test = [
        0.5,
        0.9,
        0.99,
        selected_threshold
    ]

    results = []

    print("\nVALIDATION THRESHOLD ANALYSIS")

    for threshold in thresholds_to_test:

        metrics = calculate_metrics(
            y_true,
            probabilities,
            threshold
        )

        results.append(metrics)

        print(
            f"Threshold: {threshold:.6f} | "
            f"Precision: {metrics['precision']:.4f} | "
            f"Recall: {metrics['recall']:.4f} | "
            f"F1: {metrics['f1_score']:.4f}"
        )

    return results


def create_visualizations(
    y_true,
    predictions,
    probabilities,
    val_probabilities,
    val_labels
):

    FIGURES_PATH.mkdir(parents=True, exist_ok=True)

    # Confusion matrix
    cm = confusion_matrix(y_true, predictions)

    display = ConfusionMatrixDisplay(
        confusion_matrix=cm,
        display_labels=["Legitimate", "Fraud"]
    )

    display.plot(cmap="Blues", values_format="d")
    plt.title("Credit Card Fraud Detection - Confusion Matrix")
    plt.tight_layout()

    plt.savefig(
        FIGURES_PATH / "confusion_matrix.png",
        dpi=300
    )
    plt.close()

    # Precision-Recall curve
    plt.figure(figsize=(8, 6))

    PrecisionRecallDisplay.from_predictions(
        y_true,
        probabilities
    )

    plt.title("Precision-Recall Curve")
    plt.tight_layout()

    plt.savefig(
        FIGURES_PATH / "precision_recall_curve.png",
        dpi=300
    )
    plt.close()

    # ROC curve
    plt.figure(figsize=(8, 6))

    RocCurveDisplay.from_predictions(
        y_true,
        probabilities
    )

    plt.title("ROC Curve")
    plt.tight_layout()

    plt.savefig(
        FIGURES_PATH / "roc_curve.png",
        dpi=300
    )
    plt.close()

    # Validation probability distributions
    plt.figure(figsize=(9, 6))

    plt.hist(
        val_probabilities[val_labels == 0],
        bins=50,
        alpha=0.6,
        label="Legitimate"
    )

    plt.hist(
        val_probabilities[val_labels == 1],
        bins=50,
        alpha=0.6,
        label="Fraud"
    )

    plt.xlabel("Predicted Fraud Probability")
    plt.ylabel("Number of Transactions")
    plt.title("Validation Prediction Probability Distribution")
    plt.legend()
    plt.tight_layout()

    plt.savefig(
        FIGURES_PATH / "validation_probability_distribution.png",
        dpi=300
    )
    plt.close()

    print("\nVisualizations saved successfully.")


def main():

    if not MODEL_PATH.exists():
        raise FileNotFoundError(
            "Trained model not found. Run src.train first."
        )

    print("Loading dataset...")

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

    print("\nLoading best model checkpoint...")

    model = FraudDetectionMLP(
        input_size=X_train.shape[1]
    ).to(DEVICE)

    checkpoint = torch.load(
        MODEL_PATH,
        map_location=DEVICE,
        weights_only=True
    )

    model.load_state_dict(checkpoint["model_state_dict"])

    print("Best checkpoint epoch:", checkpoint["epoch"])

    # Validation predictions
    val_probabilities = predict_probabilities(model, X_val)
    val_labels = y_val.to_numpy()

    threshold = select_threshold(
        val_labels,
        val_probabilities
    )

    # Analyze validation thresholds
    threshold_results = analyze_thresholds(
        val_labels,
        val_probabilities,
        threshold
    )

    # Validation probability statistics
    probability_statistics = {
        "minimum": float(np.min(val_probabilities)),
        "maximum": float(np.max(val_probabilities)),
        "mean": float(np.mean(val_probabilities)),
        "median": float(np.median(val_probabilities)),
        "legitimate_mean": float(
            np.mean(val_probabilities[val_labels == 0])
        ),
        "fraud_mean": float(
            np.mean(val_probabilities[val_labels == 1])
        )
    }

    print("\nVALIDATION PROBABILITY STATISTICS")

    for name, value in probability_statistics.items():
        print(f"{name}: {value:.6f}")

    # Final test evaluation
    print("\nEvaluating on test data...")

    test_probabilities = predict_probabilities(model, X_test)

    test_metrics, test_predictions = evaluate_model(
        y_test.to_numpy(),
        test_probabilities,
        threshold
    )

    # Save results
    RESULTS_PATH.mkdir(parents=True, exist_ok=True)
    METRICS_PATH.mkdir(parents=True, exist_ok=True)

    results = {
        "best_epoch": int(checkpoint["epoch"]),
        "validation_threshold": threshold,
        "validation_threshold_analysis": threshold_results,
        "validation_probability_statistics": probability_statistics,
        "test_metrics": test_metrics
    }

    with open(
        METRICS_PATH / "mlp_results.json",
        "w",
        encoding="utf-8"
    ) as file:
        json.dump(results, file, indent=4)

    create_visualizations(
        y_test.to_numpy(),
        test_predictions,
        test_probabilities,
        val_probabilities,
        val_labels
    )

    print("\nEvaluation completed successfully.")
    print("Results saved in:", RESULTS_PATH)


if __name__ == "__main__":
    main()
