
import json
from pathlib import Path

import pandas as pd
import matplotlib.pyplot as plt


PROJECT_ROOT = Path(__file__).resolve().parents[1]
METRICS_PATH = PROJECT_ROOT / "results" / "metrics"
FIGURES_PATH = PROJECT_ROOT / "results" / "figures"

FIGURES_PATH.mkdir(parents=True, exist_ok=True)


def main():

    # Previously recorded traditional ML results
    baseline_results = [
        {
            "model": "Logistic Regression",
            "accuracy": 0.9995,
            "precision": 0.9206,
            "recall": 0.7838,
            "f1_score": 0.8467,
            "roc_auc": 0.9675,
            "pr_auc": 0.7721
        },
        {
            "model": "Random Forest",
            "accuracy": 0.9995,
            "precision": 0.9649,
            "recall": 0.7432,
            "f1_score": 0.8397,
            "roc_auc": 0.9556,
            "pr_auc": 0.8266
        },
        {
            "model": "XGBoost",
            "accuracy": 0.9996,
            "precision": 0.9365,
            "recall": 0.7973,
            "f1_score": 0.8613,
            "roc_auc": 0.9675,
            "pr_auc": 0.8411
        }
    ]

    # Load actual MLP test results
    with open(METRICS_PATH / "mlp_results.json", "r") as file:
        mlp_data = json.load(file)

    mlp = mlp_data["test_metrics"]

    mlp_results = {
        "model": "PyTorch MLP",
        "accuracy": mlp["accuracy"],
        "precision": mlp["precision"],
        "recall": mlp["recall"],
        "f1_score": mlp["f1_score"],
        "roc_auc": mlp["roc_auc"],
        "pr_auc": mlp["pr_auc"]
    }

    # Load actual CNN test results
    with open(METRICS_PATH / "cnn_results.json", "r") as file:
        cnn = json.load(file)

    cnn_results = {
        "model": "1D-CNN",
        "accuracy": cnn["accuracy"],
        "precision": cnn["precision"],
        "recall": cnn["recall"],
        "f1_score": cnn["f1_score"],
        "roc_auc": cnn["roc_auc"],
        "pr_auc": cnn["pr_auc"]
    }

    # Combine all five models
    all_results = baseline_results + [mlp_results, cnn_results]

    df = pd.DataFrame(all_results)

    columns = [
        "model",
        "accuracy",
        "precision",
        "recall",
        "f1_score",
        "roc_auc",
        "pr_auc"
    ]

    df = df[columns].sort_values(
        "f1_score",
        ascending=False
    )

    # Save comparison table
    output_path = METRICS_PATH / "model_comparison.csv"
    df.to_csv(output_path, index=False)

    print("\nFINAL MODEL COMPARISON")
    print(df.to_string(index=False))

    print(f"\nComparison saved to: {output_path}")

    # Generate comparison charts
    for metric in ["f1_score", "pr_auc"]:

        plt.figure(figsize=(9, 5))

        plt.bar(df["model"], df[metric])

        plt.title(f"Model Comparison - {metric.upper()}")
        plt.ylabel(metric.upper())
        plt.xlabel("Model")
        plt.ylim(0, 1)
        plt.xticks(rotation=15)

        plt.tight_layout()

        figure_path = FIGURES_PATH / f"{metric}_comparison.png"

        plt.savefig(figure_path, dpi=300)
        plt.close()

        print(f"Chart saved: {figure_path}")


if __name__ == "__main__":
    main()
