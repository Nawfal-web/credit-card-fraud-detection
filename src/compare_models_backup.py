import json
from pathlib import Path

import pandas as pd
import matplotlib.pyplot as plt


PROJECT_ROOT = Path(__file__).resolve().parents[1]
METRICS_PATH = PROJECT_ROOT / "results" / "metrics"
FIGURES_PATH = PROJECT_ROOT / "results" / "figures"

FIGURES_PATH.mkdir(parents=True, exist_ok=True)


def main():
    # Load traditional machine-learning results
    with open(
        METRICS_PATH / "baseline_results.json",
        "r",
        encoding="utf-8"
    ) as file:
        baseline_results = json.load(file)

    # Add actual PyTorch MLP evaluation results
    mlp_results = {
        "model": "PyTorch MLP",
        "accuracy": 0.9994,
        "precision": 0.9032,
        "recall": 0.7568,
        "f1_score": 0.8235,
        "roc_auc": 0.9723,
        "pr_auc": 0.7667
    }

    all_results = baseline_results + [mlp_results]

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

    df = df[columns]
    df = df.sort_values("f1_score", ascending=False)

    # Save comparison table
    output_path = METRICS_PATH / "model_comparison.csv"
    df.to_csv(output_path, index=False)

    print("\nFINAL MODEL COMPARISON")
    print(df.to_string(index=False))

    print(f"\nComparison saved to: {output_path}")

    # Create F1-score and PR-AUC comparison charts
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