
import json
import joblib

from pathlib import Path

import pandas as pd

from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    average_precision_score
)

from xgboost import XGBClassifier


def calculate_metrics(y_true, probabilities):

    predictions = (
        probabilities >= 0.5
    ).astype(int)

    return {

        "accuracy": accuracy_score(
            y_true,
            predictions
        ),

        "precision": precision_score(
            y_true,
            predictions,
            zero_division=0
        ),

        "recall": recall_score(
            y_true,
            predictions,
            zero_division=0
        ),

        "f1_score": f1_score(
            y_true,
            predictions,
            zero_division=0
        ),

        "roc_auc": roc_auc_score(
            y_true,
            probabilities
        ),

        "pr_auc": average_precision_score(
            y_true,
            probabilities
        )
    }


def train_traditional_models(
    X_train,
    y_train,
    X_test,
    y_test
):

    Path("models").mkdir(exist_ok=True)

    results = {}

    models = {

        "Logistic Regression": LogisticRegression(
            class_weight="balanced",
            max_iter=2000,
            random_state=42
        ),

        "Random Forest": RandomForestClassifier(
            n_estimators=300,
            class_weight="balanced_subsample",
            random_state=42,
            n_jobs=-1
        ),

        "XGBoost": XGBClassifier(
            n_estimators=400,
            max_depth=6,
            learning_rate=0.05,
            subsample=0.9,
            colsample_bytree=0.9,

            scale_pos_weight=(
                (y_train == 0).sum()
                / (y_train == 1).sum()
            ),

            objective="binary:logistic",
            eval_metric="logloss",
            random_state=42,
            n_jobs=-1
        )
    }

    for name, model in models.items():

        print(f"\nTraining {name}...")

        model.fit(
            X_train,
            y_train
        )

        probabilities = model.predict_proba(
            X_test
        )[:, 1]

        metrics = calculate_metrics(
            y_test,
            probabilities
        )

        results[name] = metrics

        filename = (
            name.lower()
            .replace(" ", "_")
            + ".joblib"
        )

        joblib.dump(
            model,
            Path("models") / filename
        )

        print(f"\n{name} Results:")

        for metric, value in metrics.items():
            print(f"{metric}: {value:.6f}")

    # Save comparison results
    Path("results/metrics").mkdir(
        parents=True,
        exist_ok=True
    )

    with open(
        "results/metrics/traditional_models_metrics.json",
        "w"
    ) as file:

        json.dump(
            results,
            file,
            indent=4
        )

    pd.DataFrame(results).T.to_csv(
        "results/metrics/traditional_models_comparison.csv"
    )

    print("\nTraditional models training completed!")

    return results


if __name__ == "__main__":

    from preprocessing import load_and_prepare_data

    data = load_and_prepare_data()

    train_traditional_models(
        data["X_train"],
        data["y_train"],
        data["X_test"],
        data["y_test"]
    )
