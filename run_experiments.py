import json
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    mean_squared_error,
    precision_score,
    r2_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import train_test_split

from grader_contracts.linear_models import ClassificationInput, RegressionInput
from linear_models import (
    compare_classical_models,
    train_linear_regression,
    train_logistic_classifier,
)

ROOT = Path(__file__).resolve().parent
FIGURES = ROOT / "figures"
SEED = 42
TEST_SIZE = 0.2
REG_TARGET = "Performance Index"


def load_student():
    """Успеваемость студентов: 5 признаков, целевой — Performance Index."""
    frame = pd.read_csv(ROOT / "Student_Performance.csv")
    features = frame.drop(columns=[REG_TARGET])
    features["Extracurricular Activities"] = (
        features["Extracurricular Activities"] == "Yes"
    ).astype(float)
    target = frame[REG_TARGET].to_numpy(float)
    return train_test_split(features, target, test_size=TEST_SIZE, random_state=SEED)


def load_german():
    """Кредиты: цель Creditability в первом столбце, разделитель ';'."""
    frame = pd.read_csv(ROOT / "german.csv", sep=";")
    target = frame.iloc[:, 0]
    features = frame.iloc[:, 1:]
    return train_test_split(
        features, target, test_size=TEST_SIZE, random_state=SEED, stratify=target
    )


def classification_metrics(target, probabilities) -> dict:
    """ROC-AUC, Accuracy, Precision, Recall, F1 при пороге 0.5."""
    target = np.asarray(target).astype(int)
    predictions = (np.asarray(probabilities) >= 0.5).astype(int)
    return {
        "roc_auc": float(roc_auc_score(target, probabilities)),
        "accuracy": float(accuracy_score(target, predictions)),
        "precision": float(precision_score(target, predictions)),
        "recall": float(recall_score(target, predictions)),
        "f1": float(f1_score(target, predictions)),
    }


def metrics_dict(metrics) -> dict:
    return {
        "roc_auc": float(metrics.roc_auc),
        "accuracy": float(metrics.accuracy),
        "precision": float(metrics.precision),
        "recall": float(metrics.recall),
        "f1": float(metrics.f1),
    }


def regression_without_scaling(x_train, x_test, y_train, y_test, epochs=800, lr=0.05):
    """Тот же градиентный спуск, но без стандартизации — для сравнения.

    Возвращает test MSE или None, если обучение разошлось (веса стали inf/nan).
    """
    x_train = np.asarray(x_train, dtype=float)
    x_test = np.asarray(x_test, dtype=float)
    weights = np.zeros(x_train.shape[1])
    bias = 0.0
    with np.errstate(over="ignore", invalid="ignore"):
        for _ in range(epochs):
            error = x_train @ weights + bias - y_train
            weights -= lr * (x_train.T @ error) / len(x_train)
            bias -= lr * error.mean()
        predictions = x_test @ weights + bias
    if not np.isfinite(predictions).all():
        return None
    return float(mean_squared_error(y_test, predictions))


def main():
    sys.stdout.reconfigure(encoding="utf-8")
    FIGURES.mkdir(exist_ok=True)

    # ---------- 1. Линейная регрессия ----------
    x_train, x_test, y_train, y_test = load_student()
    regression = train_linear_regression(
        RegressionInput(x_train, y_train, x_test, y_test)
    )
    test_mse = float(mean_squared_error(y_test, regression.predictions))

    print("=== Линейная регрессия (Student_Performance) ===")
    print(f"  train MSE: {regression.loss_history[0]:.1f} -> {regression.loss_history[-1]:.4f}")
    print(f"  test MSE: {test_mse:.4f}, test R2: {r2_score(y_test, regression.predictions):.4f}")

    # эксперимент: разный шаг обучения
    steps = {}
    for lr in (0.001, 0.01, 0.05, 0.1):
        result = train_linear_regression(
            RegressionInput(x_train, y_train, x_test, y_test, learning_rate=lr)
        )
        steps[lr] = round(float(mean_squared_error(y_test, result.predictions)), 4)
    print(f"  test MSE при lr=0.001/0.01/0.05/0.1: {steps}")

    # эксперимент: обучение без масштабирования
    without_scaling = regression_without_scaling(x_train, x_test, y_train, y_test)
    print(f"  test MSE без стандартизации (lr=0.05): {without_scaling}")

    plt.figure(figsize=(6.5, 4))
    plt.plot(regression.loss_history, color="#1f77b4")
    plt.axhline(test_mse, color="#d62728", ls="--", label="test MSE")
    plt.yscale("log")
    plt.xlabel("эпоха")
    plt.ylabel("MSE")
    plt.title("Линейная регрессия: ход обучения")
    plt.grid(alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig(FIGURES / "regression_loss.png", dpi=140)
    plt.close()

    # ---------- 2 и 3. Классификация ----------
    x_train, x_test, y_train, y_test = load_german()
    classification = ClassificationInput(x_train, y_train, x_test, y_test)

    own = train_logistic_classifier(classification)
    own_metrics = classification_metrics(y_test, own.probabilities)
    baselines = compare_classical_models(classification)

    results = {
        "linear_regression": {"test_mse": test_mse},
        "classification": {
            "own_logistic": own_metrics,
            "logistic_regression": metrics_dict(baselines.logistic_regression),
            "decision_tree": metrics_dict(baselines.decision_tree),
            "knn": metrics_dict(baselines.knn),
        },
    }

    print("\n=== Классификация (german) ===")
    print(f"  кросс-энтропия: {own.loss_history[0]:.4f} -> {own.loss_history[-1]:.4f}")
    for name, metrics in results["classification"].items():
        print(
            f"  {name:<20} AUC={metrics['roc_auc']:.4f} acc={metrics['accuracy']:.4f} "
            f"P={metrics['precision']:.4f} R={metrics['recall']:.4f} F1={metrics['f1']:.4f}"
        )

    # эксперимент: разный шаг обучения
    steps = {}
    for lr in (0.001, 0.01, 0.1, 1.0):
        result = train_logistic_classifier(
            ClassificationInput(x_train, y_train, x_test, y_test, learning_rate=lr)
        )
        metrics = classification_metrics(y_test, result.probabilities)
        steps[lr] = round(metrics["roc_auc"], 4)
    print(f"  ROC-AUC при lr=0.001/0.01/0.1/1.0: {steps}")

    plt.figure(figsize=(6.5, 4))
    plt.plot(own.loss_history, color="#1f77b4")
    plt.xlabel("эпоха")
    plt.ylabel("кросс-энтропия")
    plt.title("Логистическая регрессия: ход обучения")
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(FIGURES / "classification_loss.png", dpi=140)
    plt.close()

    (ROOT / "experiment_results.json").write_text(
        json.dumps(results, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    print("\nРезультаты записаны в experiment_results.json")


if __name__ == "__main__":
    main()
