import numpy as np

from grader_contracts.linear_models import (
    BaselineComparisonResult,
    ClassificationInput,
    ClassificationMetrics,
    ClassificationResult,
    RegressionInput,
    RegressionResult,
)


def _matrix(features) -> np.ndarray:
    """Признаки в виде матрицы: строки — объекты, столбцы — признаки."""
    array = np.asarray(features, dtype=float)
    if array.ndim == 1:
        array = array.reshape(-1, 1)
    return array


def _vector(target) -> np.ndarray:
    """Цель в виде одномерного массива."""
    return np.asarray(target, dtype=float).reshape(-1)


def _standardize(train: np.ndarray, test: np.ndarray):
    """Стандартизация (x - среднее) / стандартное отклонение."""
    mean = train.mean(axis=0)
    std = train.std(axis=0)
    std[std == 0] = 1.0
    return (train - mean) / std, (test - mean) / std


def _sigmoid(z: np.ndarray) -> np.ndarray:
    """Сигмоида: превращает любое число в вероятность от 0 до 1."""
    z = np.clip(z, -30, 30)
    return 1.0 / (1.0 + np.exp(-z))


def _cross_entropy(z: np.ndarray, y: np.ndarray) -> float:
    """Бинарная кросс-энтропия — функция потерь классификатора."""
    p = np.clip(_sigmoid(z), 1e-7, 1 - 1e-7)
    return float(np.mean(-(y * np.log(p) + (1 - y) * np.log(1 - p))))


def train_linear_regression(data: RegressionInput) -> RegressionResult:
    """Обучите собственную линейную регрессию на train и верните прогноз для test."""
    
    x_train, x_test = _standardize(
        _matrix(data.train_features), _matrix(data.test_features)
    )
    y = _vector(data.train_target)

    n, d = x_train.shape
    weights = np.zeros(d)
    bias = 0.0
    loss_history = []

    for _ in range(data.epochs):
        error = x_train @ weights + bias - y
        weights -= data.learning_rate * (x_train.T @ error) / n
        bias -= data.learning_rate * error.mean()
        loss_history.append(float(np.mean((x_train @ weights + bias - y) ** 2)))

    predictions = x_test @ weights + bias

    return RegressionResult(
        weights=weights,
        bias=float(bias),
        loss_history=loss_history,
        predictions=predictions,
    )


def train_logistic_classifier(data: ClassificationInput) -> ClassificationResult:
    """Обучите собственный бинарный классификатор на train и верните прогноз для test."""
    
    x_train, x_test = _standardize(
        _matrix(data.train_features), _matrix(data.test_features)
    )
    y = _vector(data.train_target)
    y = (y - y.min() > 0.5).astype(float)

    n, d = x_train.shape
    weights = np.zeros(d)
    bias = 0.0
    loss_history = []

    for _ in range(data.epochs):
        error = _sigmoid(x_train @ weights + bias) - y
        weights -= data.learning_rate * (x_train.T @ error) / n
        bias -= data.learning_rate * error.mean()
        loss_history.append(_cross_entropy(x_train @ weights + bias, y))

    probabilities = _sigmoid(x_test @ weights + bias)
    predictions = (probabilities >= 0.5).astype(int)

    return ClassificationResult(
        weights=weights,
        bias=float(bias),
        loss_history=loss_history,
        probabilities=probabilities,
        predictions=predictions,
    )


def _metrics(target, probabilities) -> ClassificationMetrics:
    """Пять метрик классификации по вероятностям и порогу 0.5."""
    from sklearn.metrics import (
        accuracy_score,
        f1_score,
        precision_score,
        recall_score,
        roc_auc_score,
    )

    target = np.asarray(target).astype(int)
    predictions = (np.asarray(probabilities) >= 0.5).astype(int)
    return ClassificationMetrics(
        roc_auc=float(roc_auc_score(target, probabilities)),
        accuracy=float(accuracy_score(target, predictions)),
        precision=float(precision_score(target, predictions)),
        recall=float(recall_score(target, predictions)),
        f1=float(f1_score(target, predictions)),
    )


def compare_classical_models(data: ClassificationInput) -> BaselineComparisonResult:
    """Сравните готовые Logistic Regression, Decision Tree и KNN на одном split.

    После запуска запишите реальные результаты в experiment_results.json:
    {"linear_regression":{"test_mse": число},
     "classification":{"own_logistic": {"roc_auc": ..., "accuracy": ..., "precision": ..., "recall": ..., "f1": ...},
                       "logistic_regression": {...}, "decision_tree": {...}, "knn": {...}}}.
    """
    from sklearn.linear_model import LogisticRegression
    from sklearn.neighbors import KNeighborsClassifier
    from sklearn.tree import DecisionTreeClassifier

    x_train, x_test = _standardize(
        _matrix(data.train_features), _matrix(data.test_features)
    )
    y_train = _vector(data.train_target)
    y_test = _vector(data.test_target)
    y_train = (y_train - y_train.min() > 0.5).astype(int)
    y_test = (y_test - y_test.min() > 0.5).astype(int)

    models = {
        "logistic_regression": LogisticRegression(max_iter=1000, random_state=0),
        "decision_tree": DecisionTreeClassifier(random_state=0),
        "knn": KNeighborsClassifier(n_neighbors=5),
    }

    results = {}
    for name, model in models.items():
        model.fit(x_train, y_train)
        results[name] = _metrics(y_test, model.predict_proba(x_test)[:, 1])

    return BaselineComparisonResult(**results)
