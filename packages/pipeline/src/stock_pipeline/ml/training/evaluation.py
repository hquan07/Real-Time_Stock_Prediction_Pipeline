"""Time-series evaluation helpers with an explicit no-change baseline."""

import json
from pathlib import Path
from typing import Callable

import numpy as np
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.model_selection import TimeSeriesSplit


def regression_metrics(y_true, y_pred) -> dict[str, float]:
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    if y_true.size == 0 or y_true.shape != y_pred.shape:
        raise ValueError("Evaluation arrays must be non-empty and have equal shapes")
    return {
        "rmse": float(np.sqrt(mean_squared_error(y_true, y_pred))),
        "mae": float(mean_absolute_error(y_true, y_pred)),
    }


def evaluate_against_zero_return_baseline(y_true, y_pred) -> dict:
    model = regression_metrics(y_true, y_pred)
    baseline = regression_metrics(y_true, np.zeros_like(np.asarray(y_true, dtype=float)))
    return {
        "model": model,
        "baseline": baseline,
        "baseline_name": "zero_return",
        "beats_baseline": model["rmse"] < baseline["rmse"],
        "uncertainty_validated": False,
    }


def walk_forward_evaluate(
    X,
    y,
    model_factory: Callable[[], object],
    n_splits: int = 5,
) -> dict:
    X_array = np.asarray(X)
    y_array = np.asarray(y, dtype=float)
    splitter = TimeSeriesSplit(n_splits=n_splits)
    predictions = []
    actuals = []
    fold_boundaries = []

    for fold, (train_idx, test_idx) in enumerate(splitter.split(X_array), start=1):
        model = model_factory()
        model.fit(X_array[train_idx], y_array[train_idx])
        predictions.extend(model.predict(X_array[test_idx]))
        actuals.extend(y_array[test_idx])
        fold_boundaries.append(
            {
                "fold": fold,
                "train_end": int(train_idx[-1]),
                "test_start": int(test_idx[0]),
                "test_end": int(test_idx[-1]),
            }
        )

    report = evaluate_against_zero_return_baseline(actuals, predictions)
    report["method"] = "expanding_walk_forward"
    report["folds"] = fold_boundaries
    report["sample_count"] = len(actuals)
    return report


def write_evaluation_report(report: dict, destination: Path) -> Path:
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    return destination
