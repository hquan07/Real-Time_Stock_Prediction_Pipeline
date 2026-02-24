"""
Machine Learning model training module.
Supports multiple model types with TimeSeriesSplit cross-validation.
"""

import os
import pickle
from typing import Dict, Any, Optional, Union

import numpy as np
import pandas as pd
from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import mean_squared_error, mean_absolute_error
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from loguru import logger

try:
    import xgboost as xgb
    XGBOOST_AVAILABLE = True
except ImportError:
    XGBOOST_AVAILABLE = False
    logger.warning("⚠️ XGBoost not installed. XGBoost model will not be available.")


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
MODEL_DIR = os.getenv("MODEL_DIR", "src/machine_learning/models")


# ---------------------------------------------------------------------------
# Model Factory
# ---------------------------------------------------------------------------
def get_model(model_type: str, **kwargs):
    """
    Get a model instance by type.
    
    Args:
        model_type: One of 'linear', 'random_forest', 'xgboost'
        **kwargs: Additional model parameters
    
    Returns:
        Model instance
    """
    if model_type == "linear":
        return LinearRegression()
    
    elif model_type == "random_forest":
        return RandomForestRegressor(
            n_estimators=kwargs.get("n_estimators", 200),
            max_depth=kwargs.get("max_depth", 10),
            min_samples_split=kwargs.get("min_samples_split", 5),
            random_state=kwargs.get("random_state", 42),
            n_jobs=-1,
        )
    
    elif model_type == "xgboost":
        if not XGBOOST_AVAILABLE:
            raise ImportError("XGBoost not installed. Install with: pip install xgboost")
        
        return xgb.XGBRegressor(
            n_estimators=kwargs.get("n_estimators", 200),
            max_depth=kwargs.get("max_depth", 6),
            learning_rate=kwargs.get("learning_rate", 0.1),
            subsample=kwargs.get("subsample", 0.8),
            colsample_bytree=kwargs.get("colsample_bytree", 0.8),
            random_state=kwargs.get("random_state", 42),
            n_jobs=-1,
            verbosity=0,
        )
    
    else:
        raise ValueError(f"Unknown model type: {model_type}")


# ---------------------------------------------------------------------------
# Training Functions
# ---------------------------------------------------------------------------
def train_and_evaluate(
    X: Union[pd.DataFrame, np.ndarray],
    y: Union[pd.Series, np.ndarray],
    model_types: list = None,
    n_splits: int = 5,
) -> Dict[str, Dict[str, float]]:
    """
    Train and evaluate multiple models using TimeSeriesSplit.
    
    Args:
        X: Feature matrix
        y: Target values
        model_types: List of model types to train. Defaults to all.
        n_splits: Number of cross-validation splits
    
    Returns:
        Dict mapping model name to metrics.
    """
    tscv = TimeSeriesSplit(n_splits=n_splits)

    if model_types is None:
        model_types = ["linear", "random_forest"]
        if XGBOOST_AVAILABLE:
            model_types.append("xgboost")

    models = {}
    for mt in model_types:
        try:
            models[mt] = get_model(mt)
        except ImportError as e:
            logger.warning(f"Skipping {mt}: {e}")

    results = {}

    for name, model in models.items():
        logger.info(f"Training {name}...")
        rmses = []
        maes = []

        for fold, (train_idx, test_idx) in enumerate(tscv.split(X)):
            if isinstance(X, pd.DataFrame):
                X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
            else:
                X_train, X_test = X[train_idx], X[test_idx]
                
            if isinstance(y, pd.Series):
                y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]
            else:
                y_train, y_test = y[train_idx], y[test_idx]

            model.fit(X_train, y_train)
            preds = model.predict(X_test)

            rmse = np.sqrt(mean_squared_error(y_test, preds))
            mae = mean_absolute_error(y_test, preds)
            rmses.append(rmse)
            maes.append(mae)

        results[name] = {
            "mean_rmse": float(np.mean(rmses)),
            "std_rmse": float(np.std(rmses)),
            "mean_mae": float(np.mean(maes)),
            "std_mae": float(np.std(maes)),
        }
        
        logger.info(f"  {name}: RMSE={np.mean(rmses):.4f} (±{np.std(rmses):.4f})")

    return results


def train_single_model(
    X: Union[pd.DataFrame, np.ndarray],
    y: Union[pd.Series, np.ndarray],
    model_type: str = "random_forest",
    save_model: bool = True,
    model_path: str = None,
    **model_params,
) -> Dict[str, Any]:
    """
    Train a single model and optionally save it.
    
    Args:
        X: Feature matrix
        y: Target values
        model_type: Type of model to train
        save_model: Whether to save the model
        model_path: Custom path for saving
        **model_params: Additional model parameters
    
    Returns:
        Dict with model, metrics, and path.
    """
    model = get_model(model_type, **model_params)
    model.fit(X, y)
    
    # Calculate metrics on training data
    preds = model.predict(X)
    train_rmse = np.sqrt(mean_squared_error(y, preds))
    train_mae = mean_absolute_error(y, preds)
    
    result = {
        "model": model,
        "model_type": model_type,
        "metrics": {
            "train_rmse": float(train_rmse),
            "train_mae": float(train_mae),
        }
    }
    
    if save_model:
        if model_path is None:
            os.makedirs(MODEL_DIR, exist_ok=True)
            model_path = os.path.join(MODEL_DIR, f"{model_type}_model.pkl")
        
        with open(model_path, "wb") as f:
            pickle.dump(model, f)
        
        result["model_path"] = model_path
        logger.info(f"✅ Model saved to {model_path}")
    
    return result


def compare_models(
    X: Union[pd.DataFrame, np.ndarray],
    y: Union[pd.Series, np.ndarray],
    n_splits: int = 5,
) -> Dict[str, Any]:
    """
    Compare all available models and return the best one.
    
    Args:
        X: Feature matrix
        y: Target values
        n_splits: Number of cross-validation splits
    
    Returns:
        Dict with best model info and comparison results.
    """
    results = train_and_evaluate(X, y, n_splits=n_splits)
    
    # Find best model by RMSE
    best_model_name = min(results, key=lambda k: results[k]["mean_rmse"])
    
    return {
        "best_model": best_model_name,
        "best_rmse": results[best_model_name]["mean_rmse"],
        "comparison": results,
    }