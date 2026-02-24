"""
ML Model inference module.
Handles loading trained models and making predictions.
"""

import os
import pickle
import json
from datetime import date, datetime
from typing import Dict, Any, List, Optional, Union

import numpy as np
import pandas as pd
from loguru import logger


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
MODEL_DIR = os.getenv("MODEL_DIR", "src/machine_learning/models")
ARTIFACT_DIR = os.getenv("ARTIFACT_DIR", "src/machine_learning/artifacts")


# ---------------------------------------------------------------------------
# Model Loading
# ---------------------------------------------------------------------------
class ModelLoader:
    """Utility class for loading trained models."""

    _cache: Dict[str, Any] = {}

    @classmethod
    def load_model(cls, model_path: str = None, use_cache: bool = True) -> Any:
        """
        Load a trained model from disk.
        
        Args:
            model_path: Path to model file (.pkl)
            use_cache: Whether to cache loaded models
        
        Returns:
            Loaded model object.
        """
        if model_path is None:
            model_path = os.path.join(MODEL_DIR, "model.pkl")

        if use_cache and model_path in cls._cache:
            logger.debug(f"Using cached model from {model_path}")
            return cls._cache[model_path]

        try:
            with open(model_path, "rb") as f:
                model = pickle.load(f)

            if use_cache:
                cls._cache[model_path] = model

            logger.info(f"✅ Model loaded from {model_path}")
            return model

        except FileNotFoundError:
            logger.error(f"❌ Model file not found: {model_path}")
            raise
        except Exception as e:
            logger.error(f"❌ Failed to load model: {e}")
            raise

    @classmethod
    def load_scaler(cls, scaler_path: str = None) -> Any:
        """Load scaler from disk."""
        if scaler_path is None:
            scaler_path = os.path.join(ARTIFACT_DIR, "scaler.pkl")

        try:
            with open(scaler_path, "rb") as f:
                scaler_data = pickle.load(f)
            logger.info(f"✅ Scaler loaded from {scaler_path}")
            return scaler_data
        except FileNotFoundError:
            logger.warning(f"⚠️ Scaler file not found: {scaler_path}")
            return None

    @classmethod
    def load_metrics(cls, metrics_path: str = None) -> Optional[Dict]:
        """Load model metrics from disk."""
        if metrics_path is None:
            metrics_path = os.path.join(ARTIFACT_DIR, "metrics.json")

        try:
            with open(metrics_path, "r") as f:
                return json.load(f)
        except FileNotFoundError:
            return None

    @classmethod
    def clear_cache(cls):
        """Clear model cache."""
        cls._cache.clear()
        logger.info("✅ Model cache cleared")


# ---------------------------------------------------------------------------
# Inference Engine
# ---------------------------------------------------------------------------
class InferenceEngine:
    """
    Engine for making predictions with trained models.
    """

    def __init__(
        self,
        model_path: str = None,
        scaler_path: str = None,
        feature_columns: List[str] = None,
    ):
        """
        Initialize inference engine.
        
        Args:
            model_path: Path to trained model
            scaler_path: Path to fitted scaler
            feature_columns: List of feature column names
        """
        self.model = ModelLoader.load_model(model_path)
        self.scaler_data = ModelLoader.load_scaler(scaler_path)
        self.feature_columns = feature_columns

        # Extract scaler if loaded
        self.scaler = None
        if self.scaler_data:
            self.scaler = self.scaler_data.get("scaler")
            if not self.feature_columns:
                self.feature_columns = self.scaler_data.get("feature_names", [])

        logger.info(f"✅ Inference engine initialized with {len(self.feature_columns)} features")

    def preprocess(self, data: Union[pd.DataFrame, np.ndarray, Dict]) -> np.ndarray:
        """
        Preprocess input data for prediction.
        
        Args:
            data: Input data (DataFrame, array, or dict)
        
        Returns:
            Preprocessed numpy array.
        """
        # Convert dict to DataFrame
        if isinstance(data, dict):
            data = pd.DataFrame([data])

        # Convert DataFrame to array
        if isinstance(data, pd.DataFrame):
            if self.feature_columns:
                # Ensure all required columns exist
                missing = set(self.feature_columns) - set(data.columns)
                if missing:
                    raise ValueError(f"Missing required features: {missing}")
                data = data[self.feature_columns].values
            else:
                data = data.select_dtypes(include=[np.number]).values

        # Apply scaling if available
        if self.scaler is not None:
            data = self.scaler.transform(data)

        return data

    def predict(
        self,
        data: Union[pd.DataFrame, np.ndarray, Dict],
        return_proba: bool = False,
    ) -> np.ndarray:
        """
        Make predictions on input data.
        
        Args:
            data: Input features
            return_proba: Return class probabilities (for classifiers)
        
        Returns:
            Prediction array.
        """
        # Preprocess
        X = self.preprocess(data)

        # Predict
        if return_proba and hasattr(self.model, "predict_proba"):
            predictions = self.model.predict_proba(X)
        else:
            predictions = self.model.predict(X)

        logger.debug(f"✅ Generated {len(predictions)} predictions")
        return predictions

    def predict_single(self, features: Dict[str, float]) -> float:
        """
        Make a single prediction from feature dict.
        
        Args:
            features: Dict mapping feature names to values
        
        Returns:
            Single prediction value.
        """
        predictions = self.predict(features)
        return float(predictions[0])


# ---------------------------------------------------------------------------
# Batch Inference
# ---------------------------------------------------------------------------
def batch_predict(
    df: pd.DataFrame,
    model_path: str = None,
    scaler_path: str = None,
    feature_columns: List[str] = None,
    output_column: str = "prediction",
) -> pd.DataFrame:
    """
    Run batch predictions on a DataFrame.
    
    Args:
        df: Input DataFrame
        model_path: Path to model
        scaler_path: Path to scaler
        feature_columns: Feature columns to use
        output_column: Name for prediction column
    
    Returns:
        DataFrame with predictions added.
    """
    engine = InferenceEngine(model_path, scaler_path, feature_columns)

    df = df.copy()
    predictions = engine.predict(df)
    df[output_column] = predictions

    logger.info(f"✅ Batch prediction complete: {len(df)} records")
    return df


# ---------------------------------------------------------------------------
# Real-time Inference
# ---------------------------------------------------------------------------
def predict_next_price(
    ticker: str,
    current_data: Dict[str, float],
    model_path: str = None,
    scaler_path: str = None,
) -> Dict[str, Any]:
    """
    Predict next price for a single ticker.
    
    Args:
        ticker: Stock ticker symbol
        current_data: Current market data (open, high, low, close, volume, etc.)
        model_path: Path to model
        scaler_path: Path to scaler
    
    Returns:
        Dict with prediction, confidence, timestamp.
    """
    try:
        engine = InferenceEngine(model_path, scaler_path)
        prediction = engine.predict_single(current_data)

        return {
            "ticker": ticker,
            "predicted_close": prediction,
            "prediction_time": datetime.now().isoformat(),
            "input_data": current_data,
            "status": "success",
        }

    except Exception as e:
        logger.error(f"❌ Prediction failed for {ticker}: {e}")
        return {
            "ticker": ticker,
            "predicted_close": None,
            "prediction_time": datetime.now().isoformat(),
            "error": str(e),
            "status": "error",
        }


# ---------------------------------------------------------------------------
# Model Info
# ---------------------------------------------------------------------------
def get_model_info(model_path: str = None) -> Dict[str, Any]:
    """
    Get information about a trained model.
    
    Returns:
        Dict with model type, features, metrics, etc.
    """
    if model_path is None:
        model_path = os.path.join(MODEL_DIR, "model.pkl")

    info = {
        "model_path": model_path,
        "exists": os.path.exists(model_path),
    }

    if info["exists"]:
        model = ModelLoader.load_model(model_path)

        info["model_type"] = type(model).__name__

        # Get feature names if available
        if hasattr(model, "feature_names_in_"):
            info["feature_names"] = list(model.feature_names_in_)

        # Get number of features
        if hasattr(model, "n_features_in_"):
            info["n_features"] = model.n_features_in_

        # Get model-specific info
        if hasattr(model, "n_estimators"):
            info["n_estimators"] = model.n_estimators

        # Load metrics if available
        metrics = ModelLoader.load_metrics()
        if metrics:
            info["metrics"] = metrics

        # File info
        stat = os.stat(model_path)
        info["file_size_mb"] = round(stat.st_size / (1024 * 1024), 2)
        info["last_modified"] = datetime.fromtimestamp(stat.st_mtime).isoformat()

    return info
