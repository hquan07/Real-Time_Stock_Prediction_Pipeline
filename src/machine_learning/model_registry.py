import os
import json
import pickle
from datetime import datetime
from typing import Any, Dict, Optional, Tuple
from loguru import logger

REGISTRY_DIR = os.getenv("MODEL_REGISTRY_DIR", "src/machine_learning/artifacts/registry")

class ModelRegistry:
    @staticmethod
    def _get_version_dir(version: str) -> str:
        return os.path.join(REGISTRY_DIR, version)

    @staticmethod
    def save_model(model: Any, scaler: Any, features: list, metrics: Dict[str, float], model_type: str = "rf") -> str:
        """
        Save model, scaler, and metadata to a new versioned directory.
        Returns the version string.
        """
        os.makedirs(REGISTRY_DIR, exist_ok=True)
        version = f"{model_type}_v{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        version_dir = ModelRegistry._get_version_dir(version)
        os.makedirs(version_dir, exist_ok=True)
        
        # Save Model
        model_path = os.path.join(version_dir, "model.pkl")
        with open(model_path, "wb") as f:
            pickle.dump(model, f)
            
        # Save Scaler
        scaler_path = os.path.join(version_dir, "scaler.pkl")
        with open(scaler_path, "wb") as f:
            pickle.dump(scaler, f)
            
        # Save Metadata
        metadata = {
            "version": version,
            "model_type": model_type,
            "features": features,
            "metrics": metrics,
            "created_at": datetime.now().isoformat(),
        }
        metadata_path = os.path.join(version_dir, "metadata.json")
        with open(metadata_path, "w") as f:
            json.dump(metadata, f, indent=4)
            
        # Update latest symlink/file
        latest_path = os.path.join(REGISTRY_DIR, "latest.json")
        with open(latest_path, "w") as f:
            json.dump({"latest_version": version}, f)
            
        logger.info(f"✅ Model registered successfully: {version}")
        return version

    @staticmethod
    def load_latest_model() -> Tuple[Any, Any, Dict[str, Any]]:
        """
        Load the latest model, scaler, and metadata.
        Returns (model, scaler, metadata).
        """
        latest_path = os.path.join(REGISTRY_DIR, "latest.json")
        if not os.path.exists(latest_path):
            raise FileNotFoundError("No registered models found.")
            
        with open(latest_path, "r") as f:
            latest_version = json.load(f).get("latest_version")
            
        return ModelRegistry.load_model_version(latest_version)

    @staticmethod
    def load_model_version(version: str) -> Tuple[Any, Any, Dict[str, Any]]:
        version_dir = ModelRegistry._get_version_dir(version)
        if not os.path.exists(version_dir):
            raise FileNotFoundError(f"Version {version} not found in registry.")
            
        # Load Model
        with open(os.path.join(version_dir, "model.pkl"), "rb") as f:
            model = pickle.load(f)
            
        # Load Scaler
        with open(os.path.join(version_dir, "scaler.pkl"), "rb") as f:
            scaler = pickle.load(f)
            
        # Load Metadata
        with open(os.path.join(version_dir, "metadata.json"), "r") as f:
            metadata = json.load(f)
            
        return model, scaler, metadata
