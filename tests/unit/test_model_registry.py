import os
import json
import pickle
import pytest
from datetime import datetime
from src.machine_learning.model_registry import ModelRegistry

@pytest.fixture
def temp_registry(tmp_path, monkeypatch):
    """Fixture to set up a temporary model registry directory."""
    registry_dir = tmp_path / "registry"
    monkeypatch.setattr("src.machine_learning.model_registry.REGISTRY_DIR", str(registry_dir))
    return str(registry_dir)

def test_save_and_load_model(temp_registry):
    # Dummy data
    model = {"dummy_model": "RandomForestRegressor"}
    scaler = {"dummy_scaler": "StandardScaler"}
    features = ["open", "high", "low", "close", "volume"]
    metrics = {"test_rmse": 0.5, "test_mape": 2.0}
    model_type = "rf"

    # Save model
    version = ModelRegistry.save_model(model, scaler, features, metrics, model_type)

    # Check if files were created
    version_dir = os.path.join(temp_registry, version)
    assert os.path.exists(version_dir)
    assert os.path.exists(os.path.join(version_dir, "model.pkl"))
    assert os.path.exists(os.path.join(version_dir, "scaler.pkl"))
    assert os.path.exists(os.path.join(version_dir, "metadata.json"))
    
    # Check latest.json
    latest_path = os.path.join(temp_registry, "latest.json")
    assert os.path.exists(latest_path)
    with open(latest_path, "r") as f:
        latest_data = json.load(f)
        assert latest_data["latest_version"] == version

    # Load latest model
    loaded_model, loaded_scaler, loaded_metadata = ModelRegistry.load_latest_model()
    
    assert loaded_model == model
    assert loaded_scaler == scaler
    assert loaded_metadata["version"] == version
    assert loaded_metadata["model_type"] == model_type
    assert loaded_metadata["features"] == features
    assert loaded_metadata["metrics"] == metrics

def test_load_latest_model_not_found(temp_registry):
    with pytest.raises(FileNotFoundError):
        ModelRegistry.load_latest_model()

def test_load_model_version_not_found(temp_registry):
    with pytest.raises(FileNotFoundError):
        ModelRegistry.load_model_version("non_existent_version")
