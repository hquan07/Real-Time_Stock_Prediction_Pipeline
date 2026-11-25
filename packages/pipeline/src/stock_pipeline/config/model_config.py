import os

MODEL_DIR = os.getenv("MODEL_DIR", "models/")
ARTIFACT_DIR = os.getenv("ARTIFACT_DIR", "models/artifacts/")
DL_MODEL_DIR = os.getenv("DL_MODEL_DIR", "models/deep_learning/")

MODEL_PATH = os.path.join(MODEL_DIR, "model.pkl")
SCALER_PATH = os.path.join(ARTIFACT_DIR, "scaler.pkl")
ENCODER_PATH = os.path.join(ARTIFACT_DIR, "encoder.pkl")
METRICS_PATH = os.path.join(ARTIFACT_DIR, "metrics.json")

LSTM_MODEL_PATH = os.path.join(DL_MODEL_DIR, "lstm_model.h5")
TOKENIZER_PATH = os.path.join(DL_MODEL_DIR, "tokenizer.pkl")
DL_METRICS_PATH = os.path.join(DL_MODEL_DIR, "dl_metrics.json")

# Training parameters
TRAIN_TEST_SPLIT = 0.8
RANDOM_STATE = 42
EPOCHS = int(os.getenv("EPOCHS", "20"))
BATCH_SIZE = int(os.getenv("BATCH_SIZE", "32"))