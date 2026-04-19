"""
Model Utilities for AI-Assisted Genomic Selection
================================================

Handles:
- Saving & loading ML models (sklearn)
- Saving & loading DL models (Keras)
- Saving preprocessing objects (scalers, PCA)
- Versioning and naming
- Reproducibility helpers
"""

import os
import joblib
import json
from datetime import datetime

# Optional (only if DL is used)
try:
    from tensorflow.keras.models import save_model, load_model
    KERAS_AVAILABLE = True
except ImportError:
    KERAS_AVAILABLE = False

# =========================
# 1. DIRECTORY MANAGEMENT
# =========================

def ensure_dir(path):
    if not os.path.exists(path):
        os.makedirs(path)

# =========================
# 2. NAMING UTILITIES
# =========================

def generate_model_name(base_name):
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    return f"{base_name}_{timestamp}"

# =========================
# 3. SAVE SKLEARN MODEL
# =========================

def save_sklearn_model(model, path, name):
    ensure_dir(path)
    filename = os.path.join(path, name + ".pkl")
    joblib.dump(model, filename)
    return filename

# =========================
# 4. LOAD SKLEARN MODEL
# =========================

def load_sklearn_model(filepath):
    return joblib.load(filepath)

# =========================
# 5. SAVE KERAS MODEL
# =========================

def save_keras_model(model, path, name):
    if not KERAS_AVAILABLE:
        raise ImportError("TensorFlow/Keras not installed")

    ensure_dir(path)
    filepath = os.path.join(path, name + ".h5")
    save_model(model, filepath)
    return filepath

# =========================
# 6. LOAD KERAS MODEL
# =========================

def load_keras_model(filepath):
    if not KERAS_AVAILABLE:
        raise ImportError("TensorFlow/Keras not installed")
    return load_model(filepath)

# =========================
# 7. SAVE PREPROCESSOR (SCALER/PCA)
# =========================

def save_preprocessor(obj, path, name):
    ensure_dir(path)
    filepath = os.path.join(path, name + ".pkl")
    joblib.dump(obj, filepath)
    return filepath

# =========================
# 8. LOAD PREPROCESSOR
# =========================

def load_preprocessor(filepath):
    return joblib.load(filepath)

# =========================
# 9. METADATA LOGGING
# =========================

def save_metadata(metadata, path, name="metadata"):
    ensure_dir(path)
    filepath = os.path.join(path, name + ".json")
    with open(filepath, "w") as f:
        json.dump(metadata, f, indent=4)
    return filepath

# =========================
# 10. LOAD METADATA
# =========================

def load_metadata(filepath):
    with open(filepath, "r") as f:
        return json.load(f)

# =========================
# 11. FULL SAVE PIPELINE
# =========================

def save_full_pipeline(model, scaler=None, pca=None, base_path="models", model_name="model"):
    model_name = generate_model_name(model_name)

    model_path = os.path.join(base_path, "trained_models")
    scaler_path = os.path.join(base_path, "scalers")
    pca_path = os.path.join(base_path, "pca")

    saved_files = {}

    # Save model
    if hasattr(model, "predict"):
        saved_files['model'] = save_sklearn_model(model, model_path, model_name)
    else:
        saved_files['model'] = save_keras_model(model, model_path, model_name)

    # Save scaler
    if scaler is not None:
        saved_files['scaler'] = save_preprocessor(scaler, scaler_path, model_name + "_scaler")

    # Save PCA
    if pca is not None:
        saved_files['pca'] = save_preprocessor(pca, pca_path, model_name + "_pca")

    # Save metadata
    metadata = {
        "model_name": model_name,
        "timestamp": datetime.now().isoformat(),
        "files": saved_files
    }

    meta_path = save_metadata(metadata, model_path, model_name + "_meta")
    saved_files['metadata'] = meta_path

    return saved_files

# =========================
# 12. LOAD FULL PIPELINE
# =========================

def load_full_pipeline(metadata_path):
    metadata = load_metadata(metadata_path)
    files = metadata['files']

    model = None
    if files['model'].endswith(".pkl"):
        model = load_sklearn_model(files['model'])
    elif files['model'].endswith(".h5"):
        model = load_keras_model(files['model'])

    scaler = None
    if 'scaler' in files:
        scaler = load_preprocessor(files['scaler'])

    pca = None
    if 'pca' in files:
        pca = load_preprocessor(files['pca'])

    return {
        "model": model,
