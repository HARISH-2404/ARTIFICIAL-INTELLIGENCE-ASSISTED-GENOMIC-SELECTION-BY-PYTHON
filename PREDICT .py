import os
import json
import time
import numpy as np

from utils.logger import get_logger
from utils.io_utils import load_model, save_predictions


logger = get_logger("PREDICTION_PIPELINE")


# =========================
# CONFIG LOADER
# =========================

def load_config(path):
    with open(path, "r") as f:
        return json.load(f)


# =========================
# DATA LOADING
# =========================

def load_data(config):
    logger.info("Loading test data...")
    X_test = np.load(config["data"]["test_path"])
    logger.info(f"Test shape: {X_test.shape}")
    return X_test


# =========================
# MODEL LOADING
# =========================

def load_trained_model(config):
    logger.info("Loading model...")
    model = load_model(config["model"]["save_path"])
    return model


# =========================
# SINGLE PREDICTION
# =========================

def predict_single(model, X):
    logger.info("Running prediction...")
    return model.predict(X)


# =========================
# PROBABILITY OUTPUT
# =========================

def predict_proba(model, X):
    if hasattr(model, "predict_proba"):
        return model.predict_proba(X)
    return None


# =========================
# BATCH PREDICTION
# =========================

def batch_predict(model, X, batch_size=64):
    logger.info("Running batch prediction...")

    preds = []
    for i in range(0, X.shape[0], batch_size):
        batch = X[i:i+batch_size]
        preds.append(model.predict(batch))

    return np.concatenate(preds, axis=0)


# =========================
# ENSEMBLE PREDICTION
# =========================

def ensemble_predict(models, X):
    logger.info("Running ensemble prediction...")

    all_preds = []
    for m in models:
        all_preds.append(m.predict(X))

    return np.mean(all_preds, axis=0)


# =========================
# UNCERTAINTY ESTIMATION
# =========================

def compute_uncertainty(pred_matrix):
    return np.std(pred_matrix, axis=0)


# =========================
# GENOMIC SELECTION INDEX
# =========================

def genomic_selection_score(preds, weights=None):
    logger.info("Computing genomic selection index...")

    if weights is None:
        weights = np.ones(preds.shape[1])

    return np.dot(preds, weights)


# =========================
# SAVE OUTPUTS
# =========================

def save_results(preds, config, name="predictions.npy"):
    out_path = config["output"]["predictions"]
    np.save(out_path, preds)
    logger.info(f"Saved predictions -> {out_path}")


# =========================
# MAIN PIPELINE
# =========================

def predict(model, config):

    start = time.time()

    X = load_data(config)

    # prediction
    preds = predict_single(model, X)

    # probability (if classification)
    proba = predict_proba(model, X)

    # genomic selection
    if config.get("task") == "genomic_selection":
        gs_index = genomic_selection_score(preds)
        np.save(config["output"]["gs_index"], gs_index)

    save_results(preds, config)

    logger.info(f"Inference completed in {time.time() - start:.2f}s")

    return preds


# =========================
# CLI ENTRY
# =========================

if __name__ == "__main__":

    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)

    args = parser.parse_args()

    config = load_config(args.config)

    model = load_trained_model(config)

    predict(model, config)
