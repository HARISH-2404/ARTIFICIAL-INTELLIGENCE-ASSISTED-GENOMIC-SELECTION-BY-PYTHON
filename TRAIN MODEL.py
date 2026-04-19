import os
import json
import time
import numpy as np
import pandas as pd

from sklearn.model_selection import train_test_split, KFold
from sklearn.metrics import accuracy_score, r2_score
from sklearn.preprocessing import StandardScaler

from models.model_utils import build_model
from utils.logger import get_logger
from utils.io_utils import save_model, save_metrics


logger = get_logger("TRAINING_PIPELINE")


# ================================
# CONFIG LOADER
# ================================

def load_config(path):
    with open(path, "r") as f:
        config = json.load(f)
    return config


# ================================
# DATA LOADING
# ================================

def load_data(config):
    logger.info("Loading dataset...")

    X = np.load(config["data"]["X_path"])
    y = np.load(config["data"]["y_path"])

    logger.info(f"Dataset shape: X={X.shape}, y={y.shape}")
    return X, y


# ================================
# PREPROCESSING
# ================================

def preprocess(X, config):
    logger.info("Starting preprocessing...")

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    if config["preprocessing"]["save_scaler"]:
        save_model(scaler, config["model"]["scaler_path"])

    return X_scaled, scaler


# ================================
# MODEL TRAINING CORE
# ================================

def train_single_model(X_train, y_train, X_val, y_val, config):
    model = build_model(config)

    logger.info("Training model...")

    model.fit(X_train, y_train)

    preds = model.predict(X_val)

    if config["task"] == "classification":
        score = accuracy_score(y_val, preds)
    else:
        score = r2_score(y_val, preds)

    logger.info(f"Validation Score: {score}")

    return model, score


# ================================
# CROSS VALIDATION TRAINING
# ================================

def cross_validate(X, y, config):
    logger.info("Starting K-Fold Cross Validation...")

    kf = KFold(n_splits=config["training"]["k_folds"], shuffle=True, random_state=42)

    scores = []
    models = []

    fold = 1

    for train_idx, val_idx in kf.split(X):

        logger.info(f"Fold {fold}")

        X_train, X_val = X[train_idx], X[val_idx]
        y_train, y_val = y[train_idx], y[val_idx]

        model, score = train_single_model(
            X_train, y_train, X_val, y_val, config
        )

        scores.append(score)
        models.append(model)

        fold += 1

    logger.info(f"Mean CV Score: {np.mean(scores)}")

    return models, scores


# ================================
# FINAL TRAINING
# ================================

def final_train(X, y, config):
    logger.info("Final model training started...")

    X_train, X_val, y_train, y_val = train_test_split(
        X,
        y,
        test_size=config["training"]["test_size"],
        random_state=42
    )

    model, score = train_single_model(
        X_train, y_train, X_val, y_val, config
    )

    return model, score


# ================================
# FEATURE IMPORTANCE (OPTIONAL)
# ================================

def compute_feature_importance(model, config):
    if hasattr(model, "feature_importances_"):
        importance = model.feature_importances_
        np.save(config["output"]["feature_importance"], importance)
        logger.info("Feature importance saved.")


# ================================
# MAIN TRAIN FUNCTION
# ================================

def train(config):

    start_time = time.time()

    X, y = load_data(config)

    X, scaler = preprocess(X, config)

    if config["training"]["use_cv"]:
        models, scores = cross_validate(X, y, config)
        best_model = models[np.argmax(scores)]
    else:
        best_model, score = final_train(X, y, config)

    save_model(best_model, config["model"]["save_path"])

    compute_feature_importance(best_model, config)

    save_metrics({
        "score": float(np.mean(score)) if not config["training"]["use_cv"] else float(np.mean(scores)),
        "time_taken": time.time() - start_time
    }, config["output"]["metrics_path"])

    logger.info("Training completed successfully!")

    return best_model


# ================================
# CLI ENTRY POINT
# ================================

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, required=True)

    args = parser.parse_args()

    config = load_config(args.config)

    train(config)
