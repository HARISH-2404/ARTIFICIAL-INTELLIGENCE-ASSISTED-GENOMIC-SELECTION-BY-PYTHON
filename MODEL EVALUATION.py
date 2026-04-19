"""
Evaluation Module for AI-Assisted Genomic Selection
==================================================

This module provides comprehensive evaluation metrics and utilities:
- Regression metrics (RMSE, MAE, R2)
- Correlation (Pearson)
- Cross-validation
- Model comparison
- Multi-trait evaluation
- Residual diagnostics
- Visualization helpers (data prep only)

Designed for research-grade genomic prediction studies.
"""

import numpy as np
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.model_selection import KFold

# =========================
# 1. BASIC METRICS
# =========================

def calculate_rmse(y_true, y_pred):
    return np.sqrt(mean_squared_error(y_true, y_pred))


def calculate_mae(y_true, y_pred):
    return mean_absolute_error(y_true, y_pred)


def calculate_r2(y_true, y_pred):
    return r2_score(y_true, y_pred)

# =========================
# 2. CORRELATION (ACCURACY IN GS)
# =========================

def pearson_correlation(y_true, y_pred):
    return np.corrcoef(y_true, y_pred)[0, 1]

# =========================
# 3. BREEDING VALUE ACCURACY
# =========================

def prediction_accuracy(y_true, y_pred):
    return pearson_correlation(y_true, y_pred)

# =========================
# 4. RESIDUAL ANALYSIS
# =========================

def compute_residuals(y_true, y_pred):
    return y_true - y_pred


def residual_summary(residuals):
    return {
        "mean": np.mean(residuals),
        "std": np.std(residuals),
        "min": np.min(residuals),
        "max": np.max(residuals)
    }

# =========================
# 5. CROSS VALIDATION
# =========================

def cross_validate_model(model_func, X, y, folds=5):
    kf = KFold(n_splits=folds, shuffle=True, random_state=42)
    rmse_list, r2_list, corr_list = [], [], []

    for train_idx, test_idx in kf.split(X):
        X_train, X_test = X[train_idx], X[test_idx]
        y_train, y_test = y[train_idx], y[test_idx]

        model, _ = model_func(X_train, y_train)
        preds = model.predict(X_test)

        rmse_list.append(calculate_rmse(y_test, preds))
        r2_list.append(calculate_r2(y_test, preds))
        corr_list.append(pearson_correlation(y_test, preds))

    return {
        "RMSE": np.mean(rmse_list),
        "R2": np.mean(r2_list),
        "Correlation": np.mean(corr_list)
    }

# =========================
# 6. MULTI-TRAIT EVALUATION
# =========================

def multi_trait_evaluation(Y_true, Y_pred):
    results = []

    for i in range(Y_true.shape[1]):
        metrics = {
            "Trait": i,
            "RMSE": calculate_rmse(Y_true[:, i], Y_pred[:, i]),
            "R2": calculate_r2(Y_true[:, i], Y_pred[:, i]),
            "Correlation": pearson_correlation(Y_true[:, i], Y_pred[:, i])
        }
        results.append(metrics)

    return results

# =========================
# 7. MODEL COMPARISON
# =========================

def compare_models(models_dict, X, y):
    results = {}

    for name, model_func in models_dict.items():
        results[name] = cross_validate_model(model_func, X, y)

    return results

# =========================
# 8. RANK MODELS
# =========================

def rank_models(results, metric="RMSE"):
    return sorted(results.items(), key=lambda x: x[1][metric])

# =========================
# 9. ERROR DISTRIBUTION
# =========================

def error_distribution(y_true, y_pred, bins=10):
    residuals = compute_residuals(y_true, y_pred)
    hist, bin_edges = np.histogram(residuals, bins=bins)
    return hist, bin_edges

# =========================
# 10. CONFIDENCE INTERVALS
# =========================

def confidence_interval(metric_values, confidence=0.95):
    mean = np.mean(metric_values)
    std = np.std(metric_values)
    margin = 1.96 * std / np.sqrt(len(metric_values))
    return mean - margin, mean + margin

# =========================
# 11. BOOTSTRAP EVALUATION
# =========================

def bootstrap_evaluation(y_true, y_pred, n_iterations=100):
    scores = []
    n = len(y_true)

    for _ in range(n_iterations):
        indices = np.random.choice(range(n), n, replace=True)
        score = pearson_correlation(y_true[indices], y_pred[indices])
        scores.append(score)

    return np.mean(scores), np.std(scores)

# =========================
# 12. PIPELINE
# =========================

def evaluation_pipeline(model_func, X, y):
    metrics = cross_validate_model(model_func, X, y)
    return metrics

# =========================
# END MODULE
# =========================
