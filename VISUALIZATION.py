"""
Visualization Module for AI-Assisted Genomic Selection
=====================================================

This module provides visualization tools for:
- PCA plots
- Correlation heatmaps
- Prediction vs Observed plots
- Residual plots
- Feature importance
- Genomic relationship matrix (GRM) heatmaps
- Training history (DL models)

Designed for publication-quality figures.
"""

import numpy as np
import matplotlib.pyplot as plt

# =========================
# 1. PCA PLOT
# =========================

def plot_pca(components, labels=None):
    plt.figure()
    if labels is not None:
        scatter = plt.scatter(components[:, 0], components[:, 1], c=labels)
        plt.legend(*scatter.legend_elements())
    else:
        plt.scatter(components[:, 0], components[:, 1])
    plt.xlabel("PC1")
    plt.ylabel("PC2")
    plt.title("PCA Plot")
    plt.show()

# =========================
# 2. CORRELATION HEATMAP
# =========================

def plot_correlation_matrix(corr_matrix):
    plt.figure()
    plt.imshow(corr_matrix)
    plt.colorbar()
    plt.title("Correlation Heatmap")
    plt.show()

# =========================
# 3. PREDICTION VS OBSERVED
# =========================

def plot_predictions(y_true, y_pred):
    plt.figure()
    plt.scatter(y_true, y_pred)
    plt.xlabel("Observed")
    plt.ylabel("Predicted")
    plt.title("Observed vs Predicted")
    plt.show()

# =========================
# 4. RESIDUAL PLOT
# =========================

def plot_residuals(y_true, y_pred):
    residuals = y_true - y_pred
    plt.figure()
    plt.scatter(y_pred, residuals)
    plt.axhline(0)
    plt.xlabel("Predicted")
    plt.ylabel("Residuals")
    plt.title("Residual Plot")
    plt.show()

# =========================
# 5. FEATURE IMPORTANCE
# =========================

def plot_feature_importance(importances):
    plt.figure()
    plt.bar(range(len(importances)), importances)
    plt.title("Feature Importance")
    plt.show()

# =========================
# 6. GRM HEATMAP
# =========================

def plot_grm(G):
    plt.figure()
    plt.imshow(G)
    plt.colorbar()
    plt.title("Genomic Relationship Matrix")
    plt.show()

# =========================
# 7. TRAINING HISTORY
# =========================

def plot_training_history(history):
    plt.figure()
    plt.plot(history.history['loss'], label='loss')
    if 'val_loss' in history.history:
        plt.plot(history.history['val_loss'], label='val_loss')
    plt.legend()
    plt.title("Training History")
    plt.show()

# =========================
# 8. DISTRIBUTION PLOT
# =========================

def plot_distribution(data):
    plt.figure()
    plt.hist(data, bins=20)
    plt.title("Distribution")
    plt.show()

# =========================
# 9. BOXPLOT
# =========================

def plot_boxplot(data):
    plt.figure()
    plt.boxplot(data)
    plt.title("Boxplot")
    plt.show()

# =========================
# 10. PIPELINE VISUALIZATION
# =========================

def visualization_pipeline(y_true, y_pred):
    plot_predictions(y_true, y_pred)
    plot_residuals(y_true, y_pred)
    plot_distribution(y_true)

# =========================
# END MODULE
# =========================
