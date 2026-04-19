"""
Feature Engineering Module for AI-Assisted Genomic Selection
============================================================

This module includes advanced feature engineering techniques:
- PCA (manual + sklearn)
- Genomic Relationship Matrix (GRM)
- Feature selection
- Polynomial and interaction features
- Dimensionality reduction
- Statistical transformations
- Multi-trait feature construction

Designed for research-grade genomic selection pipelines.
"""

import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.preprocessing import PolynomialFeatures
from sklearn.feature_selection import VarianceThreshold
from sklearn.feature_selection import SelectKBest, f_regression

# =========================
# 1. PCA (SKLEARN)
# =========================

def perform_pca(X, n_components=2):
    pca = PCA(n_components=n_components)
    components = pca.fit_transform(X)
    explained_variance = pca.explained_variance_ratio_
    return components, explained_variance, pca


# =========================
# 2. PCA (MANUAL)
# =========================

def manual_pca(X):
    X_meaned = X - np.mean(X, axis=0)
    cov_matrix = np.cov(X_meaned, rowvar=False)
    eigenvalues, eigenvectors = np.linalg.eigh(cov_matrix)
    sorted_index = np.argsort(eigenvalues)[::-1]
    sorted_eigenvalues = eigenvalues[sorted_index]
    sorted_eigenvectors = eigenvectors[:, sorted_index]
    X_reduced = np.dot(X_meaned, sorted_eigenvectors)
    return X_reduced, sorted_eigenvalues, sorted_eigenvectors


# =========================
# 3. GENOMIC RELATIONSHIP MATRIX (GRM)
# =========================

def compute_grm(Z):
    Z = np.array(Z)
    p = np.mean(Z, axis=0) / 2
    Z_centered = Z - 2 * p
    denominator = 2 * np.sum(p * (1 - p))
    G = np.dot(Z_centered, Z_centered.T) / denominator
    return G


# =========================
# 4. VARIANCE FILTER
# =========================

def remove_low_variance_features(X, threshold=0.01):
    selector = VarianceThreshold(threshold=threshold)
    X_reduced = selector.fit_transform(X)
    return X_reduced


# =========================
# 5. UNIVARIATE FEATURE SELECTION
# =========================

def select_top_k_features(X, y, k=10):
    selector = SelectKBest(score_func=f_regression, k=k)
    X_new = selector.fit_transform(X, y)
    return X_new, selector


# =========================
# 6. POLYNOMIAL FEATURES
# =========================

def generate_polynomial_features(X, degree=2):
    poly = PolynomialFeatures(degree=degree, include_bias=False)
    X_poly = poly.fit_transform(X)
    return X_poly, poly


# =========================
# 7. INTERACTION FEATURES
# =========================

def generate_interaction_features(X):
    poly = PolynomialFeatures(degree=2, interaction_only=True, include_bias=False)
    X_interact = poly.fit_transform(X)
    return X_interact


# =========================
# 8. LOG TRANSFORMATION
# =========================

def log_transform(X):
    return np.log1p(X)


# =========================
# 9. STANDARDIZATION
# =========================

def z_score_standardization(X):
    return (X - np.mean(X, axis=0)) / np.std(X, axis=0)


# =========================
# 10. MULTI-TRAIT FEATURE CREATION
# =========================

def create_trait_interactions(df):
    cols = df.columns
    new_df = df.copy()
    for i in range(len(cols)):
        for j in range(i+1, len(cols)):
            new_col = f"{cols[i]}_x_{cols[j]}"
            new_df[new_col] = df[cols[i]] * df[cols[j]]
    return new_df


# =========================
# 11. FEATURE SCALING (MIN-MAX)
# =========================

def min_max_scale(X):
    return (X - np.min(X, axis=0)) / (np.max(X, axis=0) - np.min(X, axis=0))


# =========================
# 12. FEATURE NORMALIZATION (L2)
# =========================

def l2_normalization(X):
    norm = np.linalg.norm(X, axis=1, keepdims=True)
    return X / norm


# =========================
# 13. DIMENSION REDUCTION PIPELINE
# =========================

def dimensionality_reduction_pipeline(X, n_components=5):
    X_std = z_score_standardization(X)
    X_pca, var, _ = perform_pca(X_std, n_components)
    return X_pca, var


# =========================
# 14. FEATURE IMPORTANCE (CORRELATION)
# =========================

def compute_feature_correlation(X, y):
    corrs = []
    for i in range(X.shape[1]):
        corr = np.corrcoef(X[:, i], y)[0, 1]
        corrs.append(corr)
    return np.array(corrs)


# =========================
# 15. FEATURE RANKING
# =========================

def rank_features_by_correlation(X, y):
    corrs = compute_feature_correlation(X, y)
    return np.argsort(np.abs(corrs))[::-1]


# =========================
# 16. PIPELINE FUNCTION
# =========================

def full_feature_engineering_pipeline(X, y):
    X_var = remove_low_variance_features(X)
    X_poly, _ = generate_polynomial_features(X_var)
    X_selected, _ = select_top_k_features(X_poly, y, k=20)
    X_final, var = dimensionality_reduction_pipeline(X_selected, n_components=10)
    return X_final, var


# =========================
# 17. DEBUGGING
# =========================

def summarize_features(X):
    print("Shape:", X.shape)
    print("Mean:", np.mean(X, axis=0))
    print("Std:", np.std(X, axis=0))


# =========================
# END MODULE
# =========================
