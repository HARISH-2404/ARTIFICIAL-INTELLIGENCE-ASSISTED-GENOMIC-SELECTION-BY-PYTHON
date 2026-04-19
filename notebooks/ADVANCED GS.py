# ============================================================
# ADVANCED GENOMIC SELECTION (GBLUP)
# ============================================================

import pandas as pd
import numpy as np
from sklearn.metrics import r2_score
from sklearn.model_selection import train_test_split

# =========================
# LOAD DATA
# =========================
def load_data():
    geno = pd.read_csv("data/raw/genotype_matrix_sample.csv")
    pheno = pd.read_csv("data/raw/phenotype_traits_sample.csv")
    
    data = pd.merge(pheno, geno, on="Genotype")
    return data

# =========================
# CREATE G MATRIX
# =========================
def compute_g_matrix(X):
    """
    X = genotype matrix (0,1,2 coding)
    """
    # Center markers
    p = np.mean(X, axis=0) / 2
    Z = X - 2 * p
    
    # G matrix
    G = np.dot(Z, Z.T) / (2 * np.sum(p * (1 - p)))
    
    return G

# =========================
# GBLUP MODEL
# =========================
def gblup(data, target_trait):
    
    # Extract phenotype
    y = data[target_trait].values
    
    # Extract genotype
    X = data.drop(columns=["Genotype", target_trait]).values
    
    # Train-test split
    train_idx, test_idx = train_test_split(
        np.arange(len(y)), test_size=0.3, random_state=42
    )
    
    y_train = y[train_idx]
    y_test = y[test_idx]
    
    X_train = X[train_idx]
    X_test = X[test_idx]
    
    # Compute G matrix
    G = compute_g_matrix(X)
    
    # Subset G
    G_train = G[np.ix_(train_idx, train_idx)]
    G_test = G[np.ix_(test_idx, train_idx)]
    
    # Regularization (lambda)
    lam = 0.1
    
    # Solve mixed model equations
    alpha = np.linalg.inv(G_train + lam * np.eye(G_train.shape[0])).dot(y_train)
    
    # Predict
    y_pred = G_test.dot(alpha)
    
    # Accuracy
    acc = r2_score(y_test, y_pred)
    
    return acc, y_pred

# =========================
# RUN MODEL
# =========================
def run_gblup():
    data = load_data()
    
    trait = "Yield_per_Plant"
    
    acc, pred = gblup(data, trait)
    
    print(f"GBLUP Accuracy (R²): {acc:.3f}")

# =========================
# EXECUTE
# =========================
if __name__ == "__main__":
    run_gblup()
