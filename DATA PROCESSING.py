"""
Data Preprocessing Module for AI-Assisted Genomic Selection
===========================================================

This module handles:
- Loading genotypic and phenotypic data
- Data cleaning
- Missing value imputation
- Encoding SNP markers
- Normalization and scaling
- Train-test splitting
- PCA preparation

NOTE: This is an extended, research-grade implementation.
"""

import os
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.impute import SimpleImputer, KNNImputer
from sklearn.model_selection import train_test_split

# =========================
# 1. DATA LOADING
# =========================

def load_csv_data(file_path):
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"File not found: {file_path}")
    df = pd.read_csv(file_path)
    return df


def load_excel_data(file_path, sheet_name=0):
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"File not found: {file_path}")
    df = pd.read_excel(file_path, sheet_name=sheet_name)
    return df


# =========================
# 2. BASIC CLEANING
# =========================

def remove_duplicates(df):
    return df.drop_duplicates()


def remove_constant_columns(df):
    return df.loc[:, df.nunique() > 1]


def rename_columns(df):
    df.columns = [col.strip().replace(" ", "_") for col in df.columns]
    return df


# =========================
# 3. MISSING VALUE HANDLING
# =========================

def impute_mean(df):
    imputer = SimpleImputer(strategy='mean')
    return pd.DataFrame(imputer.fit_transform(df), columns=df.columns)


def impute_median(df):
    imputer = SimpleImputer(strategy='median')
    return pd.DataFrame(imputer.fit_transform(df), columns=df.columns)


def impute_mode(df):
    imputer = SimpleImputer(strategy='most_frequent')
    return pd.DataFrame(imputer.fit_transform(df), columns=df.columns)


def impute_knn(df, n_neighbors=5):
    imputer = KNNImputer(n_neighbors=n_neighbors)
    return pd.DataFrame(imputer.fit_transform(df), columns=df.columns)


# =========================
# 4. SNP ENCODING
# =========================

def encode_snp_additive(df):
    mapping = {
        'AA': 0,
        'AB': 1,
        'BB': 2
    }
    return df.replace(mapping)


def encode_snp_dominant(df):
    mapping = {
        'AA': 0,
        'AB': 1,
        'BB': 1
    }
    return df.replace(mapping)


def encode_snp_recessive(df):
    mapping = {
        'AA': 0,
        'AB': 0,
        'BB': 1
    }
    return df.replace(mapping)


# =========================
# 5. NORMALIZATION
# =========================

def standardize_data(df):
    scaler = StandardScaler()
    scaled = scaler.fit_transform(df)
    return pd.DataFrame(scaled, columns=df.columns)


def minmax_scale_data(df):
    scaler = MinMaxScaler()
    scaled = scaler.fit_transform(df)
    return pd.DataFrame(scaled, columns=df.columns)


# =========================
# 6. OUTLIER HANDLING
# =========================

def remove_outliers_zscore(df, threshold=3):
    z_scores = np.abs((df - df.mean()) / df.std())
    return df[(z_scores < threshold).all(axis=1)]


# =========================
# 7. FEATURE SEPARATION
# =========================

def split_features_target(df, target_column):
    X = df.drop(columns=[target_column])
    y = df[target_column]
    return X, y


# =========================
# 8. TRAIN TEST SPLIT
# =========================

def split_train_test(X, y, test_size=0.2, random_state=42):
    return train_test_split(X, y, test_size=test_size, random_state=random_state)


# =========================
# 9. CORRELATION FILTER
# =========================

def remove_highly_correlated(df, threshold=0.9):
    corr_matrix = df.corr().abs()
    upper = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(bool))
    to_drop = [column for column in upper.columns if any(upper[column] > threshold)]
    return df.drop(columns=to_drop)


# =========================
# 10. PIPELINE FUNCTION
# =========================

def full_preprocessing_pipeline(file_path, target_column):
    df = load_csv_data(file_path)
    df = rename_columns(df)
    df = remove_duplicates(df)
    df = remove_constant_columns(df)
    df = impute_mean(df)
    df = standardize_data(df)
    X, y = split_features_target(df, target_column)
    X = remove_highly_correlated(X)
    X_train, X_test, y_train, y_test = split_train_test(X, y)

    return X_train, X_test, y_train, y_test


# =========================
# 11. DEBUGGING UTILITIES
# =========================

def summarize_data(df):
    print("Shape:", df.shape)
    print("\nMissing Values:\n", df.isnull().sum())
    print("\nData Types:\n", df.dtypes)
    print("\nSummary Stats:\n", df.describe())


# =========================
# 12. ADVANCED FUNCTIONS (PLACEHOLDERS FOR EXTENSION)
# =========================

def genotype_quality_filter(df, missing_threshold=0.1):
    return df.loc[:, df.isnull().mean() < missing_threshold]


def minor_allele_frequency_filter(df, threshold=0.05):
    maf = df.apply(lambda col: min(col.mean(), 1 - col.mean()))
    return df.loc[:, maf > threshold]


def ld_pruning_placeholder(df):
    # Linkage disequilibrium pruning (to be implemented)
    return df


# =========================
# END OF MODULE
# =========================
