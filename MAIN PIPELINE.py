"""
Main Pipeline Runner
====================

End-to-end execution for AI-Assisted Genomic Selection:
1. Load data
2. Preprocess
3. Feature engineering
4. Train model
5. Evaluate
6. Save model + artifacts

Customize paths and parameters as needed.
"""

import numpy as np

# Import your modules
from src.data_preprocessing import full_preprocessing_pipeline
from src.feature_engineering import full_feature_engineering_pipeline
from src.genomic_selection_models import rr_blup
from src.evaluation import evaluation_pipeline
from src.model_utils import save_full_pipeline

# =========================
# CONFIGURATION
# =========================
DATA_PATH = "data/raw/your_dataset.csv"   # <-- update this
TARGET_COLUMN = "trait"                  # <-- update this

# =========================
# MAIN PIPELINE
# =========================

def run_pipeline():
    print("\n[1] Loading & Preprocessing Data...")
    X_train, X_test, y_train, y_test = full_preprocessing_pipeline(
        DATA_PATH, TARGET_COLUMN
    )

    print("[2] Feature Engineering...")
    X_train_fe, _ = full_feature_engineering_pipeline(X_train.values, y_train.values)
    X_test_fe, _ = full_feature_engineering_pipeline(X_test.values, y_test.values)

    print("[3] Training Model (RR-BLUP)...")
    model, train_preds = rr_blup(X_train_fe, y_train.values)

    print("[4] Evaluating Model...")
    results = evaluation_pipeline(lambda X, y: rr_blup(X, y), X_test_fe, y_test.values)
    print("Evaluation Results:", results)

    print("[5] Saving Model...")
    saved_files = save_full_pipeline(
        model=model,
        scaler=None,
        pca=None,
        base_path="models",
        model_name="rr_blup_model"
    )

    print("Saved Files:", saved_files)
    print("\nPipeline completed successfully!")

# =========================
# ENTRY POINT
# =========================

if __name__ == "__main__":
    run_pipeline()
