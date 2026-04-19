"""
AI-Genomic-Selection Package
============================

This package provides modules for:
- Data preprocessing
- Feature engineering (PCA, scaling)
- Statistical genomic selection models
- Machine learning models
- Deep learning models
- Evaluation metrics
- Visualization tools

Author: Harish GPB
Project: Artificial Intelligence Assisted Genomic Selection
"""

# =========================
# Package Metadata
# =========================
__version__ = "1.0.0"
__author__ = "Harish GPB"
__email__ = "your_email@example.com"

# =========================
# Import Core Modules
# =========================
from . import data_preprocessing
from . import feature_engineering
from . import statistical_models
from . import ml_models
from . import deep_learning_models
from . import evaluation
from . import visualization

# =========================
# Expose Key Functions
# =========================

# Data preprocessing
from .data_preprocessing import (
    load_data,
    handle_missing_values,
    normalize_data,
    encode_genotypes
)

# Feature engineering
from .feature_engineering import (
    perform_pca,
    compute_correlation_matrix
)

# Statistical models
from .statistical_models import (
    ridge_regression_gs,
    rr_blup
)

# Machine learning models
from .ml_models import (
    train_random_forest,
    train_svm,
    train_knn
)

# Deep learning models
from .deep_learning_models import (
    build_ann_model,
    train_ann_model
)

# Evaluation
from .evaluation import (
    calculate_rmse,
    calculate_r2_score,
    cross_validate_model
)

# Visualization
from .visualization import (
    plot_pca,
    plot_predictions,
    plot_feature_importance
)

# =========================
# Define Public API
# =========================
__all__ = [
    # Modules
    "data_preprocessing",
    "feature_engineering",
    "statistical_models",
    "ml_models",
    "deep_learning_models",
    "evaluation",
    "visualization",

    # Functions
    "load_data",
    "handle_missing_values",
    "normalize_data",
    "encode_genotypes",
    "perform_pca",
    "compute_correlation_matrix",
    "ridge_regression_gs",
    "rr_blup",
    "train_random_forest",
    "train_svm",
    "train_knn",
    "build_ann_model",
    "train_ann_model",
    "calculate_rmse",
    "calculate_r2_score",
    "cross_validate_model",
    "plot_pca",
    "plot_predictions",
    "plot_feature_importance"
]

# =========================
# Initialization Logic
# =========================
def initialize_package():
    """
    Initialize package settings.
    """
    print("AI-Genomic-Selection package initialized successfully.")


# Automatically initialize when imported
initialize_package()
