import pytest
import numpy as np

from sklearn.datasets import make_classification
from sklearn.metrics import accuracy_score

from models.model_utils import build_model

# =========================================================
# 🔶 DATA GENERATION UTILITIES
# =========================================================

def generate_classification_data(n_samples=200, n_features=20, seed=42):
    X, y = make_classification(
        n_samples=n_samples,
        n_features=n_features,
        n_informative=10,
        n_redundant=5,
        random_state=seed
    )
    return X, y


def generate_snp_data(n_samples=200, n_markers=50):
    # Genomic-style SNP encoding: 0,1,2
    X = np.random.randint(0, 3, (n_samples, n_markers))
    y = np.random.randint(0, 2, n_samples)
    return X, y


# =========================================================
# 🔶 MODEL INITIALIZATION TESTS
# =========================================================

@pytest.mark.parametrize("model_type", [
    "random_forest",
    "svm",
    "logistic_regression"
])
def test_model_initialization(model_type):
    config = {"model": {"type": model_type}}
    model = build_model(config)
    assert model is not None


def test_invalid_model():
    config = {"model": {"type": "invalid_model"}}
    with pytest.raises(Exception):
        build_model(config)


# =========================================================
# 🔶 TRAINING TESTS
# =========================================================

def test_training_basic():
    X, y = generate_classification_data()

    config = {"model": {"type": "random_forest"}}
    model = build_model(config)

    model.fit(X, y)
    assert hasattr(model, "predict")


@pytest.mark.parametrize("seed", [1, 42, 99])
def test_training_reproducibility(seed):
    X, y = generate_classification_data(seed=seed)

    config = {"model": {"type": "random_forest", "random_state": seed}}

    model = build_model(config)
    model.fit(X, y)

    preds = model.predict(X)
    assert len(preds) == len(y)


# =========================================================
# 🔶 PREDICTION TESTS
# =========================================================

def test_prediction_shape():
    X, y = generate_classification_data()

    model = build_model({"model": {"type": "random_forest"}})
    model.fit(X, y)

    preds = model.predict(X)
    assert preds.shape[0] == X.shape[0]


def test_probability_output():
    X, y = generate_classification_data()

    model = build_model({"model": {"type": "random_forest"}})
    model.fit(X, y)

    if hasattr(model, "predict_proba"):
        probs = model.predict_proba(X)
        assert probs.shape[0] == X.shape[0]


# =========================================================
# 🔶 EDGE CASE TESTS
# =========================================================

def test_empty_input():
    model = build_model({"model": {"type": "random_forest"}})

    X = np.array([])
    y = np.array([])

    with pytest.raises(Exception):
        model.fit(X, y)


def test_nan_input():
    model = build_model({"model": {"type": "random_forest"}})

    X = np.array([[1, np.nan], [2, 3]])
    y = np.array([0, 1])

    with pytest.raises(Exception):
        model.fit(X, y)


# =========================================================
# 🔶 SCALABILITY TESTS
# =========================================================

def test_large_dataset():
    X, y = generate_classification_data(n_samples=5000, n_features=100)

    model = build_model({"model": {"type": "random_forest", "n_estimators": 50}})
    model.fit(X, y)

    preds = model.predict(X)
    assert preds is not None


# =========================================================
# 🔶 GENOMIC / SNP TESTS (IMPORTANT FOR YOUR RESEARCH)
# =========================================================

def test_snp_encoding():
    X, y = generate_snp_data()

    model = build_model({"model": {"type": "random_forest"}})
    model.fit(X, y)

    preds = model.predict(X)
    assert len(preds) == len(y)


def test_missing_genotype_values():
    X, y = generate_snp_data()

    X = X.astype(float)
    X[0, 0] = np.nan

    model = build_model({"model": {"type": "random_forest"}})

    with pytest.raises(Exception):
        model.fit(X, y)


# =========================================================
# 🔶 MODEL COMPARISON TESTS
# =========================================================

def test_model_comparison():
    X, y = generate_classification_data()

    models = [
        build_model({"model": {"type": "random_forest"}}),
        build_model({"model": {"type": "svm"}})
    ]

    scores = []

    for m in models:
        m.fit(X, y)
        preds = m.predict(X)
        scores.append(accuracy_score(y, preds))

    assert len(scores) == 2


# =========================================================
# 🔶 STABILITY TESTS
# =========================================================

def test_prediction_stability():
    X, y = generate_classification_data()

    config = {"model": {"type": "random_forest", "random_state": 42}}

    m1 = build_model(config)
    m2 = build_model(config)

    m1.fit(X, y)
    m2.fit(X, y)

    assert np.array_equal(m1.predict(X), m2.predict(X))


# =========================================================
# 🔶 BIOLOGICAL ROBUSTNESS TESTS
# =========================================================

def test_trait_like_structure():
    # simulate traits like plant height, yield, biomass
    X = np.random.rand(200, 10)
    y = np.random.randint(0, 2, 200)

    model = build_model({"model": {"type": "random_forest"}})
    model.fit(X, y)

    preds = model.predict(X)
    assert preds.shape[0] == 200
