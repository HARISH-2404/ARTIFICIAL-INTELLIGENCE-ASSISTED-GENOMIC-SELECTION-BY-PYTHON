import pytest
import numpy as np
from src.data_preprocessing import (
    normalize_data,
    standardize_data,
    handle_missing_values,
    encode_categorical,
    split_data
)

# ---------------------------
# FIXTURES
# ---------------------------

@pytest.fixture
def sample_numeric_data():
    return np.array([
        [1.0, 2.0, 3.0],
        [4.0, 5.0, 6.0],
        [7.0, 8.0, 9.0]
    ])

@pytest.fixture
def sample_missing_data():
    return np.array([
        [1.0, np.nan, 3.0],
        [4.0, 5.0, np.nan],
        [np.nan, 8.0, 9.0]
    ])

@pytest.fixture
def sample_labels():
    return np.array([0, 1, 0])

# ---------------------------
# NORMALIZATION TESTS
# ---------------------------

def test_normalize_range(sample_numeric_data):
    norm = normalize_data(sample_numeric_data)

    assert np.min(norm) >= 0.0
    assert np.max(norm) <= 1.0

def test_normalize_shape(sample_numeric_data):
    norm = normalize_data(sample_numeric_data)
    assert norm.shape == sample_numeric_data.shape

def test_normalize_constant_input():
    data = np.ones((3, 3))
    norm = normalize_data(data)
    assert np.all(norm == 0.0 or np.all(np.isnan(norm)) is False)

# ---------------------------
# STANDARDIZATION TESTS
# ---------------------------

def test_standardize_mean(sample_numeric_data):
    std = standardize_data(sample_numeric_data)
    assert np.allclose(np.mean(std, axis=0), 0, atol=1e-7)

def test_standardize_std(sample_numeric_data):
    std = standardize_data(sample_numeric_data)
    assert np.allclose(np.std(std, axis=0), 1, atol=1e-7)

# ---------------------------
# MISSING VALUE HANDLING
# ---------------------------

def test_missing_value_imputation(sample_missing_data):
    filled = handle_missing_values(sample_missing_data)

    assert not np.isnan(filled).any()

def test_missing_value_shape(sample_missing_data):
    filled = handle_missing_values(sample_missing_data)
    assert filled.shape == sample_missing_data.shape

# ---------------------------
# CATEGORICAL ENCODING
# ---------------------------

def test_label_encoding():
    data = ["A", "B", "A", "C"]
    encoded = encode_categorical(data)

    assert len(encoded) == len(data)
    assert set(encoded) <= set([0, 1, 2])

# ---------------------------
# TRAIN TEST SPLIT
# ---------------------------

def test_split_ratio(sample_numeric_data, sample_labels):
    X_train, X_test, y_train, y_test = split_data(
        sample_numeric_data,
        sample_labels,
        test_size=0.3,
        random_state=42
    )

    total = len(sample_labels)
    assert len(X_train) + len(X_test) == total
    assert len(y_train) + len(y_test) == total

def test_split_reproducibility(sample_numeric_data, sample_labels):
    split1 = split_data(sample_numeric_data, sample_labels, 0.3, 42)
    split2 = split_data(sample_numeric_data, sample_labels, 0.3, 42)

    assert np.array_equal(split1[0], split2[0])
    assert np.array_equal(split1[2], split2[2])

# ---------------------------
# EDGE CASE TESTS
# ---------------------------

def test_empty_input():
    with pytest.raises(Exception):
        normalize_data(np.array([]))

def test_single_row():
    data = np.array([[5.0, 10.0, 15.0]])
    norm = normalize_data(data)
    assert norm.shape == data.shape

def test_large_values():
    data = np.array([[1e9, 2e9], [3e9, 4e9]])
    norm = normalize_data(data)
    assert np.isfinite(norm).all()

# ---------------------------
# INTEGRATION STYLE TEST
# ---------------------------

def test_full_pipeline(sample_numeric_data, sample_labels):
    cleaned = handle_missing_values(sample_numeric_data)
    norm = normalize_data(cleaned)
    std = standardize_data(norm)

    X_train, X_test, y_train, y_test = split_data(std, sample_labels, 0.2, 42)

    assert X_train.shape[0] > 0
    assert X_test.shape[0] > 0
