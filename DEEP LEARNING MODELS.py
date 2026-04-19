"""
Deep Learning Models for AI-Assisted Genomic Selection
=====================================================

This module implements advanced deep learning architectures:
- ANN (Fully Connected Neural Network)
- CNN for SNP data
- Autoencoders for dimensionality reduction
- Multi-task learning for multi-trait prediction
- Training utilities, callbacks, and evaluation

NOTE: This is an extended, research-oriented implementation.
"""

import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Model, Sequential
from tensorflow.keras.layers import Dense, Dropout, Conv1D, Flatten, Input, BatchNormalization
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping

# =========================
# 1. BASIC ANN MODEL
# =========================

def build_ann(input_dim):
    model = Sequential([
        Dense(256, activation='relu', input_shape=(input_dim,)),
        BatchNormalization(),
        Dropout(0.3),
        Dense(128, activation='relu'),
        Dropout(0.3),
        Dense(64, activation='relu'),
        Dense(1)
    ])

    model.compile(optimizer=Adam(learning_rate=0.001), loss='mse')
    return model


def train_ann(model, X, y, epochs=100, batch_size=32):
    early_stop = EarlyStopping(patience=10, restore_best_weights=True)
    history = model.fit(X, y, epochs=epochs, batch_size=batch_size,
                        validation_split=0.2, callbacks=[early_stop], verbose=0)
    return model, history

# =========================
# 2. CNN FOR SNP DATA
# =========================

def build_cnn(input_length):
    model = Sequential([
        Conv1D(filters=64, kernel_size=3, activation='relu', input_shape=(input_length,1)),
        BatchNormalization(),
        Conv1D(filters=128, kernel_size=3, activation='relu'),
        Flatten(),
        Dense(128, activation='relu'),
        Dropout(0.3),
        Dense(1)
    ])

    model.compile(optimizer=Adam(0.001), loss='mse')
    return model


def reshape_for_cnn(X):
    return X.reshape((X.shape[0], X.shape[1], 1))

# =========================
# 3. AUTOENCODER
# =========================

def build_autoencoder(input_dim, encoding_dim=50):
    input_layer = Input(shape=(input_dim,))

    encoded = Dense(128, activation='relu')(input_layer)
    encoded = Dense(encoding_dim, activation='relu')(encoded)

    decoded = Dense(128, activation='relu')(encoded)
    decoded = Dense(input_dim, activation='sigmoid')(decoded)

    autoencoder = Model(input_layer, decoded)
    encoder = Model(input_layer, encoded)

    autoencoder.compile(optimizer='adam', loss='mse')

    return autoencoder, encoder


def train_autoencoder(autoencoder, X, epochs=100):
    autoencoder.fit(X, X, epochs=epochs, batch_size=32, verbose=0)
    return autoencoder

# =========================
# 4. MULTI-TRAIT MODEL
# =========================

def build_multi_trait_model(input_dim, output_dim):
    inputs = Input(shape=(input_dim,))

    x = Dense(256, activation='relu')(inputs)
    x = Dropout(0.3)(x)
    x = Dense(128, activation='relu')(x)

    outputs = Dense(output_dim)(x)

    model = Model(inputs, outputs)
    model.compile(optimizer='adam', loss='mse')

    return model

# =========================
# 5. CUSTOM LOSS (OPTIONAL)
# =========================

def rmse_loss(y_true, y_pred):
    return tf.sqrt(tf.reduce_mean(tf.square(y_true - y_pred)))

# =========================
# 6. TRAINING WRAPPER
# =========================

def train_model(model, X, y, epochs=100, batch_size=32):
    history = model.fit(X, y, epochs=epochs, batch_size=batch_size,
                        validation_split=0.2, verbose=0)
    return model, history

# =========================
# 7. PREDICTION
# =========================

def predict(model, X):
    return model.predict(X)

# =========================
# 8. MODEL EVALUATION
# =========================

def evaluate_model(model, X, y):
    loss = model.evaluate(X, y, verbose=0)
    return loss

# =========================
# 9. HYPERPARAMETER LOOP (BASIC)
# =========================

def tune_ann(X, y):
    best_loss = float('inf')
    best_model = None

    for lr in [0.001, 0.0005]:
        model = build_ann(X.shape[1])
        model.compile(optimizer=Adam(lr), loss='mse')
        model, _ = train_ann(model, X, y)
        loss = evaluate_model(model, X, y)

        if loss < best_loss:
            best_loss = loss
            best_model = model

    return best_model

# =========================
# 10. PIPELINE
# =========================

def deep_learning_pipeline(X, y):
    model = build_ann(X.shape[1])
    model, history = train_ann(model, X, y)
    preds = predict(model, X)
    return model, preds, history

# =========================
# END MODULE
# =========================
