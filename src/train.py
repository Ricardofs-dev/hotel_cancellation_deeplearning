# =============================================================================
# train.py
# Funções de treino dos modelos.
# =============================================================================

import numpy as np
from tensorflow import keras
from tensorflow.keras import callbacks


def train_model(
    model: keras.Sequential,
    X_train: np.ndarray,
    y_train,
    X_val: np.ndarray,
    y_val,
    epochs: int = 20,
    batch_size: int = 32,
    patience: int = 5,
    verbose: int = 1,
) -> keras.callbacks.History:
    """Treina o modelo com EarlyStopping."""
    early_stop = callbacks.EarlyStopping(
        monitor="val_loss",
        patience=patience,
        restore_best_weights=True,
        verbose=1,
    )

    history = model.fit(
        X_train, y_train,
        validation_data=(X_val, y_val),
        epochs=epochs,
        batch_size=batch_size,
        callbacks=[early_stop],
        verbose=verbose,
    )

    return history


def train_model_no_early_stop(
    model: keras.Sequential,
    X_train: np.ndarray,
    y_train,
    X_val: np.ndarray,
    y_val,
    epochs: int = 20,
    batch_size: int = 32,
    verbose: int = 1,
) -> keras.callbacks.History:
    """Treina o modelo sem EarlyStopping."""
    history = model.fit(
        X_train, y_train,
        validation_data=(X_val, y_val),
        epochs=epochs,
        batch_size=batch_size,
        verbose=verbose,
    )

    return history
