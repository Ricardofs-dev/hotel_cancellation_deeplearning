# =============================================================================
# models.py
# Definição das arquiteturas FNN e CNN usadas no projeto.
# =============================================================================

from tensorflow import keras
from tensorflow.keras import layers


def _compile(model: keras.Sequential) -> keras.Sequential:
    """Compila o modelo para classificação binária."""
    model.compile(
        optimizer="adam",
        loss="binary_crossentropy",
        metrics=["accuracy"]
    )
    return model


def build_fnn_baseline(input_dim: int) -> keras.Sequential:
    """FNN baseline com uma única camada de saída."""
    model = keras.Sequential([
        layers.Input(shape=(input_dim,)),
        layers.Dense(1, activation="sigmoid"),
    ], name="fnn_baseline")
    return _compile(model)


def build_fnn_dense64_32(input_dim: int) -> keras.Sequential:
    """FNN intermédia com duas camadas ocultas."""
    model = keras.Sequential([
        layers.Input(shape=(input_dim,)),
        layers.Dense(64, activation="relu"),
        layers.Dense(32, activation="relu"),
        layers.Dense(1, activation="sigmoid"),
    ], name="fnn_dense64_32")
    return _compile(model)


def build_fnn_final(input_dim: int, dropout: float = 0.5) -> keras.Sequential:
    """FNN final com dropout entre as camadas ocultas."""
    model = keras.Sequential([
        layers.Input(shape=(input_dim,)),
        layers.Dense(64, activation="relu"),
        layers.Dropout(dropout),
        layers.Dense(32, activation="relu"),
        layers.Dropout(dropout),
        layers.Dense(1, activation="sigmoid"),
    ], name="fnn_final")
    return _compile(model)


def build_cnn_baseline(input_dim: int) -> keras.Sequential:
    """CNN baseline com uma camada convolucional."""
    model = keras.Sequential([
        layers.Input(shape=(input_dim, 1)),
        layers.Conv1D(filters=32, kernel_size=3, activation="relu"),
        layers.MaxPooling1D(pool_size=2),
        layers.Flatten(),
        layers.Dense(1, activation="sigmoid"),
    ], name="cnn_baseline")
    return _compile(model)


def build_cnn_final(input_dim: int, dropout: float = 0.5) -> keras.Sequential:
    """CNN final com duas camadas convolucionais e dropout."""
    model = keras.Sequential([
        layers.Input(shape=(input_dim, 1)),
        layers.Conv1D(filters=32, kernel_size=3, activation="relu"),
        layers.Conv1D(filters=64, kernel_size=3, activation="relu"),
        layers.MaxPooling1D(pool_size=2),
        layers.Flatten(),
        layers.Dense(32, activation="relu"),
        layers.Dropout(dropout),
        layers.Dense(1, activation="sigmoid"),
    ], name="cnn_final")
    return _compile(model)