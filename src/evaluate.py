# =============================================================================
# evaluate.py
# Avaliação, visualização e comparação de modelos.
# =============================================================================

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.metrics import classification_report, confusion_matrix
from tensorflow import keras


def evaluate_model(
    model: keras.Sequential,
    X_train: np.ndarray,
    y_train,
    X_val: np.ndarray,
    y_val,
    X_test: np.ndarray,
    y_test,
) -> dict:
    """Avalia o modelo em treino, validação e teste."""
    train_loss, train_acc = model.evaluate(X_train, y_train, verbose=0)
    val_loss, val_acc = model.evaluate(X_val, y_val, verbose=0)
    test_loss, test_acc = model.evaluate(X_test, y_test, verbose=0)

    print(f"Train Accuracy      : {train_acc:.4f}  (loss: {train_loss:.4f})")
    print(f"Validation Accuracy : {val_acc:.4f}  (loss: {val_loss:.4f})")
    print(f"Test Accuracy       : {test_acc:.4f}  (loss: {test_loss:.4f})")

    return {
        "train_acc": train_acc,
        "val_acc": val_acc,
        "test_acc": test_acc,
        "train_loss": train_loss,
        "val_loss": val_loss,
        "test_loss": test_loss,
    }


def plot_history(history: keras.callbacks.History, title: str = "Model") -> None:
    """Plota accuracy e loss ao longo das épocas."""
    fig, axes = plt.subplots(1, 2, figsize=(13, 4))

    for ax, metric in zip(axes, ["accuracy", "loss"]):
        ax.plot(history.history[metric], label="Train", linewidth=1.8)
        ax.plot(history.history[f"val_{metric}"], label="Validation", linewidth=1.8)
        ax.set_title(f"{title} — {metric.capitalize()}")
        ax.set_xlabel("Epoch")
        ax.set_ylabel(metric.capitalize())
        ax.legend()
        ax.grid(alpha=0.3)

    plt.tight_layout()
    plt.show()


def plot_confusion_matrix(
    model: keras.Sequential,
    X_test: np.ndarray,
    y_test,
    title: str = "Model",
    threshold: float = 0.5,
) -> None:
    """Mostra a matriz de confusão e o classification report."""
    y_pred_prob = model.predict(X_test, verbose=0).ravel()
    y_pred = (y_pred_prob >= threshold).astype(int)
    y_true = np.array(y_test)

    print(f"\n=== Classification Report — {title} ===")
    print(classification_report(
        y_true, y_pred,
        target_names=["Not Canceled (0)", "Canceled (1)"]
    ))

    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(5, 4))
    sns.heatmap(
        cm,
        annot=True,
        fmt="d",
        cmap="Blues",
        xticklabels=["Not Canceled", "Canceled"],
        yticklabels=["Not Canceled", "Canceled"],
    )
    plt.title(f"Confusion Matrix — {title}")
    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    plt.tight_layout()
    plt.show()


def compare_models(results: dict) -> pd.DataFrame:
    """Compara os resultados dos modelos numa tabela."""
    df = pd.DataFrame(results).T[["train_acc", "val_acc", "test_acc"]]
    df.columns = ["Train Acc", "Val Acc", "Test Acc"]
    return df.map(lambda x: f"{x:.4f}")


def reshape_for_cnn(X: np.ndarray) -> np.ndarray:
    """Converte X para o formato esperado por Conv1D."""
    return X.reshape(X.shape[0], X.shape[1], 1)