# =============================================================================
# preprocessing.py
# Limpeza, divisão temporal e pipeline de encoding/scaling.
# =============================================================================

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from .config import (
    TARGET, CATEGORICAL_COLS, NUMERIC_COLS,
    COLS_TO_DROP, AUX_COLS,
    TRAIN_END, VAL_START, VAL_END, TEST_START,
)

# ── Limpeza ──────────────────────────────────────────────────────────────────

def clean_data(df: pd.DataFrame) -> pd.DataFrame:
    """Remove colunas excluídas e trata missing values principais."""
    df = df.copy()

    # Remover colunas com leakage ou missing excessivo
    cols_present = [c for c in COLS_TO_DROP if c in df.columns]
    df = df.drop(columns=cols_present)

    # Imputar missing values
    df["children"] = df["children"].fillna(0)
    df["country"]  = df["country"].fillna("Unknown")
    df["agent"]    = df["agent"].fillna("No_Agent")

    print(f"Dataset após limpeza: {df.shape[0]:,} observações, {df.shape[1]} colunas.")
    print(f"Missing values restantes: {df.isnull().sum().sum()}")

    return df

# ── Divisão temporal ─────────────────────────────────────────────────────────

def split_data(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Divide o dataset em treino, validação e teste com base em booking_date.

    | Conjunto   | Período                       | Proporção aprox. |
    |------------|-------------------------------|-----------------|
    | Treino     | até 2016-06-30                | ~60%            |
    | Validação  | 2016-07-01 a 2016-12-31       | ~20%            |
    | Teste      | a partir de 2017-01-01        | ~20%            |

    """
    train_df = df[df["booking_date"] <= TRAIN_END].copy()
    val_df   = df[(df["booking_date"] >= VAL_START) &
                  (df["booking_date"] <= VAL_END)].copy()
    test_df  = df[df["booking_date"] >= TEST_START].copy()

    total = len(df)
    for name, split in [("Train", train_df), ("Validation", val_df), ("Test", test_df)]:
        pct = len(split) / total * 100
        cancel_rate = split[TARGET].mean() * 100
        print(f"  {name:12s}: {len(split):6,} obs ({pct:.1f}%)  |  "
              f"cancel rate: {cancel_rate:.1f}%  |  "
              f"{split['booking_date'].min().date()} → {split['booking_date'].max().date()}")

    return train_df, val_df, test_df

# ── Pré-processamento ─────────────────────────────────────────────────────────

def build_preprocessor() -> ColumnTransformer:
    """Cria o pipeline com StandardScaler e OneHotEncoder."""
    return ColumnTransformer(transformers=[
        ("num", StandardScaler(), NUMERIC_COLS),
        ("cat", OneHotEncoder(handle_unknown="ignore"), CATEGORICAL_COLS),
    ])


def prepare_splits(
    train_df: pd.DataFrame,
    val_df:   pd.DataFrame,
    test_df:  pd.DataFrame,
) -> tuple:
    """Aplica o preprocessing e devolve os conjuntos prontos para modelação."""
    exclude = [TARGET] + AUX_COLS

    X_train = train_df.drop(columns=[c for c in exclude if c in train_df.columns]).copy()
    X_val   = val_df.drop(columns=[c for c in exclude if c in val_df.columns]).copy()
    X_test  = test_df.drop(columns=[c for c in exclude if c in test_df.columns]).copy()

    y_train = train_df[TARGET].copy()
    y_val   = val_df[TARGET].copy()
    y_test  = test_df[TARGET].copy()

    # Garantir que colunas categóricas são strings (evita erros no OHE)
    for col in CATEGORICAL_COLS:
        X_train[col] = X_train[col].astype(str)
        X_val[col]   = X_val[col].astype(str)
        X_test[col]  = X_test[col].astype(str)

    # Fit apenas no treino
    preprocessor = build_preprocessor()
    X_train_p = preprocessor.fit_transform(X_train).toarray()
    X_val_p   = preprocessor.transform(X_val).toarray()
    X_test_p  = preprocessor.transform(X_test).toarray()

    print(f"X_train: {X_train_p.shape}  |  X_val: {X_val_p.shape}  |  X_test: {X_test_p.shape}")

    return X_train_p, X_val_p, X_test_p, y_train, y_val, y_test, preprocessor
