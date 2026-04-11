# =============================================================================
# eda.py
# Funções de análise exploratória usadas no notebook final.
# =============================================================================

import pandas as pd
import matplotlib.pyplot as plt


def dataset_overview(df: pd.DataFrame) -> None:
    """Visão geral do dataset: shape, missing values, duplicados e distribuição da target."""
    print(f"Shape: {df.shape}")
    missing = df.isnull().sum()
    missing = missing[missing > 0]
    print(f"\nMissing values:\n{missing}" if not missing.empty else "\nMissing values: nenhum")
    print(f"\nDuplicados exatos: {df.duplicated().sum():,}")
    print(f"\nDistribuição da target (is_canceled):\n"
          f"{df['is_canceled'].value_counts(normalize=True).round(3)}")


def plot_missing(df: pd.DataFrame) -> None:
    """Gráfico de barras com a percentagem de missing values por coluna."""
    missing = df.isnull().sum()
    missing = missing[missing > 0].sort_values(ascending=False)
    if missing.empty:
        print("Sem missing values.")
        return
    (missing / len(df) * 100).plot(kind="bar", figsize=(8, 4), color="steelblue")
    plt.ylabel("% missing")
    plt.title("Percentage of Missing Values by Column")
    plt.tight_layout()
    plt.show()


def plot_target(df: pd.DataFrame) -> None:
    """Distribuição da variável target is_canceled."""
    counts = df["is_canceled"].value_counts().sort_index()
    proportions = (df["is_canceled"].value_counts(normalize=True).sort_index() * 100).round(2)
    print("Distribuição (contagem):\n", counts)
    print("\nDistribuição (%):\n", proportions)
    counts.plot(kind="bar", figsize=(6, 4))
    plt.title("Distribution of is_canceled")
    plt.xlabel("is_canceled")
    plt.ylabel("Count")
    plt.xticks(rotation=0)
    plt.tight_layout()
    plt.show()


def analyze_duplicates(df: pd.DataFrame) -> None:
    """Contagem de duplicados e impacto na distribuição da target."""
    n_dup = df.duplicated().sum()
    print(f"Duplicados exatos : {n_dup:,}  ({n_dup/len(df)*100:.1f}%)")
    print("\nDistribuição target — com duplicados:")
    print(df["is_canceled"].value_counts(normalize=True).round(4))
    print("\nDistribuição target — sem duplicados:")
    print(df.drop_duplicates()["is_canceled"].value_counts(normalize=True).round(4))


def plot_target_correlation(df: pd.DataFrame) -> pd.Series:
    """Correlação de Pearson entre variáveis numéricas e a target."""
    num_cols = df.select_dtypes(include="number").columns.tolist()
    corr = (df[num_cols].corr()["is_canceled"]
            .drop("is_canceled")
            .sort_values(key=abs, ascending=True))
    colors = ["coral" if x < 0 else "steelblue" for x in corr]
    fig, ax = plt.subplots(figsize=(8, 6))
    corr.plot(kind="barh", ax=ax, color=colors, edgecolor="black")
    ax.axvline(0, color="black", linewidth=0.8)
    ax.set_title("Correlação das features numéricas com is_canceled")
    ax.set_xlabel("Pearson correlation")
    plt.tight_layout()
    plt.show()
    return corr.sort_values(key=abs, ascending=False)
