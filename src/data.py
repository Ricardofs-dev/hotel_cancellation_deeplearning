# =============================================================================
# data.py
# Carregamento e transformações iniciais do dataset.
# =============================================================================

import pandas as pd
from .config import DEFAULT_DATA_PATH, MONTH_MAP


def load_data(path: str = DEFAULT_DATA_PATH) -> pd.DataFrame:
    """ Carrega o dataset a partir de um ficheiro CSV."""

    df = pd.read_csv(path)
    print(f"Dataset carregado: {df.shape[0]:,} observações, {df.shape[1]} colunas.")
    return df


def reconstruct_booking_date(df: pd.DataFrame) -> pd.DataFrame:
    """ 
    Reconstrói a booking_date a partir da data de chegada e do lead_time.
    
    Cria três colunas auxiliares:
    - arrival_date_month_num : mês como inteiro
    - arrival_date           : data de chegada como datetime
    - booking_date           : data estimada da reserva (arrival_date - lead_time)

   """
    df = df.copy()

    df["arrival_date_month_num"] = df["arrival_date_month"].map(MONTH_MAP)

    df["arrival_date"] = pd.to_datetime(dict(
        year=df["arrival_date_year"],
        month=df["arrival_date_month_num"],
        day=df["arrival_date_day_of_month"]
    ))

    df["booking_date"] = (
        df["arrival_date"] - pd.to_timedelta(df["lead_time"], unit="D")
    )

    return df
