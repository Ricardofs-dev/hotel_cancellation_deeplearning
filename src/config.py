# =============================================================================
# config.py
# Configurações globais do projeto: colunas, splits, paths.
# Alterar aqui propaga-se a todos os módulos que importam este ficheiro.
# =============================================================================

# ── Variável target ──────────────────────────────────────────────────────────
TARGET = "is_canceled"

# ── Path por defeito para os dados ───────────────────────────────────────────
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parents[1]
DEFAULT_DATA_PATH = BASE_DIR / "data" / "hotel_bookings.csv"

# ── Variáveis categóricas usadas no modelo ───────────────────────────────────
CATEGORICAL_COLS = [
    "hotel",
    "arrival_date_month",
    "meal",
    "country",
    "market_segment",
    "distribution_channel",
    "reserved_room_type",
    "deposit_type",
    "agent",
    "customer_type",
]

# ── Variáveis numéricas usadas no modelo ─────────────────────────────────────
NUMERIC_COLS = [
    "lead_time",
    "arrival_date_year",
    "arrival_date_week_number",
    "arrival_date_day_of_month",
    "stays_in_weekend_nights",
    "stays_in_week_nights",
    "adults",
    "children",
    "babies",
    "is_repeated_guest",
    "previous_cancellations",
    "previous_bookings_not_canceled",
    "adr",
    "required_car_parking_spaces",
    "total_of_special_requests",
]

# ── Colunas a remover por leakage ou missing excessivo ───────────────────────
COLS_TO_DROP = [
    "reservation_status",
    "reservation_status_date",
    "booking_changes",
    "days_in_waiting_list",
    "assigned_room_type",
    "company",
]

# ── Colunas auxiliares (criadas no preprocessing, excluídas das features) ────
AUX_COLS = [
    "arrival_date",
    "booking_date",
    "arrival_date_month_num",
]

# ── Fronteiras temporais da divisão train / val / test ───────────────────────
TRAIN_END  = "2016-06-30"
VAL_START  = "2016-07-01"
VAL_END    = "2016-12-31"
TEST_START = "2017-01-01"

# ── Mapeamento mês → número ──────────────────────────────────────────────────
MONTH_MAP = {
    "January": 1,  "February": 2,  "March": 3,    "April": 4,
    "May": 5,      "June": 6,      "July": 7,      "August": 8,
    "September": 9, "October": 10, "November": 11, "December": 12,
}
