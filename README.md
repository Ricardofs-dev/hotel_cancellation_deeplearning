# Hotel Booking Cancellation — Deep Learning

> Previsão de cancelamento de reservas hoteleiras com redes neuronais (FNN e CNN)  
> Pós-graduação em Data Science e Business Intelligence — Deep Learning — ISAG 2026

---

## Problema

As reservas canceladas representam um desafio operacional e financeiro significativo para a indústria hoteleira. Este projeto desenvolve e compara modelos de deep learning para prever, **no momento da reserva**, se essa reserva será cancelada.

- **Tipo de problema:** Classificação binária
- **Target:** `is_canceled` (0 = não cancelado · 1 = cancelado)
- **Métrica principal:** Accuracy. Na avaliação final: precision, recall e F1-score por classe como métricas complementares.

---

## Dataset

[Hotel Bookings](https://www.kaggle.com/datasets/mexwell/hotel-bookings) — Kaggle

| Característica | Valor |
|---|---|
| Observações | 119.390 |
| Variáveis originais | 32 |
| Variáveis no modelo | 25 (após exclusão de leakage) |
| Taxa de cancelamento | ~37% |
| Período | Julho 2015 — Agosto 2017 |

---

## Abordagem

O projeto segue a metodologia sugerida na unidade curricular:

1. Definição do problema e recolha de dados
2. Definição do objetivo e da métrica
3. Protocolo de avaliação — divisão temporal (train/val/test ≈ 60/20/20%)
4. Preparação dos dados — limpeza, remoção de leakage, OHE + StandardScaler
5. Baseline — FNN mínimo (`Dense(1, sigmoid)`)
6. Aumento de complexidade — arquiteturas mais profundas até overfitting
7. Regularização — Dropout + EarlyStopping
8. Utilidade do modelo — matriz de confusão, interpretação de negócio, exemplo simulado

Duas famílias de modelos foram desenvolvidas e comparadas:

- **FNN** (Feedforward Neural Network) — baseline principal, mais natural para dados tabulares
- **CNN** (Conv1D) — proposta pelo docente; avalia se convolution layers acrescentam valor preditivo neste contexto

---

## Resultados

| Modelo | Train Acc | Val Acc | Test Acc |
|---|---:|---:|---:|
| FNN Baseline — `Dense(1, sigmoid)` | 0.8574 | 0.7757 | 0.7719 |
| FNN Final ★ — `Dense(64) + Dropout(0.5) + Dense(32) + Dropout(0.5)` | **0.9002** | **0.7877** | **0.7852** |
| CNN Baseline — `Conv1D(32) + MaxPool + Flatten` | 0.8700 | 0.7597 | 0.7739 |
| CNN Final — `Conv1D(32) + Conv1D(64) + MaxPool + Dense(32) + Dropout(0.5)` | 0.8958 | 0.7907 | 0.7772 |

O **FNN Final** obteve o melhor desempenho global. A **CNN Final** apresentou accuracy ligeiramente inferior, mas F1-score mais equilibrado na classe *Canceled* (1) — maior interesse prático em contexto hoteleiro.

---

## Estrutura do Projeto

```
hotel_cancellation_deeplearning/
│
├── notebooks/
│   └── hotel_cancellation_neural_networks.ipynb
│
├── src/
│   ├── __init__.py
│   ├── config.py          # colunas, datas de split, paths
│   ├── data.py            # load_data(), reconstruct_booking_date()
│   ├── preprocessing.py   # clean_data(), split_data(), prepare_splits()
│   ├── eda.py             # funções de análise exploratória
│   ├── models.py          # arquiteturas FNN e CNN
│   ├── train.py           # train_model(), train_model_no_early_stop()
│   └── evaluate.py        # evaluate_model(), compare_models(), plot_history()
│
├── data/
│   └── hotel_bookings.csv     # descarregar via Kaggle (ver abaixo)
│
├── README.md
├── requirements.txt
└── .gitignore
```

---

## Setup

```bash
# 1. Clonar o repositório
git clone https://github.com/<username>/hotel_cancellation_deeplearning.git
cd hotel_cancellation_deeplearning

# 2. Instalar dependências
pip install -r requirements.txt

# 3. Colocar o dataset em data/hotel_bookings.csv
#    Download: https://www.kaggle.com/datasets/mexwell/hotel-bookings

# 4. Correr o notebook
jupyter notebook notebooks/hotel_cancellation_neural_networks.ipynb
```

---

## Decisões Metodológicas

**Duplicados mantidos** — o dataset contém ~32k registos com valores idênticos em todas as colunas. Sem identificador único de reserva, não é possível confirmar redundância. A remoção alteraria a distribuição da target de 37% para 27%.

**Divisão temporal** — os dados foram divididos cronologicamente via `booking_date = arrival_date − lead_time`, simulando um cenário real de previsão e evitando data leakage temporal.

**Leakage removido** — excluídas variáveis não disponíveis no momento da reserva: `reservation_status`, `reservation_status_date`, `booking_changes`, `days_in_waiting_list`, `assigned_room_type`.

**CNN em dados tabulares** — a entrada foi reformulada de `(n, features)` para `(n, features, 1)` para compatibilidade com `Conv1D`. Adaptação metodologicamente válida, sem estrutura naturalmente convolucional.

---

## Autor

Ricardo Filipe Fernandes da Silva  
Pós-graduação em Data Science e Business Intelligence — ISAG — 2026
