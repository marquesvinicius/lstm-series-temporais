# LSTM para séries temporais: temperatura diária de Rio Verde-GO

[![Abrir no Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/drive/1hOm8uIOrYiDYmf4ErH_pvSOSJSHCEcDK)

**Resultado (teste, 1.900 dias, out./2020 → dez./2025):** RMSE = **1,141 °C** · MAE = **0,783 °C** · R² = 0,794. Isso é **9,4% menos erro (RMSE) que a persistência** ("amanhã = hoje").

Trabalho da disciplina de Redes Neurais (UniRV, Prof. Sandro Moreira): Laboratório de LSTM com Séries Temporais.
O notebook `lstm_series_temporais.ipynb` contém o código, as saídas e o relatório (seções marcadas com 📄).

- **Problema:** prever a temperatura média do dia seguinte (*one-step-ahead forecasting*).
- **Dados:** [Open-Meteo Historical Weather API](https://open-meteo.com/en/docs/historical-weather-api) (reanálise ERA5), Rio Verde-GO, diário de 2000 a 2025 (9.497 dias). São 6 variáveis meteorológicas mais o dia do ano em seno/cosseno. Há uma cópia em `data/clima_rio_verde.csv`.
- **Pré-processamento:**
  - divisão cronológica: 80% treino (os últimos 10% dele viram validação) e 20% teste;
  - `MinMaxScaler` [0, 1] ajustado só no treino;
  - janela deslizante de T = 30 dias.
- **Modelo:**
  - camadas: Input (30, 8) → LSTM(64, return_sequences) → Dropout(0,2) → LSTM(32) → Dropout(0,2) → Dense(16, ReLU) → Dense(1, linear);
  - treino: Adam com perda MSE, EarlyStopping e ReduceLROnPlateau.
- **Avaliação:** `inverse_transform` para °C; RMSE, MAE e R²; comparação com 3 baselines (persistência, média móvel de 7 dias e climatologia); análise dos erros por mês.
- **Experimentos:** janela T = 7, 30 e 60; univariada × multivariada; `shuffle` dos lotes de treino ligado × desligado.

| Modelo (teste) | RMSE (°C) | MAE (°C) | R² |
|---|---|---|---|
| **LSTM (T=30, multivariada)** | **1,141** | **0,783** | **0,794** |
| Persistência | 1,259 | 0,861 | 0,749 |
| Média móvel 7 dias | 1,747 | 1,268 | 0,517 |
| Climatologia | 1,962 | 1,449 | 0,392 |

## Como executar
**Google Colab:** clique no botão acima e use *Ambiente de execução → Executar tudo*. Funciona com CPU e leva cerca de 10–20 min. Os dados são baixados automaticamente da API.

**Local:**
```bash
python -m venv .venv
.venv\Scripts\pip install -r requirements.txt jupyter
.venv\Scripts\jupyter notebook lstm_series_temporais.ipynb
```
Para um teste rápido do código (2 épocas), defina a variável de ambiente `LSTM_QUICK=1`.

## Arquivos
- `lstm_series_temporais.ipynb`: notebook executado (código, saídas e relatório)
- `lstm_series_temporais.html`: o mesmo notebook, para abrir no navegador
- `data/clima_rio_verde.csv`: cópia dos dados
- `outputs/`: figuras, `metricas_teste.csv`, `experimentos.csv`, `resumo.json` e o modelo treinado (`modelo_lstm.keras`)
