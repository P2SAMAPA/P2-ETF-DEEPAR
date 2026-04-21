# P2-ETF-DEEPAR

**DeepAR Probabilistic Multi‑Horizon Forecasting for ETF Selection**

[![Daily Run](https://github.com/P2SAMAPA/P2-ETF-DEEPAR/actions/workflows/daily_run.yml/badge.svg)](https://github.com/P2SAMAPA/P2-ETF-DEEPAR/actions/workflows/daily_run.yml)
[![Hugging Face Dataset](https://img.shields.io/badge/🤗%20Dataset-p2--etf--deepar--results-blue)](https://huggingface.co/datasets/P2SAMAPA/p2-etf-deepar-results)

## Overview

`P2-ETF-DEEPAR` uses **DeepAR**, an autoregressive recurrent neural network (LSTM), to generate probabilistic forecasts of ETF returns at multiple horizons (1‑day, 5‑day, 22‑day). The engine ranks ETFs by their 1‑day forecast and displays the top picks per universe, along with the full term‑structure of expected returns.

DeepAR is well‑suited for time series with complex patterns and provides a unified view across short, medium, and longer‑term horizons.

## Universe Coverage

| Universe | Tickers |
|----------|---------|
| **FI / Commodities** | TLT, VCIT, LQD, HYG, VNQ, GLD, SLV |
| **Equity Sectors** | SPY, QQQ, XLK, XLF, XLE, XLV, XLI, XLY, XLP, XLU, GDX, XME, IWF, XSD, XBI, IWM |
| **Combined** | All tickers above |

Data is sourced from: [`P2SAMAPA/fi-etf-macro-signal-master-data`](https://huggingface.co/datasets/P2SAMAPA/fi-etf-macro-signal-master-data)

## Methodology

1. **Data Preparation**: Log returns are computed from daily closing prices.
2. **Sequence Creation**: Sliding windows of 60 past days are used as input to predict the next 22 days.
3. **Model Architecture**: Two‑layer LSTM with 32 hidden units, trained to minimize MSE.
4. **Training**: One model per ETF, with early stopping to prevent overfitting.
5. **Forecasting**: The model produces point forecasts for horizons 1, 5, and 22 days.
6. **Ranking**: ETFs are ranked within each universe by the 1‑day forecast (annualized).

## File Structure
P2-ETF-DEEPAR/
├── config.py # Paths, universes, DeepAR parameters
├── data_manager.py # Data loading and preprocessing
├── deepar_model.py # DeepAR model implementation (PyTorch)
├── trainer.py # Main orchestration script
├── push_results.py # Upload results to Hugging Face
├── streamlit_app.py # Interactive dashboard
├── us_calendar.py # U.S. market calendar utilities
├── requirements.txt # Python dependencies
├── .github/workflows/ # Scheduled GitHub Action
└── .streamlit/ # Streamlit theme

text

## Configuration

Key parameters in `config.py`:

| Parameter | Default | Description |
|-----------|---------|-------------|
| `CONTEXT_LENGTH` | 60 | Past days used as input |
| `PREDICTION_LENGTH` | 22 | Maximum forecast horizon |
| `HIDDEN_SIZE` | 32 | LSTM hidden units |
| `NUM_LAYERS` | 2 | LSTM layers |
| `EPOCHS` | 50 | Training epochs |
| `EARLY_STOP_PATIENCE` | 5 | Stop if no validation improvement |

## Running Locally

```bash
git clone https://github.com/P2SAMAPA/P2-ETF-DEEPAR.git
cd P2-ETF-DEEPAR
pip install -r requirements.txt
export HF_TOKEN="your_token_here"
python trainer.py
streamlit run streamlit_app.py
Dashboard Features
Hero Card: Top pick with 1‑day, 5‑day, and 22‑day forecasts.

Multi‑Horizon Table: All ETFs ranked by 1‑day forecast, showing full term‑structure.

Next Trading Day: U.S. market calendar integration.

Integration with Other Engines
DeepAR's multi‑horizon forecasts can be used to:

Blend with single‑horizon engines (BSTS, Particle Filter) for a consensus signal.

Detect term‑structure inversions (e.g., bullish 1‑day but bearish 22‑day) to adjust position sizing.

License
MIT License
