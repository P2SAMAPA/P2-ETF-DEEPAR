# P2-ETF-DEEPAR + N‑BEATS

**Multi‑Horizon Probabilistic Forecasting with DeepAR and N‑BEATS for ETF Selection**

[![Daily Run](https://github.com/P2SAMAPA/P2-ETF-DEEPAR/actions/workflows/daily_run.yml/badge.svg)](https://github.com/P2SAMAPA/P2-ETF-DEEPAR/actions/workflows/daily_run.yml)
[![Hugging Face Dataset](https://img.shields.io/badge/🤗%20Dataset-p2--etf--deepar--results-blue)](https://huggingface.co/datasets/P2SAMAPA/p2-etf-deepar-results)

## Overview

`P2-ETF-DEEPAR` now combines two state‑of‑the‑art time‑series forecasting models:

- **DeepAR**: An autoregressive LSTM‑based probabilistic model that captures complex temporal patterns.
- **N‑BEATS**: A deep neural network with interpretable trend and seasonality stacks.

Both models are trained on up to 4 years of daily log returns for each ETF and produce forecasts at 1‑day, 5‑day, and 22‑day horizons. The dashboard displays the top picks and full forecast tables for each model in separate tabs.

## Universe Coverage

| Universe | Tickers |
|----------|---------|
| **FI / Commodities** | TLT, VCIT, LQD, HYG, VNQ, GLD, SLV |
| **Equity Sectors** | SPY, QQQ, XLK, XLF, XLE, XLV, XLI, XLY, XLP, XLU, GDX, XME, IWF, XSD, XBI, IWM |
| **Combined** | All tickers above |

Data is sourced from: [`P2SAMAPA/fi-etf-macro-signal-master-data`](https://huggingface.co/datasets/P2SAMAPA/fi-etf-macro-signal-master-data)

## Methodology

### DeepAR
- **Context window**: 126 trading days (~6 months)
- **Architecture**: 3‑layer LSTM with 128 hidden units
- **Training**: 100 epochs with early stopping (patience=10)
- **Forecast**: Autoregressive decoding for 22 days ahead

### N‑BEATS
- **Context window**: 126 trading days
- **Architecture**: Two interpretable stacks (trend + seasonality), each with 3 blocks
- **Hidden size**: 128
- **Training**: 100 epochs with early stopping
- **Forecast**: Direct multi‑horizon output

### Ranking
ETFs are ranked within each universe by their **1‑day forecast** (annualized). The top 3 are displayed as hero picks.

## File Structure
P2-ETF-DEEPAR/
├── config.py # Paths, universes, DeepAR & N‑BEATS parameters
├── data_manager.py # Data loading and preprocessing
├── deepar_model.py # DeepAR implementation (PyTorch)
├── nbeats_model.py # N‑BEATS implementation (PyTorch)
├── trainer.py # Orchestrates training for both models
├── push_results.py # Upload results to Hugging Face
├── streamlit_app.py # Interactive dashboard (two main tabs)
├── us_calendar.py # U.S. market calendar utilities
├── requirements.txt # Python dependencies
├── .github/workflows/ # Scheduled GitHub Action
└── .streamlit/ # Streamlit theme

text

## Configuration

Key parameters in `config.py`:

| Parameter | DeepAR | N‑BEATS |
|-----------|--------|---------|
| Context length | 126 | 126 |
| Hidden size | 128 | 128 |
| Layers / Stacks | 3 LSTM layers | 2 stacks (trend + seasonality) |
| Blocks per stack | – | 3 |
| Epochs | 100 | 100 |
| Batch size | 32 | 32 |
| Learning rate | 0.0005 | 0.0005 |
| Early stopping patience | 10 | 10 |

## Running Locally

```bash
git clone https://github.com/P2SAMAPA/P2-ETF-DEEPAR.git
cd P2-ETF-DEEPAR
pip install -r requirements.txt
export HF_TOKEN="your_token_here"
python trainer.py
streamlit run streamlit_app.py
Dashboard Features
Two Main Tabs: Switch between DeepAR and N‑BEATS forecasts.

Sub‑tabs per Universe: Combined, Equity Sectors, and FI/Commodities.

Hero Cards: Top pick with 1‑day, 5‑day, and 22‑day forecasts.

Full Forecast Table: All ETFs ranked by 1‑day forecast, showing complete term‑structure.

Next Trading Day: U.S. market calendar integration.

Performance
Training time: ~1.5–2 hours on GitHub Actions (CPU only)

Inference: < 1 second per ETF

Data used: Most recent ~4 years (up to 1,008 trading days) per ETF

Integration with Other Engines
The multi‑horizon forecasts can be used to:

Blend with single‑horizon engines (BSTS, Particle Filter) for consensus signals.

Detect term‑structure inversions (e.g., bullish 1‑day but bearish 22‑day).

Adjust position sizing based on forecast confidence.

License
MIT License
