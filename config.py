"""
Configuration for P2-ETF-DEEPAR engine.
"""

import os
from datetime import datetime

# --- Hugging Face Repositories ---
HF_DATA_REPO = "P2SAMAPA/fi-etf-macro-signal-master-data"
HF_DATA_FILE = "master_data.parquet"
HF_OUTPUT_REPO = "P2SAMAPA/p2-etf-deepar-results"

# --- Universe Definitions ---
FI_COMMODITIES_TICKERS = ["TLT", "VCIT", "LQD", "HYG", "VNQ", "GLD", "SLV"]
EQUITY_SECTORS_TICKERS = [
    "SPY", "QQQ", "XLK", "XLF", "XLE", "XLV",
    "XLI", "XLY", "XLP", "XLU", "GDX", "XME",
    "IWF", "XSD", "XBI", "IWM"
]
ALL_TICKERS = list(set(FI_COMMODITIES_TICKERS + EQUITY_SECTORS_TICKERS))

UNIVERSES = {
    "FI_COMMODITIES": FI_COMMODITIES_TICKERS,
    "EQUITY_SECTORS": EQUITY_SECTORS_TICKERS,
    "COMBINED": ALL_TICKERS
}

# --- DeepAR Parameters (Increased Complexity) ---
CONTEXT_LENGTH = 126                  # Past days used as input (~6 months)
PREDICTION_LENGTH = 22                # Forecast up to 22 days ahead
HIDDEN_SIZE = 128                     # LSTM hidden size (increased from 32)
NUM_LAYERS = 3                        # LSTM layers (increased from 2)
EPOCHS = 100                          # More epochs
BATCH_SIZE = 32                       # Smaller batch for better generalization
LEARNING_RATE = 0.0005                # Slightly lower learning rate
EARLY_STOP_PATIENCE = 10              # More patience before stopping
RANDOM_SEED = 42
MIN_OBSERVATIONS = 504                # Minimum data required (2 years)

# --- Forecasting ---
FORECAST_HORIZONS = [1, 5, 22]        # Horizons to output

# --- Shrinking Windows ---
SHRINKING_WINDOW_START_YEARS = list(range(2010, 2025))

# --- Date Handling ---
TODAY = datetime.now().strftime("%Y-%m-%d")

# --- Optional: Hugging Face Token ---
HF_TOKEN = os.environ.get("HF_TOKEN", None)
