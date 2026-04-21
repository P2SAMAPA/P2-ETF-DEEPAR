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

# --- DeepAR Parameters ---
DEEPAR_CONTEXT_LENGTH = 126
DEEPAR_PREDICTION_LENGTH = 22
DEEPAR_HIDDEN_SIZE = 128
DEEPAR_NUM_LAYERS = 3
DEEPAR_EPOCHS = 100
DEEPAR_BATCH_SIZE = 32
DEEPAR_LEARNING_RATE = 0.0005
DEEPAR_EARLY_STOP_PATIENCE = 10

# --- N-BEATS Parameters ---
NBEATS_CONTEXT_LENGTH = 126          # Input window (same as DeepAR for consistency)
NBEATS_PREDICTION_LENGTH = 22        # Forecast horizon
NBEATS_STACK_TYPES = ["trend", "seasonality"]  # N-BEATS interpretable stacks
NBEATS_N_BLOCKS_PER_STACK = 3        # Blocks per stack
NBEATS_THEO_DIM = 4                  # Theta dimensions (for trend/seasonality)
NBEATS_HIDDEN_SIZE = 128             # Hidden layer size in each block
NBEATS_EPOCHS = 100
NBEATS_BATCH_SIZE = 32
NBEATS_LEARNING_RATE = 0.0005
NBEATS_EARLY_STOP_PATIENCE = 10

# --- Common Parameters ---
RANDOM_SEED = 42
MIN_OBSERVATIONS = 504
FORECAST_HORIZONS = [1, 5, 22]

# --- Shrinking Windows ---
SHRINKING_WINDOW_START_YEARS = list(range(2010, 2025))

# --- Date Handling ---
TODAY = datetime.now().strftime("%Y-%m-%d")

# --- Optional: Hugging Face Token ---
HF_TOKEN = os.environ.get("HF_TOKEN", None)
