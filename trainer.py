"""
Main training script for DeepAR engine.
"""

import json
import pandas as pd
import numpy as np

import config
import data_manager
from deepar_model import DeepARTrainer
import push_results

def run_deepar():
    print(f"=== P2-ETF-DEEPAR Run: {config.TODAY} ===")
    df_master = data_manager.load_master_data()

    all_results = {}
    top_picks = {}

    trainer = DeepARTrainer(
        context_len=config.CONTEXT_LENGTH,
        pred_len=config.PREDICTION_LENGTH,
        hidden_size=config.HIDDEN_SIZE,
        num_layers=config.NUM_LAYERS,
        epochs=config.EPOCHS,
        batch_size=config.BATCH_SIZE,
        lr=config.LEARNING_RATE,
        patience=config.EARLY_STOP_PATIENCE,
        seed=config.RANDOM_SEED
    )

    for universe_name, tickers in config.UNIVERSES.items():
        print(f"\n--- Processing Universe: {universe_name} ---")
        universe_results = {}

        for ticker in tickers:
            print(f"  Training DeepAR for {ticker}...")
            returns = data_manager.prepare_returns_series(df_master, ticker)
            if len(returns) < config.MIN_OBSERVATIONS:
                continue
            # Use up to 4 years of data for training (more data)
            recent = returns.iloc[-min(len(returns), 1008):].values
            success = trainer.fit(recent)
            if not success:
                continue
            forecasts = trainer.forecast(recent)
            universe_results[ticker] = {
                'ticker': ticker,
                'forecast_1d': forecasts.get(1),
                'forecast_5d': forecasts.get(5),
                'forecast_22d': forecasts.get(22)
            }

        all_results[universe_name] = universe_results
        sorted_tickers = sorted(universe_results.items(),
                                key=lambda x: x[1].get('forecast_1d', -np.inf),
                                reverse=True)
        top_picks[universe_name] = [{'ticker': t, **d} for t, d in sorted_tickers[:3]]

    output_payload = {
        "run_date": config.TODAY,
        "config": {
            "context_length": config.CONTEXT_LENGTH,
            "prediction_length": config.PREDICTION_LENGTH,
            "hidden_size": config.HIDDEN_SIZE,
            "num_layers": config.NUM_LAYERS,
            "epochs": config.EPOCHS
        },
        "daily_trading": {
            "universes": all_results,
            "top_picks": top_picks
        }
    }

    push_results.push_daily_result(output_payload)
    print("\n=== Run Complete ===")

if __name__ == "__main__":
    run_deepar()
