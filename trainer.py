"""
Main training script for DeepAR + N-BEATS engine.
"""

import json
import pandas as pd
import numpy as np

import config
import data_manager
from deepar_model import DeepARTrainer
from nbeats_model import NBEATSTrainer
import push_results

def run_models():
    print(f"=== P2-ETF-DEEPAR+NBEATS Run: {config.TODAY} ===")
    df_master = data_manager.load_master_data()

    # --- DeepAR ---
    print("\n" + "="*50)
    print("TRAINING DEEPAR MODELS")
    print("="*50)
    deepar_trainer = DeepARTrainer(
        context_len=config.DEEPAR_CONTEXT_LENGTH,
        pred_len=config.DEEPAR_PREDICTION_LENGTH,
        hidden_size=config.DEEPAR_HIDDEN_SIZE,
        num_layers=config.DEEPAR_NUM_LAYERS,
        epochs=config.DEEPAR_EPOCHS,
        batch_size=config.DEEPAR_BATCH_SIZE,
        lr=config.DEEPAR_LEARNING_RATE,
        patience=config.DEEPAR_EARLY_STOP_PATIENCE,
        seed=config.RANDOM_SEED
    )

    deepar_results = {}
    deepar_top_picks = {}

    for universe_name, tickers in config.UNIVERSES.items():
        print(f"\n--- DeepAR Universe: {universe_name} ---")
        universe_results = {}
        for ticker in tickers:
            print(f"  Training {ticker}...")
            returns = data_manager.prepare_returns_series(df_master, ticker)
            if len(returns) < config.MIN_OBSERVATIONS:
                continue
            recent = returns.iloc[-min(len(returns), 1008):].values
            success = deepar_trainer.fit(recent)
            if not success:
                continue
            forecasts = deepar_trainer.forecast(recent)
            universe_results[ticker] = {
                'ticker': ticker,
                'forecast_1d': forecasts.get(1),
                'forecast_5d': forecasts.get(5),
                'forecast_22d': forecasts.get(22)
            }
        deepar_results[universe_name] = universe_results
        sorted_tickers = sorted(universe_results.items(),
                                key=lambda x: x[1].get('forecast_1d', -np.inf),
                                reverse=True)
        deepar_top_picks[universe_name] = [{'ticker': t, **d} for t, d in sorted_tickers[:3]]

    # --- N-BEATS ---
    print("\n" + "="*50)
    print("TRAINING N-BEATS MODELS")
    print("="*50)
    nbeats_trainer = NBEATSTrainer(
        backcast_length=config.NBEATS_CONTEXT_LENGTH,
        forecast_length=config.NBEATS_PREDICTION_LENGTH,
        stack_types=config.NBEATS_STACK_TYPES,
        n_blocks_per_stack=config.NBEATS_N_BLOCKS_PER_STACK,
        hidden_size=config.NBEATS_HIDDEN_SIZE,
        epochs=config.NBEATS_EPOCHS,
        batch_size=config.NBEATS_BATCH_SIZE,
        lr=config.NBEATS_LEARNING_RATE,
        patience=config.NBEATS_EARLY_STOP_PATIENCE,
        seed=config.RANDOM_SEED
    )

    nbeats_results = {}
    nbeats_top_picks = {}

    for universe_name, tickers in config.UNIVERSES.items():
        print(f"\n--- N-BEATS Universe: {universe_name} ---")
        universe_results = {}
        for ticker in tickers:
            print(f"  Training {ticker}...")
            returns = data_manager.prepare_returns_series(df_master, ticker)
            if len(returns) < config.MIN_OBSERVATIONS:
                continue
            recent = returns.iloc[-min(len(returns), 1008):].values
            success = nbeats_trainer.fit(recent)
            if not success:
                continue
            forecasts = nbeats_trainer.forecast(recent)
            universe_results[ticker] = {
                'ticker': ticker,
                'forecast_1d': forecasts.get(1),
                'forecast_5d': forecasts.get(5),
                'forecast_22d': forecasts.get(22)
            }
        nbeats_results[universe_name] = universe_results
        sorted_tickers = sorted(universe_results.items(),
                                key=lambda x: x[1].get('forecast_1d', -np.inf),
                                reverse=True)
        nbeats_top_picks[universe_name] = [{'ticker': t, **d} for t, d in sorted_tickers[:3]]

    output_payload = {
        "run_date": config.TODAY,
        "config": {
            "deepar": {
                "context_length": config.DEEPAR_CONTEXT_LENGTH,
                "hidden_size": config.DEEPAR_HIDDEN_SIZE,
                "num_layers": config.DEEPAR_NUM_LAYERS,
                "epochs": config.DEEPAR_EPOCHS
            },
            "nbeats": {
                "context_length": config.NBEATS_CONTEXT_LENGTH,
                "hidden_size": config.NBEATS_HIDDEN_SIZE,
                "stack_types": config.NBEATS_STACK_TYPES,
                "n_blocks_per_stack": config.NBEATS_N_BLOCKS_PER_STACK,
                "epochs": config.NBEATS_EPOCHS
            }
        },
        "deepar": {
            "universes": deepar_results,
            "top_picks": deepar_top_picks
        },
        "nbeats": {
            "universes": nbeats_results,
            "top_picks": nbeats_top_picks
        }
    }

    push_results.push_daily_result(output_payload)
    print("\n=== Run Complete ===")

if __name__ == "__main__":
    run_models()
