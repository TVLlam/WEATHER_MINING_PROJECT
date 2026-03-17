"""Quick test for Phase 3: WeatherClassifier + TimeForecaster."""
import os
import sys
import logging

project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, project_root)
os.chdir(project_root)

logging.basicConfig(level=logging.INFO, format="%(levelname)s - %(message)s")

import pandas as pd
from src.data.loader import load_config, resolve_path

config = load_config()
parquet_path = resolve_path(config["data"]["cleaned_parquet"])
df = pd.read_parquet(parquet_path)
print(f"Loaded data: {df.shape}")

# --- Test Classification ---
print("\n" + "=" * 60)
print("TEST: WeatherClassifier")
print("=" * 60)
from src.models.classification import WeatherClassifier

classifier = WeatherClassifier(config)
cls_results = classifier.run(df)
classifier.save_results(cls_results["results_df"])
print(f"\nBest model: {cls_results['best_model']}")
print(cls_results["results_df"])

# --- Test Forecasting ---
print("\n" + "=" * 60)
print("TEST: TimeForecaster")
print("=" * 60)
from src.models.forecasting import TimeForecaster

forecaster = TimeForecaster(config)
fc_results = forecaster.run(df)
forecaster.save_results()
print(fc_results["results_df"])

print("\n✅ PHASE 3 PIPELINE TEST PASSED!")
