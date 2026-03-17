"""Quick test for Phase 2 pipeline: AssociationMiner + WeatherClusterer."""
import os
import sys
import logging

project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, project_root)
os.chdir(project_root)

logging.basicConfig(level=logging.INFO, format="%(levelname)s - %(message)s")

import pandas as pd
from src.data.loader import load_config, resolve_path
from src.mining.association import AssociationMiner
from src.mining.clustering import WeatherClusterer

config = load_config()
parquet_path = resolve_path(config["data"]["cleaned_parquet"])
df = pd.read_parquet(parquet_path)
print(f"Loaded data: {df.shape}")

# --- Test Association Mining ---
print("\n" + "=" * 60)
print("TEST: AssociationMiner")
print("=" * 60)
miner = AssociationMiner(config)
results_assoc = miner.run(df)
miner.save_results()
print(f"\nSeasons with rules: {list(results_assoc['rules_by_season'].keys())}")
print(f"Total rules: {len(results_assoc['comparison']) if not results_assoc['comparison'].empty else 0}")

# --- Test Clustering ---
print("\n" + "=" * 60)
print("TEST: WeatherClusterer")
print("=" * 60)
clusterer = WeatherClusterer(config)
results_cluster = clusterer.run(df)
clusterer.save_results()
print(f"\nBest K: {results_cluster['best_k']}")
print(f"Cluster profiles:\n{results_cluster['profiles'][['Cluster_Name', 'Count']]}")

print("\n✅ PHASE 2 PIPELINE TEST PASSED!")
