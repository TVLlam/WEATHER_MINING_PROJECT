"""Quick test for Phase 1 pipeline: WeatherCleaner + FeatureBuilder."""
import os
import sys
import logging

# Ensure project root is in path
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, project_root)
os.chdir(project_root)

logging.basicConfig(level=logging.INFO, format="%(levelname)s - %(message)s")

from src.data.cleaner import WeatherCleaner
from src.features.builder import FeatureBuilder

print("=" * 60)
print("TEST: WeatherCleaner")
print("=" * 60)
cleaner = WeatherCleaner()
df_clean = cleaner.run()
save_path = cleaner.save(df_clean, fmt="parquet")
print(f"Cleaned shape: {df_clean.shape}")
print(f"Saved to: {save_path}")

print("\n" + "=" * 60)
print("TEST: FeatureBuilder")
print("=" * 60)
builder = FeatureBuilder()
df_feat = builder.run(df_clean)
save_path2 = builder.save(df_feat, fmt="parquet")
print(f"Feature shape: {df_feat.shape}")
print(f"Saved to: {save_path2}")

print("\n" + "=" * 60)
print("FINAL VERIFICATION")
print("=" * 60)
print(f"Columns: {df_feat.columns.tolist()}")
print(f"\nSeason counts:\n{df_feat['Season'].value_counts()}")
print(f"\nTemp_Bin counts:\n{df_feat['Temp_Bin'].value_counts()}")
print(f"\nHumidity_Bin counts:\n{df_feat['Humidity_Bin'].value_counts()}")
print(f"\nWind_Bin counts:\n{df_feat['Wind_Bin'].value_counts()}")
print(f"\nMissing values:\n{df_feat.isnull().sum()}")
print("\n✅ PHASE 1 PIPELINE TEST PASSED!")
