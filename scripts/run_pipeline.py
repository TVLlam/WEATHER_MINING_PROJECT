"""
Script: run_pipeline.py
Chạy toàn bộ pipeline khai phá dữ liệu thời tiết end-to-end.

Usage:
    python scripts/run_pipeline.py
"""

import os
import sys
import logging
import time

# Setup project root
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, project_root)
os.chdir(project_root)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)

import pandas as pd
from src.data.loader import load_config, resolve_path


def main():
    t0 = time.time()
    config = load_config()

    print("=" * 70)
    print("   🌦️  WEATHER MINING PROJECT — FULL PIPELINE")
    print("   Đề tài 5: Dự báo thời tiết Szeged Hungary 2006-2016")
    print("=" * 70)

    # ─────────── PHASE 1: Tiền xử lý & Feature Engineering ───────────
    print("\n📌 PHASE 1: Tiền xử lý & Feature Engineering")
    print("-" * 50)

    from src.data.cleaner import WeatherCleaner
    cleaner = WeatherCleaner(config)
    df_clean = cleaner.run()
    cleaner.save(df_clean, fmt="parquet")

    from src.features.builder import FeatureBuilder
    builder = FeatureBuilder(config)
    df = builder.run(df_clean)
    builder.save(df, fmt="parquet")
    print(f"  ✅ Data: {df.shape[0]} rows × {df.shape[1]} cols")

    # ─────────── PHASE 2: Khai phá tri thức ───────────
    print("\n📌 PHASE 2: Khai phá tri thức (Association + Clustering)")
    print("-" * 50)

    from src.mining.association import AssociationMiner
    miner = AssociationMiner(config)
    assoc_results = miner.run(df)
    miner.save_results()
    n_rules = len(assoc_results["comparison"]) if not assoc_results["comparison"].empty else 0
    print(f"  ✅ Association: {n_rules} luật")

    from src.mining.clustering import WeatherClusterer
    clusterer = WeatherClusterer(config)
    clust_results = clusterer.run(df)
    clusterer.save_results()
    print(f"  ✅ Clustering: K={clust_results['best_k']}")

    # ─────────── PHASE 3: Mô hình dự đoán ───────────
    print("\n📌 PHASE 3: Mô hình dự đoán (Classification + Forecasting)")
    print("-" * 50)

    from src.models.classification import WeatherClassifier
    classifier = WeatherClassifier(config)
    cls_results = classifier.run(df)
    classifier.save_results(cls_results["results_df"])
    print(f"  ✅ Classification: Best={cls_results['best_model']} "
          f"(F1={cls_results['results_df'].iloc[0]['F1_macro']:.4f})")

    from src.models.forecasting import TimeForecaster
    forecaster = TimeForecaster(config)
    fc_results = forecaster.run(df)
    forecaster.save_results()
    print(f"  ✅ Forecasting: {fc_results['results_df'].to_dict('records')}")

    # ─────────── PHASE 4: Nhánh thay thế (Anomaly Detection) ───────────
    print("\n📌 PHASE 4: Nhánh thay thế — Anomaly Detection")
    print("-" * 50)

    from src.models.anomaly import WeatherAnomalyDetector
    detector = WeatherAnomalyDetector(config)
    anomaly_results = detector.run(df)
    detector.save_results()
    print(f"  ✅ Anomaly Detection: IF vs LOF done")

    # ─────────── TỔNG KẾT ───────────
    elapsed = time.time() - t0
    print("\n" + "=" * 70)
    print(f"   🎉 PIPELINE HOÀN TẤT trong {elapsed:.1f}s")
    print("=" * 70)
    print(f"\n📁 Outputs:")
    print(f"   figures/: {len(os.listdir(resolve_path('outputs/figures')))} files")
    print(f"   tables/:  {len(os.listdir(resolve_path('outputs/tables')))} files")


if __name__ == "__main__":
    main()
