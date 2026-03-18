"""
Module: evaluation/report.py
Tổng hợp bảng/biểu đồ kết quả cho toàn bộ pipeline.

Tự động tạo bảng tổng kết từ kết quả của các module:
- Classification results
- Forecasting results
- Clustering profiles
- Association rules summary
- Anomaly detection summary
"""

import os
import logging
from typing import Dict, Optional

import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from src.data.loader import load_config, resolve_path

logger = logging.getLogger(__name__)


class PipelineReporter:
    """
    Tổng hợp kết quả từ tất cả các module thành báo cáo tổng hợp.

    Parameters
    ----------
    config : dict, optional
        Dictionary config từ params.yaml.
    """

    def __init__(self, config: dict = None):
        self.config = config or load_config()
        self.tables_dir = resolve_path(self.config["outputs"]["tables_dir"])
        self.figures_dir = resolve_path(self.config["outputs"]["figures_dir"])
        self.reports = {}

    def load_all_results(self) -> dict:
        """Đọc tất cả file kết quả từ outputs/tables/."""
        logger.info("=" * 60)
        logger.info("TỔNG HỢP KẾT QUẢ PIPELINE")
        logger.info("=" * 60)

        result_files = {
            "classification": "classification_results.csv",
            "forecasting": "forecasting_results.csv",
            "cluster_profiles": "cluster_profiles.csv",
            "association_rules": "association_rules_by_season.csv",
            "anomaly_summary": "anomaly_summary.csv",
            "anomaly_by_season": "anomaly_by_season.csv",
        }

        for key, filename in result_files.items():
            path = os.path.join(self.tables_dir, filename)
            if os.path.exists(path):
                try:
                    self.reports[key] = pd.read_csv(path)
                    logger.info("  ✅ Loaded %s (%d rows)", filename, len(self.reports[key]))
                except Exception as e:
                    logger.warning("  ⚠️ Lỗi đọc %s: %s", filename, e)
            else:
                logger.info("  ⏭️ Không tìm thấy %s (bỏ qua)", filename)

        return self.reports

    def generate_summary_table(self) -> pd.DataFrame:
        """
        Tạo bảng tổng kết tất cả kết quả pipeline.

        Returns
        -------
        pd.DataFrame
            Bảng tổng kết.
        """
        summary_rows = []

        # Classification
        if "classification" in self.reports:
            cls_df = self.reports["classification"]
            best = cls_df.iloc[0]
            summary_rows.append({
                "Module": "Classification",
                "Phương pháp tốt nhất": best.get("Model", "N/A"),
                "Metric chính": f"F1-macro = {best.get('F1_macro', 0):.4f}",
                "Metric phụ": f"Accuracy = {best.get('Accuracy', 0):.4f}",
            })

        # Forecasting
        if "forecasting" in self.reports:
            fc_df = self.reports["forecasting"]
            for _, row in fc_df.iterrows():
                summary_rows.append({
                    "Module": "Forecasting",
                    "Phương pháp tốt nhất": row.get("Model", "N/A"),
                    "Metric chính": f"MAE = {row.get('MAE', 0):.4f}°C",
                    "Metric phụ": f"RMSE = {row.get('RMSE', 0):.4f}°C",
                })

        # Clustering
        if "cluster_profiles" in self.reports:
            n_clusters = len(self.reports["cluster_profiles"])
            summary_rows.append({
                "Module": "Clustering",
                "Phương pháp tốt nhất": "K-Means",
                "Metric chính": f"K = {n_clusters}",
                "Metric phụ": "Silhouette-based",
            })

        # Association
        if "association_rules" in self.reports:
            n_rules = len(self.reports["association_rules"])
            summary_rows.append({
                "Module": "Association Mining",
                "Phương pháp tốt nhất": "FP-Growth",
                "Metric chính": f"{n_rules} luật",
                "Metric phụ": "4 mùa",
            })

        # Anomaly Detection
        if "anomaly_summary" in self.reports:
            anom_df = self.reports["anomaly_summary"]
            for _, row in anom_df.iterrows():
                summary_rows.append({
                    "Module": "Anomaly Detection",
                    "Phương pháp tốt nhất": row.get("Method", "N/A"),
                    "Metric chính": f"{row.get('N_Anomalies', 0)} anomalies",
                    "Metric phụ": f"{row.get('Pct_Anomalies', 0):.2f}%",
                })

        summary_df = pd.DataFrame(summary_rows)
        self.reports["pipeline_summary"] = summary_df

        logger.info("\n📊 BẢNG TỔNG KẾT PIPELINE:")
        logger.info("\n%s", summary_df.to_string(index=False))

        return summary_df

    def save_summary(self, output_dir: str = None) -> str:
        """Lưu bảng tổng kết ra CSV."""
        if output_dir is None:
            output_dir = self.tables_dir
        os.makedirs(output_dir, exist_ok=True)

        if "pipeline_summary" in self.reports:
            path = os.path.join(output_dir, "pipeline_summary.csv")
            self.reports["pipeline_summary"].to_csv(
                path, index=False, encoding="utf-8-sig"
            )
            logger.info("Đã lưu bảng tổng kết tại: %s", path)
            return path
        return ""

    def run(self) -> dict:
        """Chạy toàn bộ pipeline tổng hợp."""
        self.load_all_results()
        summary = self.generate_summary_table()
        self.save_summary()
        return {
            "reports": self.reports,
            "summary": summary,
        }


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(levelname)s - %(message)s")

    reporter = PipelineReporter()
    results = reporter.run()

    print("\n✅ Pipeline Report hoàn tất!")
    if "pipeline_summary" in results["reports"]:
        print(results["summary"])
