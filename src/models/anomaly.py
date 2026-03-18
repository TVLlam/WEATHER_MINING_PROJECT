"""
Module: WeatherAnomalyDetector
Phát hiện ngày thời tiết bất thường (Anomaly Detection).

ĐIỂM CỐT LÕI — NHÁNH THAY THẾ cho Semi-supervised:
- Isolation Forest: phát hiện outlier dựa trên random partitioning
- Local Outlier Factor (LOF): phát hiện outlier dựa trên mật độ cục bộ
- So sánh IF vs LOF: overlap, phân bố theo mùa, top ngày bất thường
- Đánh giá: contamination tuning, biểu đồ trực quan
"""

import os
import logging
from typing import Dict, Optional

import numpy as np
import pandas as pd
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.ensemble import IsolationForest
from sklearn.neighbors import LocalOutlierFactor
from sklearn.preprocessing import StandardScaler

from src.data.loader import load_config, resolve_path

logger = logging.getLogger(__name__)


class WeatherAnomalyDetector:
    """
    Phát hiện ngày thời tiết bất thường bằng Isolation Forest và LOF.

    Parameters
    ----------
    config : dict, optional
        Dictionary config từ params.yaml.
    """

    # Các cột số dùng cho phát hiện bất thường
    FEATURE_COLS = [
        "Temperature (C)",
        "Apparent Temperature (C)",
        "Humidity",
        "Wind Speed (km/h)",
        "Wind Bearing (degrees)",
        "Visibility (km)",
        "Pressure (millibars)",
    ]

    def __init__(self, config: dict = None):
        self.config = config or load_config()
        self.anomaly_cfg = self.config.get("anomaly_detection", {})
        self.contamination = self.anomaly_cfg.get("contamination", 0.05)
        self.random_seed = self.anomaly_cfg.get("random_seed", 42)
        self.n_neighbors = self.anomaly_cfg.get("n_neighbors", 20)

        self.scaler = StandardScaler()
        self.X_scaled = None
        self.df_input = None
        self.results = {}

    def prepare_data(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Chuẩn bị dữ liệu: chọn features, chuẩn hóa.

        Parameters
        ----------
        df : pd.DataFrame
            DataFrame đã qua Feature Engineering.

        Returns
        -------
        pd.DataFrame
            DataFrame với features đã scale.
        """
        logger.info("=" * 60)
        logger.info("CHUẨN BỊ DỮ LIỆU CHO ANOMALY DETECTION")
        logger.info("=" * 60)

        self.df_input = df.copy()
        feature_cols = [c for c in self.FEATURE_COLS if c in df.columns]

        logger.info("Features sử dụng: %s", feature_cols)
        logger.info("Số mẫu: %d", len(df))

        # Chuẩn hóa
        self.X_scaled = self.scaler.fit_transform(df[feature_cols].values)
        logger.info("Đã chuẩn hóa %d features", len(feature_cols))

        return df

    def run_isolation_forest(self) -> np.ndarray:
        """
        Chạy Isolation Forest để phát hiện anomaly.

        Returns
        -------
        np.ndarray
            Array nhãn: 1 = bình thường, -1 = bất thường.
        """
        logger.info("\n" + "=" * 60)
        logger.info("ISOLATION FOREST (contamination=%.2f%%)", self.contamination * 100)
        logger.info("=" * 60)

        iso_forest = IsolationForest(
            contamination=self.contamination,
            random_state=self.random_seed,
            n_estimators=200,
            max_samples="auto",
            n_jobs=-1,
        )

        labels = iso_forest.fit_predict(self.X_scaled)
        scores = iso_forest.decision_function(self.X_scaled)

        n_anomalies = (labels == -1).sum()
        n_total = len(labels)

        logger.info("  Tổng mẫu: %d", n_total)
        logger.info("  Số ngày bất thường: %d (%.2f%%)",
                     n_anomalies, n_anomalies / n_total * 100)

        self.results["Isolation Forest"] = {
            "labels": labels,
            "scores": scores,
            "n_anomalies": n_anomalies,
            "model_name": "Isolation Forest",
        }

        return labels

    def run_lof(self) -> np.ndarray:
        """
        Chạy Local Outlier Factor (LOF) để phát hiện anomaly.

        Returns
        -------
        np.ndarray
            Array nhãn: 1 = bình thường, -1 = bất thường.
        """
        logger.info("\n" + "=" * 60)
        logger.info("LOCAL OUTLIER FACTOR (n_neighbors=%d, contamination=%.2f%%)",
                     self.n_neighbors, self.contamination * 100)
        logger.info("=" * 60)

        lof = LocalOutlierFactor(
            n_neighbors=self.n_neighbors,
            contamination=self.contamination,
            novelty=False,
            n_jobs=-1,
        )

        labels = lof.fit_predict(self.X_scaled)
        scores = lof.negative_outlier_factor_

        n_anomalies = (labels == -1).sum()
        n_total = len(labels)

        logger.info("  Tổng mẫu: %d", n_total)
        logger.info("  Số ngày bất thường: %d (%.2f%%)",
                     n_anomalies, n_anomalies / n_total * 100)

        self.results["LOF"] = {
            "labels": labels,
            "scores": scores,
            "n_anomalies": n_anomalies,
            "model_name": "LOF",
        }

        return labels

    def compare_methods(self) -> pd.DataFrame:
        """
        So sánh kết quả giữa Isolation Forest và LOF.

        Returns
        -------
        pd.DataFrame
            DataFrame tổng hợp so sánh.
        """
        logger.info("\n" + "=" * 60)
        logger.info("SO SÁNH ISOLATION FOREST vs LOF")
        logger.info("=" * 60)

        if_labels = self.results["Isolation Forest"]["labels"]
        lof_labels = self.results["LOF"]["labels"]

        # Thống kê overlap
        both_anomaly = ((if_labels == -1) & (lof_labels == -1)).sum()
        if_only = ((if_labels == -1) & (lof_labels == 1)).sum()
        lof_only = ((if_labels == 1) & (lof_labels == -1)).sum()
        both_normal = ((if_labels == 1) & (lof_labels == 1)).sum()

        logger.info("  Cả hai phát hiện bất thường: %d", both_anomaly)
        logger.info("  Chỉ IF phát hiện:            %d", if_only)
        logger.info("  Chỉ LOF phát hiện:           %d", lof_only)
        logger.info("  Cả hai cho là bình thường:    %d", both_normal)

        # Tạo DataFrame anomaly
        df_anomaly = self.df_input.copy()
        df_anomaly["IF_Label"] = if_labels
        df_anomaly["LOF_Label"] = lof_labels
        df_anomaly["IF_Score"] = self.results["Isolation Forest"]["scores"]
        df_anomaly["LOF_Score"] = self.results["LOF"]["scores"]
        df_anomaly["Consensus"] = "Normal"
        df_anomaly.loc[
            (if_labels == -1) & (lof_labels == -1), "Consensus"
        ] = "Both Anomaly"
        df_anomaly.loc[
            (if_labels == -1) & (lof_labels == 1), "Consensus"
        ] = "IF Only"
        df_anomaly.loc[
            (if_labels == 1) & (lof_labels == -1), "Consensus"
        ] = "LOF Only"

        self.results["comparison_df"] = df_anomaly

        # Bảng tổng hợp so sánh
        summary = pd.DataFrame({
            "Method": ["Isolation Forest", "LOF"],
            "N_Anomalies": [
                self.results["Isolation Forest"]["n_anomalies"],
                self.results["LOF"]["n_anomalies"],
            ],
            "Pct_Anomalies": [
                round(self.results["Isolation Forest"]["n_anomalies"] / len(if_labels) * 100, 2),
                round(self.results["LOF"]["n_anomalies"] / len(lof_labels) * 100, 2),
            ],
        })
        summary.loc[len(summary)] = [
            "Consensus (both)", both_anomaly,
            round(both_anomaly / len(if_labels) * 100, 2)
        ]

        self.results["summary_df"] = summary
        logger.info("\n%s", summary.to_string(index=False))

        return summary

    def analyze_anomalies_by_season(self) -> pd.DataFrame:
        """
        Phân tích phân bố anomaly theo mùa (Season).

        Returns
        -------
        pd.DataFrame
            Bảng phân bố anomaly theo mùa.
        """
        logger.info("\n" + "=" * 60)
        logger.info("PHÂN TÍCH ANOMALY THEO MÙA")
        logger.info("=" * 60)

        df = self.results["comparison_df"]

        if "Season" not in df.columns:
            logger.warning("Không có cột Season, bỏ qua phân tích theo mùa.")
            return pd.DataFrame()

        season_stats = []
        for season in ["Spring", "Summer", "Autumn", "Winter"]:
            mask = df["Season"] == season
            total = mask.sum()
            if total == 0:
                continue

            if_anom = ((df["IF_Label"] == -1) & mask).sum()
            lof_anom = ((df["LOF_Label"] == -1) & mask).sum()
            both_anom = ((df["Consensus"] == "Both Anomaly") & mask).sum()

            season_stats.append({
                "Season": season,
                "Total": total,
                "IF_Anomalies": if_anom,
                "IF_Pct": round(if_anom / total * 100, 2),
                "LOF_Anomalies": lof_anom,
                "LOF_Pct": round(lof_anom / total * 100, 2),
                "Consensus": both_anom,
                "Consensus_Pct": round(both_anom / total * 100, 2),
            })

            logger.info(
                "  %s: %d mẫu | IF=%d (%.1f%%) | LOF=%d (%.1f%%) | Both=%d (%.1f%%)",
                season, total, if_anom, if_anom / total * 100,
                lof_anom, lof_anom / total * 100,
                both_anom, both_anom / total * 100
            )

        season_df = pd.DataFrame(season_stats)
        self.results["season_analysis"] = season_df
        return season_df

    def get_top_anomalies(self, n: int = 20) -> pd.DataFrame:
        """
        Lấy top N ngày bất thường nhất (consensus giữa IF và LOF).

        Parameters
        ----------
        n : int
            Số lượng top anomaly.

        Returns
        -------
        pd.DataFrame
            Top N ngày bất thường.
        """
        logger.info("\n--- Top %d ngày THỜI TIẾT BẤT THƯỜNG nhất ---", n)

        df = self.results["comparison_df"]
        # Tính combined anomaly score (normalize cả 2)
        if_scores = df["IF_Score"]
        lof_scores = df["LOF_Score"]

        # Normalize về [0, 1] — score càng nhỏ càng bất thường
        if_norm = (if_scores - if_scores.max()) / (if_scores.min() - if_scores.max() + 1e-8)
        lof_norm = (lof_scores - lof_scores.max()) / (lof_scores.min() - lof_scores.max() + 1e-8)
        df["Combined_Score"] = (if_norm + lof_norm) / 2

        # Sort theo combined score giảm dần
        top_cols = ["Consensus", "Combined_Score"]
        if "Formatted Date" in df.columns:
            top_cols = ["Formatted Date"] + top_cols
        if "Season" in df.columns:
            top_cols.append("Season")
        top_cols += [c for c in self.FEATURE_COLS if c in df.columns]

        top_anomalies = df.nlargest(n, "Combined_Score")[top_cols]
        self.results["top_anomalies"] = top_anomalies

        logger.info("\n%s", top_anomalies.head(10).to_string())
        return top_anomalies

    def plot_anomaly_scatter(self, save_path: str = None) -> str:
        """Vẽ scatter plot Nhiệt độ vs Độ ẩm, đánh dấu anomaly."""
        df = self.results["comparison_df"]

        fig, axes = plt.subplots(1, 2, figsize=(18, 7))

        # IF
        colors_if = np.where(df["IF_Label"] == -1, "#FF4444", "#4488FF")
        axes[0].scatter(
            df["Temperature (C)"], df["Humidity"],
            c=colors_if, alpha=0.3, s=8, edgecolors="none"
        )
        axes[0].set_xlabel("Nhiệt độ (°C)", fontsize=12)
        axes[0].set_ylabel("Độ ẩm", fontsize=12)
        n_if = self.results["Isolation Forest"]["n_anomalies"]
        axes[0].set_title(
            f"Isolation Forest — {n_if} ngày bất thường",
            fontsize=14, fontweight="bold"
        )
        axes[0].grid(alpha=0.3)

        # LOF
        colors_lof = np.where(df["LOF_Label"] == -1, "#FF8800", "#44BB77")
        axes[1].scatter(
            df["Temperature (C)"], df["Humidity"],
            c=colors_lof, alpha=0.3, s=8, edgecolors="none"
        )
        axes[1].set_xlabel("Nhiệt độ (°C)", fontsize=12)
        axes[1].set_ylabel("Độ ẩm", fontsize=12)
        n_lof = self.results["LOF"]["n_anomalies"]
        axes[1].set_title(
            f"Local Outlier Factor — {n_lof} ngày bất thường",
            fontsize=14, fontweight="bold"
        )
        axes[1].grid(alpha=0.3)

        # Legend
        from matplotlib.lines import Line2D
        legend_elements = [
            Line2D([0], [0], marker="o", color="w", markerfacecolor="#FF4444",
                   label="Bất thường", markersize=8),
            Line2D([0], [0], marker="o", color="w", markerfacecolor="#4488FF",
                   label="Bình thường", markersize=8),
        ]
        axes[0].legend(handles=legend_elements, loc="upper right", fontsize=10)

        legend_lof = [
            Line2D([0], [0], marker="o", color="w", markerfacecolor="#FF8800",
                   label="Bất thường", markersize=8),
            Line2D([0], [0], marker="o", color="w", markerfacecolor="#44BB77",
                   label="Bình thường", markersize=8),
        ]
        axes[1].legend(handles=legend_lof, loc="upper right", fontsize=10)

        plt.suptitle(
            "Phát hiện Ngày Thời Tiết Bất Thường (Anomaly Detection)",
            fontsize=15, fontweight="bold"
        )
        plt.tight_layout()

        if save_path is None:
            fig_dir = resolve_path(self.config["outputs"]["figures_dir"])
            os.makedirs(fig_dir, exist_ok=True)
            save_path = os.path.join(fig_dir, "anomaly_scatter_comparison.png")

        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        plt.close()
        logger.info("Đã lưu scatter plot anomaly tại: %s", save_path)
        return save_path

    def plot_anomaly_by_season(self, save_path: str = None) -> str:
        """Vẽ biểu đồ phân bố anomaly theo mùa."""
        season_df = self.results.get("season_analysis")
        if season_df is None or season_df.empty:
            logger.warning("Không có dữ liệu phân tích theo mùa để vẽ.")
            return ""

        fig, ax = plt.subplots(figsize=(12, 6))
        x = np.arange(len(season_df))
        width = 0.25

        bars1 = ax.bar(x - width, season_df["IF_Pct"], width,
                        label="Isolation Forest", color="#FF4444", alpha=0.85)
        bars2 = ax.bar(x, season_df["LOF_Pct"], width,
                        label="LOF", color="#FF8800", alpha=0.85)
        bars3 = ax.bar(x + width, season_df["Consensus_Pct"], width,
                        label="Consensus (cả hai)", color="#9C27B0", alpha=0.85)

        # Hiển thị giá trị trên cột
        for bars in [bars1, bars2, bars3]:
            for bar in bars:
                h = bar.get_height()
                if h > 0:
                    ax.text(bar.get_x() + bar.get_width() / 2, h + 0.1,
                            f"{h:.1f}%", ha="center", va="bottom", fontsize=9)

        ax.set_xticks(x)
        ax.set_xticklabels(season_df["Season"], fontsize=12)
        ax.set_ylabel("% Ngày bất thường", fontsize=12)
        ax.set_title("Phân bố Anomaly theo Mùa — IF vs LOF",
                      fontsize=14, fontweight="bold")
        ax.legend(fontsize=11)
        ax.grid(axis="y", alpha=0.3)
        plt.tight_layout()

        if save_path is None:
            fig_dir = resolve_path(self.config["outputs"]["figures_dir"])
            os.makedirs(fig_dir, exist_ok=True)
            save_path = os.path.join(fig_dir, "anomaly_by_season.png")

        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        plt.close()
        logger.info("Đã lưu biểu đồ anomaly theo mùa tại: %s", save_path)
        return save_path

    def plot_anomaly_timeline(self, save_path: str = None) -> str:
        """Vẽ timeline các ngày bất thường trên trục thời gian."""
        df = self.results["comparison_df"]

        if "Formatted Date" not in df.columns:
            logger.warning("Không có cột 'Formatted Date', bỏ qua timeline.")
            return ""

        fig, axes = plt.subplots(2, 1, figsize=(18, 10), sharex=True)

        # Nhiệt độ + anomaly markers
        axes[0].plot(
            df["Formatted Date"], df["Temperature (C)"],
            alpha=0.5, linewidth=0.5, color="#2196F3", label="Nhiệt độ"
        )
        consensus_mask = df["Consensus"] == "Both Anomaly"
        if consensus_mask.sum() > 0:
            axes[0].scatter(
                df.loc[consensus_mask, "Formatted Date"],
                df.loc[consensus_mask, "Temperature (C)"],
                c="red", s=15, alpha=0.7, label=f"Anomaly ({consensus_mask.sum()} ngày)",
                zorder=5, edgecolors="darkred", linewidth=0.5
            )
        axes[0].set_ylabel("Nhiệt độ (°C)", fontsize=12)
        axes[0].set_title("Phát hiện Ngày Bất Thường trên Chuỗi Thời Gian",
                           fontsize=14, fontweight="bold")
        axes[0].legend(fontsize=10)
        axes[0].grid(alpha=0.3)

        # Combined anomaly score
        if "Combined_Score" in df.columns:
            axes[1].fill_between(
                df["Formatted Date"], df["Combined_Score"],
                alpha=0.4, color="#FF6B6B", label="Combined Anomaly Score"
            )
            threshold = df["Combined_Score"].quantile(1 - self.contamination)
            axes[1].axhline(y=threshold, color="red", linestyle="--",
                            alpha=0.8, label=f"Ngưỡng (top {self.contamination:.0%})")
        axes[1].set_ylabel("Anomaly Score", fontsize=12)
        axes[1].set_xlabel("Thời gian", fontsize=12)
        axes[1].legend(fontsize=10)
        axes[1].grid(alpha=0.3)

        plt.tight_layout()

        if save_path is None:
            fig_dir = resolve_path(self.config["outputs"]["figures_dir"])
            os.makedirs(fig_dir, exist_ok=True)
            save_path = os.path.join(fig_dir, "anomaly_timeline.png")

        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        plt.close()
        logger.info("Đã lưu timeline anomaly tại: %s", save_path)
        return save_path

    def run(self, df: pd.DataFrame) -> dict:
        """
        Chạy toàn bộ pipeline Anomaly Detection.

        Parameters
        ----------
        df : pd.DataFrame
            DataFrame đã qua Feature Engineering.

        Returns
        -------
        dict
            Kết quả tổng hợp.
        """
        # 1. Chuẩn bị dữ liệu
        self.prepare_data(df)

        # 2. Isolation Forest
        self.run_isolation_forest()

        # 3. LOF
        self.run_lof()

        # 4. So sánh hai phương pháp
        summary = self.compare_methods()

        # 5. Phân tích theo mùa
        season_analysis = self.analyze_anomalies_by_season()

        # 6. Top ngày bất thường nhất
        top_anomalies = self.get_top_anomalies(n=20)

        # 7. Biểu đồ
        scatter_path = self.plot_anomaly_scatter()
        season_path = self.plot_anomaly_by_season()
        timeline_path = self.plot_anomaly_timeline()

        # 8. Insight tổng hợp
        self._log_insights()

        return {
            "summary": summary,
            "season_analysis": season_analysis,
            "top_anomalies": top_anomalies,
            "scatter_plot": scatter_path,
            "season_plot": season_path,
            "timeline_plot": timeline_path,
        }

    def _log_insights(self):
        """Log các insight chính từ kết quả anomaly detection."""
        logger.info("\n" + "=" * 60)
        logger.info("INSIGHT — NGÀY THỜI TIẾT BẤT THƯỜNG")
        logger.info("=" * 60)

        n_if = self.results["Isolation Forest"]["n_anomalies"]
        n_lof = self.results["LOF"]["n_anomalies"]
        df = self.results["comparison_df"]
        n_both = (df["Consensus"] == "Both Anomaly").sum()

        logger.info(
            "🔹 Isolation Forest phát hiện %d ngày bất thường (%.1f%%)",
            n_if, n_if / len(df) * 100
        )
        logger.info(
            "🔹 LOF phát hiện %d ngày bất thường (%.1f%%)",
            n_lof, n_lof / len(df) * 100
        )
        logger.info(
            "🔹 Cả hai đồng thuận: %d ngày bất thường (%.1f%%)",
            n_both, n_both / len(df) * 100
        )

        # Phân tích đặc trưng anomaly
        if n_both > 0:
            anomaly_rows = df[df["Consensus"] == "Both Anomaly"]
            normal_rows = df[df["Consensus"] == "Normal"]

            logger.info("\n📊 So sánh đặc trưng: Bất thường vs Bình thường:")
            for col in self.FEATURE_COLS:
                if col in df.columns:
                    anom_mean = anomaly_rows[col].mean()
                    norm_mean = normal_rows[col].mean()
                    diff = anom_mean - norm_mean
                    logger.info(
                        "  %s: Anomaly=%.2f | Normal=%.2f | Chênh lệch=%+.2f",
                        col, anom_mean, norm_mean, diff
                    )

    def save_results(self, output_dir: str = None) -> str:
        """Lưu kết quả ra file CSV."""
        if output_dir is None:
            output_dir = resolve_path(self.config["outputs"]["tables_dir"])
        os.makedirs(output_dir, exist_ok=True)

        saved_paths = []

        # Lưu summary
        if "summary_df" in self.results:
            path = os.path.join(output_dir, "anomaly_summary.csv")
            self.results["summary_df"].to_csv(path, index=False, encoding="utf-8-sig")
            saved_paths.append(path)
            logger.info("Đã lưu anomaly summary tại: %s", path)

        # Lưu season analysis
        if "season_analysis" in self.results and not self.results["season_analysis"].empty:
            path = os.path.join(output_dir, "anomaly_by_season.csv")
            self.results["season_analysis"].to_csv(path, index=False, encoding="utf-8-sig")
            saved_paths.append(path)
            logger.info("Đã lưu anomaly by season tại: %s", path)

        # Lưu top anomalies
        if "top_anomalies" in self.results:
            path = os.path.join(output_dir, "anomaly_top_days.csv")
            self.results["top_anomalies"].to_csv(path, index=False, encoding="utf-8-sig")
            saved_paths.append(path)
            logger.info("Đã lưu top anomalies tại: %s", path)

        return saved_paths[0] if saved_paths else ""


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(levelname)s - %(message)s")

    config = load_config()
    parquet_path = resolve_path(config["data"]["cleaned_parquet"])
    df = pd.read_parquet(parquet_path)

    detector = WeatherAnomalyDetector(config)
    results = detector.run(df)
    detector.save_results()

    print("\n✅ Anomaly Detection hoàn tất!")
    print(results["summary"])
