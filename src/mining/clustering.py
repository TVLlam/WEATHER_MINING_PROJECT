"""
Module: WeatherClusterer
Phân cụm K-Means trên dữ liệu thời tiết đã chuẩn hóa.

ĐIỂM CỐT LÕI: Tự động tìm K tối ưu (Elbow + Silhouette),
tạo hồ sơ cụm (Cluster Profile) và gợi ý tên cụm tự động.
"""

import os
import logging
from typing import Optional, Tuple

import numpy as np
import pandas as pd
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score, silhouette_samples

from src.data.loader import load_config, resolve_path

logger = logging.getLogger(__name__)


class WeatherClusterer:
    """
    Phân cụm K-Means với khả năng tự tìm K tối ưu và tạo hồ sơ cụm.

    Parameters
    ----------
    config : dict, optional
        Dictionary config từ params.yaml.
    """

    # Các cột dùng cho phân cụm (đã scaled)
    SCALED_COLS = [
        "Temperature (C)_scaled",
        "Apparent Temperature (C)_scaled",
        "Humidity_scaled",
        "Wind Speed (km/h)_scaled",
        "Wind Bearing (degrees)_scaled",
        "Visibility (km)_scaled",
        "Pressure (millibars)_scaled",
    ]

    # Các cột gốc tương ứng (dùng cho profiling)
    ORIGINAL_COLS = [
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
        self.cluster_cfg = self.config["clustering"]
        self.k_range = tuple(self.cluster_cfg["k_range"])
        self.random_seed = self.cluster_cfg["random_seed"]
        self.n_init = self.cluster_cfg["n_init"]
        self.max_iter = self.cluster_cfg["max_iter"]

        self.best_k = None
        self.best_model = None
        self.labels = None
        self.inertias = []
        self.silhouette_scores = []
        self.cluster_profiles = None

    def find_optimal_k(self, df: pd.DataFrame) -> int:
        """
        Tìm số cụm K tối ưu bằng Elbow Method + Silhouette Score.

        Parameters
        ----------
        df : pd.DataFrame
            DataFrame chứa các cột scaled.

        Returns
        -------
        int
            Số cụm K tối ưu.
        """
        logger.info("=" * 60)
        logger.info("TÌM SỐ CỤM K TỐI ƯU (Elbow + Silhouette)")
        logger.info("=" * 60)

        X = df[self.SCALED_COLS].values
        k_min, k_max = self.k_range

        # Sample for silhouette (full dataset too large for pairwise distances)
        max_sil_samples = 10000
        if len(X) > max_sil_samples:
            rng = np.random.RandomState(self.random_seed)
            sil_idx = rng.choice(len(X), max_sil_samples, replace=False)
        else:
            sil_idx = np.arange(len(X))

        self.inertias = []
        self.silhouette_scores = []
        k_values = list(range(k_min, k_max + 1))

        for k in k_values:
            kmeans = KMeans(
                n_clusters=k,
                random_state=self.random_seed,
                n_init=self.n_init,
                max_iter=self.max_iter,
            )
            labels = kmeans.fit_predict(X)
            inertia = kmeans.inertia_
            sil_score = silhouette_score(X[sil_idx], labels[sil_idx])

            self.inertias.append(inertia)
            self.silhouette_scores.append(sil_score)

            logger.info("  K=%d | Inertia=%.0f | Silhouette=%.4f", k, inertia, sil_score)

        # Chọn K có Silhouette Score cao nhất
        best_idx = int(np.argmax(self.silhouette_scores))
        self.best_k = k_values[best_idx]

        logger.info("\n✅ K tối ưu = %d (Silhouette = %.4f)",
                     self.best_k, self.silhouette_scores[best_idx])

        return self.best_k

    def fit(self, df: pd.DataFrame, k: int = None) -> np.ndarray:
        """
        Huấn luyện K-Means với K đã chọn.

        Parameters
        ----------
        df : pd.DataFrame
            DataFrame chứa các cột scaled.
        k : int, optional
            Số cụm. Nếu None, dùng best_k.

        Returns
        -------
        np.ndarray
            Array nhãn cụm cho mỗi mẫu.
        """
        if k is None:
            k = self.best_k or 4  # Default

        logger.info("Huấn luyện K-Means với K=%d", k)

        X = df[self.SCALED_COLS].values

        self.best_model = KMeans(
            n_clusters=k,
            random_state=self.random_seed,
            n_init=self.n_init,
            max_iter=self.max_iter,
        )
        self.labels = self.best_model.fit_predict(X)

        logger.info("Phân bố cụm:")
        unique, counts = np.unique(self.labels, return_counts=True)
        for u, c in zip(unique, counts):
            logger.info("  Cụm %d: %d mẫu (%.1f%%)", u, c, c / len(self.labels) * 100)

        return self.labels

    def cluster_profiling(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        ĐIỂM CỐT LÕI: Tạo hồ sơ cụm (Cluster Profile).

        Tính Mean/Median của các đặc trưng GỐC theo từng cụm.
        Dựa trên profile, tự động gợi ý tên cho mỗi cụm.

        Parameters
        ----------
        df : pd.DataFrame
            DataFrame gốc.

        Returns
        -------
        pd.DataFrame
            DataFrame hồ sơ cụm với gợi ý tên.
        """
        logger.info("=" * 60)
        logger.info("TẠO HỒ SƠ CỤM (CLUSTER PROFILING)")
        logger.info("=" * 60)

        if self.labels is None:
            raise ValueError("Chưa fit model. Hãy gọi .fit() trước.")

        df_with_cluster = df.copy()
        df_with_cluster["Cluster"] = self.labels

        # Tính mean và median theo từng cụm
        profile_mean = df_with_cluster.groupby("Cluster")[self.ORIGINAL_COLS].mean()
        profile_median = df_with_cluster.groupby("Cluster")[self.ORIGINAL_COLS].median()

        # Tính thêm mode của Season và Precip Type cho mỗi cụm
        season_mode = df_with_cluster.groupby("Cluster")["Season"].agg(
            lambda x: x.mode().iloc[0] if not x.mode().empty else "N/A"
        )
        precip_mode = df_with_cluster.groupby("Cluster")["Precip Type"].agg(
            lambda x: x.mode().iloc[0] if not x.mode().empty else "N/A"
        )

        profile_mean["Season_Mode"] = season_mode
        profile_mean["Precip_Mode"] = precip_mode
        profile_mean["Count"] = df_with_cluster.groupby("Cluster").size()

        # Tự động gợi ý tên cụm
        cluster_names = []
        overall_mean = df[self.ORIGINAL_COLS].mean()

        for cluster_id in profile_mean.index:
            row = profile_mean.loc[cluster_id]
            name_parts = []

            # Nhiệt độ
            temp = row["Temperature (C)"]
            if temp < overall_mean["Temperature (C)"] - 5:
                name_parts.append("Lạnh")
            elif temp > overall_mean["Temperature (C)"] + 5:
                name_parts.append("Nóng")
            else:
                name_parts.append("Ôn hòa")

            # Độ ẩm
            hum = row["Humidity"]
            if hum > overall_mean["Humidity"] + 0.05:
                name_parts.append("Ẩm ướt")
            elif hum < overall_mean["Humidity"] - 0.05:
                name_parts.append("Hanh khô")
            else:
                name_parts.append("Độ ẩm TB")

            # Sức gió
            wind = row["Wind Speed (km/h)"]
            if wind > overall_mean["Wind Speed (km/h)"] + 3:
                name_parts.append("Gió mạnh")
            elif wind < overall_mean["Wind Speed (km/h)"] - 3:
                name_parts.append("Gió yếu")

            # Tầm nhìn
            vis = row["Visibility (km)"]
            if vis < overall_mean["Visibility (km)"] - 2:
                name_parts.append("Tầm nhìn thấp")

            # Chênh lệch nhiệt độ thực - cảm nhận
            temp_diff = abs(row["Temperature (C)"] - row["Apparent Temperature (C)"])
            if temp_diff > 5:
                name_parts.append("Chênh lệch nhiệt cao")

            cluster_name = f"Cụm {cluster_id}: {', '.join(name_parts)}"
            cluster_names.append(cluster_name)

            logger.info(
                "\n📊 %s\n"
                "   Nhiệt độ: %.1f°C | Cảm nhận: %.1f°C\n"
                "   Độ ẩm: %.1f%% | Gió: %.1f km/h\n"
                "   Tầm nhìn: %.1f km | Áp suất: %.0f mb\n"
                "   Mùa phổ biến: %s | Loại mưa: %s | Số mẫu: %d",
                cluster_name,
                row["Temperature (C)"], row["Apparent Temperature (C)"],
                row["Humidity"] * 100, row["Wind Speed (km/h)"],
                row["Visibility (km)"], row["Pressure (millibars)"],
                row["Season_Mode"], row["Precip_Mode"], row["Count"],
            )

        profile_mean["Cluster_Name"] = cluster_names
        self.cluster_profiles = profile_mean

        return profile_mean

    def plot_elbow(self, save_path: str = None) -> str:
        """Vẽ biểu đồ Elbow Method + Silhouette Score."""
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
        k_values = list(range(self.k_range[0], self.k_range[0] + len(self.inertias)))

        # Elbow
        ax1.plot(k_values, self.inertias, "bo-", linewidth=2, markersize=8)
        if self.best_k:
            idx = k_values.index(self.best_k)
            ax1.axvline(x=self.best_k, color="r", linestyle="--", alpha=0.7, label=f"K tối ưu = {self.best_k}")
        ax1.set_xlabel("Số cụm K", fontsize=12)
        ax1.set_ylabel("Inertia (SSE)", fontsize=12)
        ax1.set_title("Elbow Method", fontsize=14, fontweight="bold")
        ax1.legend()
        ax1.grid(True, alpha=0.3)

        # Silhouette
        ax2.plot(k_values, self.silhouette_scores, "rs-", linewidth=2, markersize=8)
        if self.best_k:
            idx = k_values.index(self.best_k)
            ax2.axvline(x=self.best_k, color="r", linestyle="--", alpha=0.7, label=f"K tối ưu = {self.best_k}")
        ax2.set_xlabel("Số cụm K", fontsize=12)
        ax2.set_ylabel("Silhouette Score", fontsize=12)
        ax2.set_title("Silhouette Score", fontsize=14, fontweight="bold")
        ax2.legend()
        ax2.grid(True, alpha=0.3)

        plt.tight_layout()

        if save_path is None:
            fig_dir = resolve_path(self.config["outputs"]["figures_dir"])
            os.makedirs(fig_dir, exist_ok=True)
            save_path = os.path.join(fig_dir, "clustering_elbow_silhouette.png")

        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        plt.close()
        logger.info("Đã lưu biểu đồ Elbow tại: %s", save_path)
        return save_path

    def plot_cluster_profile(self, save_path: str = None) -> str:
        """Vẽ biểu đồ radar (bar) so sánh profile các cụm."""
        if self.cluster_profiles is None:
            raise ValueError("Chưa có profile. Hãy gọi .cluster_profiling() trước.")

        profile = self.cluster_profiles[self.ORIGINAL_COLS].copy()

        # Normalize về [0, 1] để so sánh trực quan
        profile_norm = (profile - profile.min()) / (profile.max() - profile.min() + 1e-8)

        fig, ax = plt.subplots(figsize=(14, 6))
        profile_norm.T.plot(kind="bar", ax=ax, width=0.8, alpha=0.85)

        ax.set_xlabel("Đặc trưng", fontsize=12)
        ax.set_ylabel("Giá trị chuẩn hóa [0, 1]", fontsize=12)
        ax.set_title("So sánh Profile các Cụm Thời Tiết", fontsize=14, fontweight="bold")
        ax.legend(
            [self.cluster_profiles.loc[i, "Cluster_Name"]
             for i in self.cluster_profiles.index],
            loc="upper right", fontsize=8
        )
        ax.set_xticklabels(ax.get_xticklabels(), rotation=30, ha="right")
        ax.grid(axis="y", alpha=0.3)
        plt.tight_layout()

        if save_path is None:
            fig_dir = resolve_path(self.config["outputs"]["figures_dir"])
            os.makedirs(fig_dir, exist_ok=True)
            save_path = os.path.join(fig_dir, "clustering_profile.png")

        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        plt.close()
        logger.info("Đã lưu biểu đồ profile tại: %s", save_path)
        return save_path

    def run(self, df: pd.DataFrame) -> dict:
        """
        Chạy toàn bộ pipeline phân cụm.

        Parameters
        ----------
        df : pd.DataFrame
            DataFrame đã qua Feature Engineering (chứa cả cột scaled + gốc).

        Returns
        -------
        dict
            Kết quả gồm best_k, labels, profiles, plots.
        """
        # 1. Tìm K tối ưu
        best_k = self.find_optimal_k(df)

        # 2. Fit model với K tối ưu
        labels = self.fit(df, k=best_k)

        # 3. Tạo hồ sơ cụm
        profiles = self.cluster_profiling(df)

        # 4. Vẽ biểu đồ
        elbow_path = self.plot_elbow()
        profile_path = self.plot_cluster_profile()

        return {
            "best_k": best_k,
            "labels": labels,
            "profiles": profiles,
            "elbow_plot": elbow_path,
            "profile_plot": profile_path,
        }

    def save_results(self, output_dir: str = None) -> str:
        """Lưu hồ sơ cụm ra CSV."""
        if output_dir is None:
            output_dir = resolve_path(self.config["outputs"]["tables_dir"])
        os.makedirs(output_dir, exist_ok=True)

        if self.cluster_profiles is not None:
            save_path = os.path.join(output_dir, "cluster_profiles.csv")
            self.cluster_profiles.to_csv(save_path, encoding="utf-8-sig")
            logger.info("Đã lưu hồ sơ cụm tại: %s", save_path)
            return save_path

        return ""


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(levelname)s - %(message)s")

    config = load_config()
    parquet_path = resolve_path(config["data"]["cleaned_parquet"])
    df = pd.read_parquet(parquet_path)

    clusterer = WeatherClusterer(config)
    results = clusterer.run(df)
    clusterer.save_results()

    print(f"\n✅ Clustering hoàn tất! K tối ưu = {results['best_k']}")
