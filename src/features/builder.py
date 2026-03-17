"""
Module: FeatureBuilder
Trích xuất và xây dựng đặc trưng (Feature Engineering) cho dữ liệu thời tiết.

Các chức năng chính:
- Parse datetime và sort theo thời gian
- Trích xuất Month, Season (Xuân/Hạ/Thu/Đông)
- Rời rạc hóa (Binning) các cột liên tục → Category
- Chuẩn hóa (StandardScaler) cho các biến liên tục
"""

import os
import logging

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler

from src.data.loader import load_config, resolve_path

logger = logging.getLogger(__name__)


class FeatureBuilder:
    """
    Class xây dựng đặc trưng từ dữ liệu thời tiết đã làm sạch.
    
    Parameters
    ----------
    config : dict, optional
        Dictionary config từ params.yaml.
    """
    
    # Các cột số liên tục dùng cho scale
    NUMERIC_COLS = [
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
        self.scaler = StandardScaler()
        self._season_map = self._build_season_map()
    
    def _build_season_map(self) -> dict:
        """Xây dựng mapping tháng → mùa từ config."""
        season_cfg = self.config["features"]["season_mapping"]
        month_to_season = {}
        for season_name, months in season_cfg.items():
            for m in months:
                month_to_season[m] = season_name.capitalize()
        return month_to_season
    
    def parse_datetime(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Ép kiểu cột 'Formatted Date' sang Datetime và sort theo thời gian.
        
        Returns
        -------
        pd.DataFrame
            DataFrame đã được sort theo thời gian.
        """
        logger.info("Ép kiểu 'Formatted Date' → datetime và sort theo thời gian")
        df = df.copy()
        df["Formatted Date"] = pd.to_datetime(df["Formatted Date"], utc=True)
        df = df.sort_values("Formatted Date").reset_index(drop=True)
        return df
    
    def extract_time_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Trích xuất các đặc trưng thời gian:
        - Year, Month, Day, Hour
        - Season (Xuân, Hạ, Thu, Đông)
        """
        logger.info("Trích xuất đặc trưng thời gian: Year, Month, Hour, Season")
        df = df.copy()
        
        dt = df["Formatted Date"]
        df["Year"] = dt.dt.year
        df["Month"] = dt.dt.month
        df["Day"] = dt.dt.day
        df["Hour"] = dt.dt.hour
        
        # Tạo cột Season dựa theo mapping từ config
        df["Season"] = df["Month"].map(self._season_map)
        
        logger.info("Phân bố Season:\n%s", df["Season"].value_counts().to_string())
        return df
    
    def discretize_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Rời rạc hóa (Binning) các cột Nhiệt độ, Độ ẩm, Sức gió
        thành Category (Low, Normal, High) để dùng cho thuật toán Apriori.
        
        Sử dụng pd.qcut (quantile-based) để chia đều số lượng mẫu
        vào mỗi bin.
        
        Returns
        -------
        pd.DataFrame
            DataFrame có thêm các cột binned: Temp_Bin, Humidity_Bin, Wind_Bin
        """
        logger.info("Rời rạc hóa (Binning) các đặc trưng liên tục")
        df = df.copy()
        
        feat_cfg = self.config["features"]
        
        # Binning Nhiệt độ
        temp_labels = feat_cfg["temperature_bins"]["labels"]
        temp_nbins = feat_cfg["temperature_bins"]["num_bins"]
        df["Temp_Bin"] = pd.qcut(
            df["Temperature (C)"],
            q=temp_nbins,
            labels=temp_labels,
            duplicates="drop"
        )
        
        # Binning Độ ẩm
        hum_labels = feat_cfg["humidity_bins"]["labels"]
        hum_nbins = feat_cfg["humidity_bins"]["num_bins"]
        df["Humidity_Bin"] = pd.qcut(
            df["Humidity"],
            q=hum_nbins,
            labels=hum_labels,
            duplicates="drop"
        )
        
        # Binning Sức gió
        wind_labels = feat_cfg["wind_speed_bins"]["labels"]
        wind_nbins = feat_cfg["wind_speed_bins"]["num_bins"]
        df["Wind_Bin"] = pd.qcut(
            df["Wind Speed (km/h)"],
            q=wind_nbins,
            labels=wind_labels,
            duplicates="drop"
        )
        
        logger.info("Binning hoàn tất:")
        logger.info("  Temp_Bin:     %s", df["Temp_Bin"].value_counts().to_dict())
        logger.info("  Humidity_Bin: %s", df["Humidity_Bin"].value_counts().to_dict())
        logger.info("  Wind_Bin:     %s", df["Wind_Bin"].value_counts().to_dict())
        
        return df
    
    def scale_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Chuẩn hóa (StandardScaler) các biến số liên tục.
        Kết quả: mean ≈ 0, std ≈ 1 → Phù hợp cho K-Means.
        
        Tạo các cột mới có suffix '_scaled' để giữ nguyên cột gốc.
        
        Returns
        -------
        pd.DataFrame
            DataFrame có thêm các cột scaled.
        """
        logger.info("Chuẩn hóa (StandardScaler) các biến số liên tục")
        df = df.copy()
        
        # Chỉ scale các cột có trong DataFrame
        cols_to_scale = [c for c in self.NUMERIC_COLS if c in df.columns]
        
        scaled_values = self.scaler.fit_transform(df[cols_to_scale])
        scaled_cols = [f"{c}_scaled" for c in cols_to_scale]
        
        df_scaled = pd.DataFrame(
            scaled_values,
            columns=scaled_cols,
            index=df.index
        )
        
        df = pd.concat([df, df_scaled], axis=1)
        logger.info("Đã thêm %d cột scaled: %s", len(scaled_cols), scaled_cols)
        
        return df
    
    def run(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Chạy toàn bộ pipeline Feature Engineering.
        
        Parameters
        ----------
        df : pd.DataFrame
            DataFrame đã được làm sạch (output từ WeatherCleaner).
        
        Returns
        -------
        pd.DataFrame
            DataFrame với đầy đủ đặc trưng mới.
        """
        logger.info("=" * 60)
        logger.info("BẮT ĐẦU PIPELINE FEATURE ENGINEERING")
        logger.info("=" * 60)
        
        # 1. Parse datetime và sort
        df = self.parse_datetime(df)
        
        # 2. Trích xuất đặc trưng thời gian + Season
        df = self.extract_time_features(df)
        
        # 3. Rời rạc hóa
        df = self.discretize_features(df)
        
        # 4. Chuẩn hóa
        df = self.scale_features(df)
        
        logger.info("Shape dữ liệu sau Feature Engineering: %s", df.shape)
        logger.info("Các cột: %s", df.columns.tolist())
        logger.info("HOÀN TẤT pipeline Feature Engineering!")
        
        return df
    
    def save(self, df: pd.DataFrame, fmt: str = "parquet") -> str:
        """
        Lưu DataFrame đã xử lý đặc trưng.
        
        Parameters
        ----------
        df : pd.DataFrame
            DataFrame cần lưu.
        fmt : str
            'parquet' hoặc 'csv'.
        
        Returns
        -------
        str
            Đường dẫn file đã lưu.
        """
        processed_dir = resolve_path(self.config["data"]["processed_dir"])
        os.makedirs(processed_dir, exist_ok=True)
        
        if fmt == "parquet":
            save_path = resolve_path(self.config["data"]["cleaned_parquet"])
            df.to_parquet(save_path, index=False)
        else:
            save_path = resolve_path(self.config["data"]["cleaned_csv"])
            df.to_csv(save_path, index=False)
        
        logger.info("Đã lưu dữ liệu Feature Engineering tại: %s", save_path)
        return save_path


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(levelname)s - %(message)s")
    
    from src.data.cleaner import WeatherCleaner
    
    # Pipeline: Clean → Feature Engineering
    cleaner = WeatherCleaner()
    df_clean = cleaner.run()
    
    builder = FeatureBuilder()
    df_features = builder.run(df_clean)
    save_path = builder.save(df_features, fmt="parquet")
    
    print(f"\n✅ Feature Engineering hoàn tất: {df_features.shape}")
    print(f"📁 Đã lưu tại: {save_path}")
    print(df_features.head())
