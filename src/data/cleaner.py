"""
Module: WeatherCleaner
Tiền xử lý và làm sạch dữ liệu thời tiết Szeged Hungary 2006-2016.

Xử lý các vấn đề:
- Bẫy 1: Cột 'Loud Cover' toàn giá trị 0 → Drop
- Bẫy 2: Missing Values ở cột 'Precip Type' → FillNA
- Bẫy 3: Outlier cột 'Pressure (millibars)' = 0 → Thay bằng Median
"""

import os
import logging

import numpy as np
import pandas as pd

from src.data.loader import load_config, resolve_path

logger = logging.getLogger(__name__)


class WeatherCleaner:
    """
    Class xử lý và làm sạch dữ liệu thời tiết.
    
    Parameters
    ----------
    config : dict, optional
        Dictionary config từ params.yaml. Nếu None sẽ tự load.
    """
    
    def __init__(self, config: dict = None):
        self.config = config or load_config()
        self.raw_path = resolve_path(self.config["data"]["raw_path"])
        self.processed_dir = resolve_path(self.config["data"]["processed_dir"])
        self.df_raw = None
        self.df_cleaned = None
    
    def load_raw_data(self) -> pd.DataFrame:
        """Đọc dữ liệu thô từ CSV."""
        logger.info("Đọc dữ liệu thô từ: %s", self.raw_path)
        self.df_raw = pd.read_csv(self.raw_path)
        logger.info("Shape dữ liệu thô: %s", self.df_raw.shape)
        return self.df_raw
    
    def drop_loud_cover(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        BẪY 1: Drop cột 'Loud Cover'.
        Cột này toàn bộ giá trị = 0, không có ý nghĩa phân tích.
        """
        if "Loud Cover" in df.columns:
            logger.info("Drop cột 'Loud Cover' (toàn bộ = 0, vô nghĩa)")
            df = df.drop(columns=["Loud Cover"])
        return df
    
    def handle_missing_precip_type(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        BẪY 2: Xử lý Missing Values ở cột 'Precip Type'.
        Điền NaN bằng giá trị phổ biến nhất (mode) hoặc theo config.
        """
        col = "Precip Type"
        n_missing = df[col].isna().sum()
        
        if n_missing > 0:
            fill_value = self.config["preprocessing"].get("precip_type_fill", None)
            
            if fill_value is None:
                # Dùng mode (giá trị xuất hiện nhiều nhất)
                fill_value = df[col].mode()[0]
            
            logger.info(
                "Xử lý %d missing values ở '%s' → Fill bằng '%s'",
                n_missing, col, fill_value
            )
            df[col] = df[col].fillna(fill_value)
        
        return df
    
    def handle_pressure_outliers(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        BẪY 3: Xử lý Outlier cột 'Pressure (millibars)'.
        Giá trị = 0 là dữ liệu lỗi máy đo → Chuyển thành NaN → Fill Median.
        """
        col = "Pressure (millibars)"
        n_zeros = (df[col] == 0).sum()
        
        if n_zeros > 0:
            logger.info(
                "Xử lý %d outlier ở '%s' (giá trị = 0 → NaN → Median)",
                n_zeros, col
            )
            # Chuyển giá trị 0 thành NaN
            df[col] = df[col].replace(0, np.nan)
            # Fill bằng Median của các giá trị hợp lệ
            median_val = df[col].median()
            df[col] = df[col].fillna(median_val)
            logger.info("Median Pressure sử dụng: %.2f millibars", median_val)
        
        return df
    
    def drop_duplicates(self, df: pd.DataFrame) -> pd.DataFrame:
        """Loại bỏ các dòng trùng lặp."""
        n_before = len(df)
        df = df.drop_duplicates()
        n_after = len(df)
        n_dropped = n_before - n_after
        if n_dropped > 0:
            logger.info("Đã loại bỏ %d dòng trùng lặp", n_dropped)
        return df
    
    def run(self) -> pd.DataFrame:
        """
        Chạy toàn bộ pipeline làm sạch dữ liệu.
        
        Returns
        -------
        pd.DataFrame
            DataFrame đã được làm sạch.
        """
        logger.info("=" * 60)
        logger.info("BẮT ĐẦU PIPELINE LÀM SẠCH DỮ LIỆU")
        logger.info("=" * 60)
        
        # 1. Load dữ liệu thô
        df = self.load_raw_data()
        
        # 2. Drop duplicates
        df = self.drop_duplicates(df)
        
        # 3. BẪY 1: Drop cột Loud Cover
        df = self.drop_loud_cover(df)
        
        # 4. BẪY 2: Xử lý Missing Precip Type  
        df = self.handle_missing_precip_type(df)
        
        # 5. BẪY 3: Xử lý Pressure Outliers
        df = self.handle_pressure_outliers(df)
        
        self.df_cleaned = df
        
        logger.info("Shape dữ liệu sau làm sạch: %s", df.shape)
        logger.info("Missing values còn lại:\n%s", df.isnull().sum())
        logger.info("HOÀN TẤT pipeline làm sạch dữ liệu!")
        
        return df
    
    def save(self, df: pd.DataFrame = None, fmt: str = "parquet") -> str:
        """
        Lưu DataFrame đã làm sạch.
        
        Parameters
        ----------
        df : pd.DataFrame, optional
            DataFrame cần lưu. Mặc định dùng self.df_cleaned.
        fmt : str
            Định dạng file: 'parquet' hoặc 'csv'.
        
        Returns
        -------
        str
            Đường dẫn file đã lưu.
        """
        if df is None:
            df = self.df_cleaned
        
        if df is None:
            raise ValueError("Không có dữ liệu để lưu. Hãy chạy .run() trước.")
        
        os.makedirs(self.processed_dir, exist_ok=True)
        
        if fmt == "parquet":
            save_path = resolve_path(self.config["data"]["cleaned_parquet"])
            df.to_parquet(save_path, index=False)
        else:
            save_path = resolve_path(self.config["data"]["cleaned_csv"])
            df.to_csv(save_path, index=False)
        
        logger.info("Đã lưu dữ liệu sạch tại: %s", save_path)
        return save_path


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(levelname)s - %(message)s")
    cleaner = WeatherCleaner()
    df_clean = cleaner.run()
    cleaner.save(df_clean, fmt="parquet")
    print(f"\n✅ Dữ liệu sạch: {df_clean.shape}")
    print(df_clean.head())