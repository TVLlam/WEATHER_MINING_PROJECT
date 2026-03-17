"""
Module: TimeForecaster
Dự báo chuỗi thời gian nhiệt độ (Temperature).

ĐIỂM CỐT LÕI:
- Split Train/Test THEO THỜI GIAN (không shuffle!)
- ACF/PACF để xác định tính mùa vụ
- Baseline Naive vs ARIMA (Holt-Winters)
- Đánh giá MAE, RMSE
- Phân tích phần dư (Residual Analysis)
"""

import os
import logging
import warnings
from typing import Dict, Tuple

import numpy as np
import pandas as pd
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import mean_absolute_error, mean_squared_error

from src.data.loader import load_config, resolve_path

logger = logging.getLogger(__name__)

# Tắt cảnh báo convergence của statsmodels
warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=FutureWarning)


class TimeForecaster:
    """
    Dự báo nhiệt độ theo chuỗi thời gian.

    Parameters
    ----------
    config : dict, optional
        Dictionary config từ params.yaml.
    """

    def __init__(self, config: dict = None):
        self.config = config or load_config()
        self.fc_cfg = self.config["forecasting"]
        self.target_col = self.fc_cfg["target_column"]
        self.train_ratio = self.fc_cfg["train_ratio"]

        self.df_daily = None
        self.train = None
        self.test = None
        self.results = {}

    def prepare_time_series(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Chuẩn bị chuỗi thời gian: tổng hợp theo ngày, tính trung bình.

        Parameters
        ----------
        df : pd.DataFrame
            DataFrame gốc (hourly data).

        Returns
        -------
        pd.DataFrame
            DataFrame daily với index là datetime.
        """
        logger.info("Chuẩn bị chuỗi thời gian (tổng hợp theo ngày)")

        df_ts = df[["Formatted Date", self.target_col]].copy()
        df_ts["Formatted Date"] = pd.to_datetime(df_ts["Formatted Date"], utc=True)
        df_ts = df_ts.set_index("Formatted Date")

        # Resampling theo ngày: lấy trung bình
        self.df_daily = df_ts.resample("D").mean().dropna()
        self.df_daily.index = self.df_daily.index.tz_localize(None)

        logger.info("Chuỗi thời gian: %d ngày (từ %s đến %s)",
                     len(self.df_daily),
                     self.df_daily.index.min().strftime("%Y-%m-%d"),
                     self.df_daily.index.max().strftime("%Y-%m-%d"))

        return self.df_daily

    def split_time_series(self) -> Tuple[pd.Series, pd.Series]:
        """
        BẮT BUỘC: Split Train/Test theo thứ tự thời gian.
        TUYỆT ĐỐI KHÔNG shuffle!
        """
        n = len(self.df_daily)
        split_idx = int(n * self.train_ratio)

        self.train = self.df_daily.iloc[:split_idx][self.target_col]
        self.test = self.df_daily.iloc[split_idx:][self.target_col]

        logger.info("Split theo thời gian (KHÔNG shuffle):")
        logger.info("  Train: %d ngày (%s → %s)",
                     len(self.train),
                     self.train.index.min().strftime("%Y-%m-%d"),
                     self.train.index.max().strftime("%Y-%m-%d"))
        logger.info("  Test:  %d ngày (%s → %s)",
                     len(self.test),
                     self.test.index.min().strftime("%Y-%m-%d"),
                     self.test.index.max().strftime("%Y-%m-%d"))

        return self.train, self.test

    def plot_acf_pacf(self, save_path: str = None) -> str:
        """
        ĐIỂM 10: Vẽ biểu đồ ACF/PACF để xác định tính tự tương quan
        và chu kỳ mùa vụ.
        """
        from statsmodels.graphics.tsaplots import plot_acf, plot_pacf

        n_lags = self.fc_cfg.get("n_lags_acf", 50)

        fig, axes = plt.subplots(2, 1, figsize=(14, 8))

        plot_acf(self.train, lags=n_lags, ax=axes[0], title="")
        axes[0].set_title("Autocorrelation Function (ACF)", fontsize=14, fontweight="bold")
        axes[0].set_xlabel("Lag (ngày)")
        axes[0].set_ylabel("ACF")

        plot_pacf(self.train, lags=n_lags, ax=axes[1], method="ywm", title="")
        axes[1].set_title("Partial Autocorrelation Function (PACF)", fontsize=14, fontweight="bold")
        axes[1].set_xlabel("Lag (ngày)")
        axes[1].set_ylabel("PACF")

        plt.tight_layout()

        if save_path is None:
            fig_dir = resolve_path(self.config["outputs"]["figures_dir"])
            os.makedirs(fig_dir, exist_ok=True)
            save_path = os.path.join(fig_dir, "forecasting_acf_pacf.png")

        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        plt.close()
        logger.info("Đã lưu ACF/PACF tại: %s", save_path)
        return save_path

    def naive_forecast(self) -> Dict:
        """
        Baseline Naive: Dự báo ngày mai = giá trị hôm nay.
        """
        logger.info("\n--- Naive Forecast (Baseline) ---")

        # Naive: shift 1 ngày
        y_pred_naive = self.test.shift(1).dropna()
        y_true_naive = self.test.iloc[1:]

        mae = mean_absolute_error(y_true_naive, y_pred_naive)
        rmse = np.sqrt(mean_squared_error(y_true_naive, y_pred_naive))

        logger.info("  MAE:  %.4f°C", mae)
        logger.info("  RMSE: %.4f°C", rmse)

        self.results["Naive"] = {
            "y_pred": y_pred_naive,
            "y_true": y_true_naive,
            "mae": mae,
            "rmse": rmse,
            "residuals": y_true_naive.values - y_pred_naive.values,
        }

        return self.results["Naive"]

    def arima_forecast(self) -> Dict:
        """
        Mô hình ARIMA / Holt-Winters Exponential Smoothing.
        """
        from statsmodels.tsa.holtwinters import ExponentialSmoothing

        logger.info("\n--- Holt-Winters Exponential Smoothing ---")

        try:
            model = ExponentialSmoothing(
                self.train,
                seasonal_periods=365,
                trend="add",
                seasonal="add",
                use_boxcox=False,
            )
            fitted = model.fit(optimized=True)
            y_pred = fitted.forecast(steps=len(self.test))
            y_pred.index = self.test.index

        except Exception as e:
            logger.warning("Holt-Winters thất bại (%s), dùng Simple Exponential Smoothing", e)
            from statsmodels.tsa.holtwinters import SimpleExpSmoothing
            model = SimpleExpSmoothing(self.train)
            fitted = model.fit(optimized=True)
            y_pred = fitted.forecast(steps=len(self.test))
            y_pred.index = self.test.index

        mae = mean_absolute_error(self.test, y_pred)
        rmse = np.sqrt(mean_squared_error(self.test, y_pred))

        logger.info("  MAE:  %.4f°C", mae)
        logger.info("  RMSE: %.4f°C", rmse)

        self.results["Holt-Winters"] = {
            "y_pred": y_pred,
            "y_true": self.test,
            "mae": mae,
            "rmse": rmse,
            "residuals": self.test.values - y_pred.values,
        }

        return self.results["Holt-Winters"]

    def residual_analysis(self, model_name: str = "Holt-Winters", save_path: str = None) -> Dict:
        """
        ĐIỂM 10 TIÊU CHÍ G: Phân tích phần dư (Residual).

        - Vẽ biểu đồ line/histogram của phần dư
        - Xác định các điểm Outlier (dự báo lệch quá xa)

        Parameters
        ----------
        model_name : str
            Tên mô hình để phân tích.

        Returns
        -------
        dict
            Kết quả phân tích gồm residuals, outliers.
        """
        logger.info("=" * 60)
        logger.info("PHÂN TÍCH PHẦN DƯ - %s", model_name)
        logger.info("=" * 60)

        result = self.results[model_name]
        residuals = result["residuals"]
        y_true = result["y_true"]

        # Thống kê phần dư
        logger.info("  Mean Residual: %.4f°C", np.mean(residuals))
        logger.info("  Std  Residual: %.4f°C", np.std(residuals))
        logger.info("  Max  Residual: %.4f°C", np.max(np.abs(residuals)))

        # Xác định Outlier: |residual| > 2 * std
        threshold = 2 * np.std(residuals)
        outlier_mask = np.abs(residuals) > threshold
        n_outliers = outlier_mask.sum()

        logger.info("\n  Ngưỡng Outlier: ±%.2f°C (2σ)", threshold)
        logger.info("  Số ngày Outlier: %d / %d (%.1f%%)",
                     n_outliers, len(residuals), n_outliers / len(residuals) * 100)

        # Tạo DataFrame outlier
        if isinstance(y_true, pd.Series):
            outlier_dates = y_true.index[outlier_mask]
            df_outliers = pd.DataFrame({
                "Date": outlier_dates,
                "Actual": y_true.values[outlier_mask],
                "Predicted": result["y_pred"].values[outlier_mask] if isinstance(result["y_pred"], pd.Series) else result["y_pred"][outlier_mask],
                "Residual": residuals[outlier_mask],
            })
        else:
            df_outliers = pd.DataFrame({
                "Actual": y_true[outlier_mask] if hasattr(y_true, '__getitem__') else [],
                "Residual": residuals[outlier_mask],
            })

        if len(df_outliers) > 0:
            logger.info("\n  Top 5 ngày dự báo lệch xa nhất:")
            top5 = df_outliers.reindex(df_outliers["Residual"].abs().sort_values(ascending=False).index).head(5)
            for _, row in top5.iterrows():
                date_str = row["Date"].strftime("%Y-%m-%d") if "Date" in row and hasattr(row["Date"], "strftime") else "N/A"
                logger.info("    %s: Thực tế=%.1f°C, Dự báo=%.1f°C, Sai lệch=%.1f°C",
                             date_str, row["Actual"], row["Predicted"], row["Residual"])

        # Vẽ biểu đồ
        fig, axes = plt.subplots(2, 2, figsize=(16, 10))

        # 1. Residual Line Plot
        axes[0, 0].plot(residuals, alpha=0.7, linewidth=0.8, color="#2196F3")
        axes[0, 0].axhline(y=0, color="r", linestyle="--", alpha=0.5)
        axes[0, 0].axhline(y=threshold, color="orange", linestyle=":", alpha=0.7, label=f"+2σ = {threshold:.1f}°C")
        axes[0, 0].axhline(y=-threshold, color="orange", linestyle=":", alpha=0.7, label=f"-2σ = {-threshold:.1f}°C")
        axes[0, 0].set_title("Phần dư theo thời gian", fontsize=12, fontweight="bold")
        axes[0, 0].set_xlabel("Ngày")
        axes[0, 0].set_ylabel("Residual (°C)")
        axes[0, 0].legend(fontsize=9)
        axes[0, 0].grid(alpha=0.3)

        # 2. Residual Histogram
        axes[0, 1].hist(residuals, bins=50, alpha=0.75, color="#4CAF50", edgecolor="white")
        axes[0, 1].axvline(x=0, color="r", linestyle="--", alpha=0.7)
        axes[0, 1].set_title("Phân bố Phần dư", fontsize=12, fontweight="bold")
        axes[0, 1].set_xlabel("Residual (°C)")
        axes[0, 1].set_ylabel("Tần suất")
        axes[0, 1].grid(alpha=0.3)

        # 3. Actual vs Predicted
        y_pred_values = result["y_pred"].values if isinstance(result["y_pred"], pd.Series) else result["y_pred"]
        y_true_values = y_true.values if isinstance(y_true, pd.Series) else y_true
        axes[1, 0].plot(y_true_values[-200:], label="Thực tế", alpha=0.9, linewidth=1.2)
        axes[1, 0].plot(y_pred_values[-200:], label="Dự báo", alpha=0.8, linewidth=1.2, linestyle="--")
        axes[1, 0].set_title("Thực tế vs Dự báo (200 ngày cuối)", fontsize=12, fontweight="bold")
        axes[1, 0].set_xlabel("Ngày")
        axes[1, 0].set_ylabel("Nhiệt độ (°C)")
        axes[1, 0].legend()
        axes[1, 0].grid(alpha=0.3)

        # 4. Q-Q plot / Scatter actual vs predicted
        axes[1, 1].scatter(y_true_values, y_pred_values, alpha=0.3, s=10, color="#9C27B0")
        min_val = min(y_true_values.min(), y_pred_values.min())
        max_val = max(y_true_values.max(), y_pred_values.max())
        axes[1, 1].plot([min_val, max_val], [min_val, max_val], "r--", alpha=0.7, label="Perfect Forecast")
        axes[1, 1].set_title("Scatter: Thực tế vs Dự báo", fontsize=12, fontweight="bold")
        axes[1, 1].set_xlabel("Nhiệt độ Thực tế (°C)")
        axes[1, 1].set_ylabel("Nhiệt độ Dự báo (°C)")
        axes[1, 1].legend()
        axes[1, 1].grid(alpha=0.3)

        plt.suptitle(f"Phân tích Phần dư — {model_name}", fontsize=15, fontweight="bold", y=1.02)
        plt.tight_layout()

        if save_path is None:
            fig_dir = resolve_path(self.config["outputs"]["figures_dir"])
            os.makedirs(fig_dir, exist_ok=True)
            save_path = os.path.join(fig_dir, "forecasting_residual_analysis.png")

        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        plt.close()
        logger.info("Đã lưu biểu đồ phần dư tại: %s", save_path)

        return {
            "residuals": residuals,
            "outlier_mask": outlier_mask,
            "n_outliers": n_outliers,
            "outlier_df": df_outliers,
            "threshold": threshold,
            "plot_path": save_path,
        }

    def plot_forecast_comparison(self, save_path: str = None) -> str:
        """Vẽ biểu đồ so sánh Naive vs Holt-Winters."""
        fig, ax = plt.subplots(figsize=(14, 6))

        ax.plot(self.test.index, self.test.values,
                label="Thực tế", alpha=0.9, linewidth=1.5, color="#333")

        for name, result in self.results.items():
            y_pred = result["y_pred"]
            if isinstance(y_pred, pd.Series):
                ax.plot(y_pred.index, y_pred.values,
                        label=f"{name} (MAE={result['mae']:.2f}°C)",
                        alpha=0.7, linewidth=1.2, linestyle="--")
            else:
                ax.plot(self.test.index[:len(y_pred)], y_pred,
                        label=f"{name} (MAE={result['mae']:.2f}°C)",
                        alpha=0.7, linewidth=1.2, linestyle="--")

        ax.set_xlabel("Thời gian", fontsize=12)
        ax.set_ylabel("Nhiệt độ (°C)", fontsize=12)
        ax.set_title("So sánh Dự báo Nhiệt độ: Naive vs Holt-Winters",
                     fontsize=14, fontweight="bold")
        ax.legend(fontsize=10)
        ax.grid(alpha=0.3)
        plt.tight_layout()

        if save_path is None:
            fig_dir = resolve_path(self.config["outputs"]["figures_dir"])
            os.makedirs(fig_dir, exist_ok=True)
            save_path = os.path.join(fig_dir, "forecasting_comparison.png")

        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        plt.close()
        logger.info("Đã lưu biểu đồ so sánh dự báo tại: %s", save_path)
        return save_path

    def run(self, df: pd.DataFrame) -> Dict:
        """
        Chạy toàn bộ pipeline dự báo chuỗi thời gian.

        Parameters
        ----------
        df : pd.DataFrame
            DataFrame gốc.

        Returns
        -------
        dict
            Kết quả tổng hợp.
        """
        # 1. Chuẩn bị chuỗi thời gian
        self.prepare_time_series(df)

        # 2. Split theo thời gian
        self.split_time_series()

        # 3. ACF/PACF
        acf_path = self.plot_acf_pacf()

        # 4. Naive Forecast
        self.naive_forecast()

        # 5. Holt-Winters / ARIMA
        self.arima_forecast()

        # 6. So sánh kết quả
        results_summary = []
        for name, r in self.results.items():
            results_summary.append({
                "Model": name,
                "MAE": round(r["mae"], 4),
                "RMSE": round(r["rmse"], 4),
            })
        results_df = pd.DataFrame(results_summary)
        logger.info("\nSo sánh mô hình dự báo:\n%s", results_df.to_string(index=False))

        # 7. Phân tích phần dư (Holt-Winters)
        residual_info = self.residual_analysis("Holt-Winters")

        # 8. Vẽ so sánh forecast
        comparison_path = self.plot_forecast_comparison()

        return {
            "results_df": results_df,
            "residual_analysis": residual_info,
            "acf_plot": acf_path,
            "comparison_plot": comparison_path,
        }

    def save_results(self, output_dir: str = None) -> str:
        """Lưu kết quả ra CSV."""
        if output_dir is None:
            output_dir = resolve_path(self.config["outputs"]["tables_dir"])
        os.makedirs(output_dir, exist_ok=True)

        results_list = []
        for name, r in self.results.items():
            results_list.append({"Model": name, "MAE": r["mae"], "RMSE": r["rmse"]})

        if results_list:
            df_results = pd.DataFrame(results_list)
            save_path = os.path.join(output_dir, "forecasting_results.csv")
            df_results.to_csv(save_path, index=False, encoding="utf-8-sig")
            logger.info("Đã lưu kết quả dự báo tại: %s", save_path)
            return save_path
        return ""


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(levelname)s - %(message)s")

    config = load_config()
    parquet_path = resolve_path(config["data"]["cleaned_parquet"])
    df = pd.read_parquet(parquet_path)

    forecaster = TimeForecaster(config)
    results = forecaster.run(df)
    forecaster.save_results()

    print("\n✅ Forecasting hoàn tất!")
    print(results["results_df"])
