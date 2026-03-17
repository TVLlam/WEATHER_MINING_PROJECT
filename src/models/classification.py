"""
Module: WeatherClassifier
Phân loại thời tiết (target: Summary → gộp thành 5 nhóm chính).

ĐIỂM CỐT LÕI:
- Label Grouping: 27 nhãn Summary → 5 nhóm (Clear, Cloudy, Rain/Drizzle, Snow/Foggy, Breezy/Windy)
- class_weight='balanced' để xử lý mất cân bằng
- Baseline (LR, DT) + Cải tiến (RF, XGBoost)
- Metric: F1-macro (bắt buộc) + Cross-Validation
- Error Analysis: phân tích lỗi theo Season, giao mùa, cực trị
"""

import os
import logging
import warnings
import joblib

import numpy as np
import pandas as pd
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    f1_score,
    accuracy_score,
)
from sklearn.preprocessing import LabelEncoder

from src.data.loader import load_config, resolve_path

logger = logging.getLogger(__name__)


# ═══════════════════════════════════════════════════════════════
# LABEL GROUPING: 27 nhãn → 5 nhóm chính
# ═══════════════════════════════════════════════════════════════
LABEL_MAP = {
    # ☀️ Clear
    "Clear": "Clear",

    # ☁️ Cloudy (gộp Partly Cloudy, Mostly Cloudy, Overcast)
    "Partly Cloudy": "Cloudy",
    "Mostly Cloudy": "Cloudy",
    "Overcast": "Cloudy",
    "Dry": "Cloudy",
    "Dry and Partly Cloudy": "Cloudy",
    "Dry and Mostly Cloudy": "Cloudy",
    "Humid and Partly Cloudy": "Cloudy",
    "Humid and Mostly Cloudy": "Cloudy",

    # 🌧️ Rain/Drizzle
    "Rain": "Rain",
    "Drizzle": "Rain",
    "Light Rain": "Rain",

    # ❄️ Snow/Foggy
    "Foggy": "Foggy",
    "Breezy and Foggy": "Foggy",

    # 💨 Breezy/Windy
    "Breezy": "Windy",
    "Breezy and Mostly Cloudy": "Windy",
    "Breezy and Overcast": "Windy",
    "Breezy and Partly Cloudy": "Windy",
    "Windy and Mostly Cloudy": "Windy",
    "Windy and Overcast": "Windy",
    "Windy and Partly Cloudy": "Windy",
    "Windy": "Windy",
    "Windy and Foggy": "Windy",
    "Humid and Overcast": "Cloudy",
    "Dangerously Windy and Partly Cloudy": "Windy",
}


class WeatherClassifier:
    """
    Phân loại loại thời tiết (Summary → 5 nhóm) với nhiều mô hình.

    Parameters
    ----------
    config : dict, optional
        Dictionary config từ params.yaml.
    """

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
        self.cls_cfg = self.config.get("classification", {})
        self.target_col = self.cls_cfg.get("target_column", "Summary")
        self.test_size = self.cls_cfg.get("test_size", 0.2)
        self.random_seed = self.cls_cfg.get("random_seed", 42)
        self.cv_folds = self.cls_cfg.get("cv_folds", 5)

        self.X_train = None
        self.X_test = None
        self.y_train = None
        self.y_test = None
        self.df_test = None
        self.models = {}
        self.results = {}
        self.best_model_name = None
        self.label_encoder = LabelEncoder()

    def prepare_data(self, df: pd.DataFrame):
        """
        Chuẩn bị dữ liệu cho phân loại.
        BẮT BUỘC: Gộp nhãn Summary 27 lớp → 5 nhóm chính.
        """
        logger.info("Chuẩn bị dữ liệu phân loại (Target: '%s')", self.target_col)

        # ═══ LABEL GROUPING ═══
        df = df.copy()
        df["Weather_Group"] = df[self.target_col].map(LABEL_MAP)
        # Các nhãn không nằm trong map → gộp vào "Cloudy"
        df["Weather_Group"] = df["Weather_Group"].fillna("Cloudy")

        logger.info("LABEL GROUPING: 27 nhãn → 5 nhóm:")
        group_counts = df["Weather_Group"].value_counts()
        for grp, cnt in group_counts.items():
            logger.info("  %s: %d mẫu (%.1f%%)", grp, cnt, cnt / len(df) * 100)

        # Features & Target
        feature_cols = [c for c in self.FEATURE_COLS if c in df.columns]
        X = df[feature_cols].values
        y = df["Weather_Group"].values

        # Encode labels
        y_encoded = self.label_encoder.fit_transform(y)
        class_names = self.label_encoder.classes_
        logger.info("Số lớp sau gộp: %d — %s", len(class_names), list(class_names))

        # Stratified split
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(
            X, y_encoded,
            test_size=self.test_size,
            random_state=self.random_seed,
            stratify=y_encoded,
        )

        # Lưu df_test để phân tích lỗi
        test_idx = train_test_split(
            np.arange(len(df)), test_size=self.test_size,
            random_state=self.random_seed, stratify=y_encoded,
        )[1]
        self.df_test = df.iloc[test_idx].copy()

        logger.info("Train: %d | Test: %d", len(self.X_train), len(self.X_test))

    def _build_models(self):
        """Tạo dictionary các mô hình — BẮT BUỘC dùng class_weight='balanced'."""
        n_classes = len(self.label_encoder.classes_)

        self.models = {
            "Logistic Regression": LogisticRegression(
                max_iter=2000,
                random_state=self.random_seed,
                class_weight="balanced",
                C=1.0,
                solver="lbfgs",
            ),
            "Decision Tree": DecisionTreeClassifier(
                max_depth=15,
                random_state=self.random_seed,
                class_weight="balanced",
            ),
            "Random Forest": RandomForestClassifier(
                n_estimators=200,
                max_depth=20,
                random_state=self.random_seed,
                class_weight="balanced",
                n_jobs=2,
            ),
            "XGBoost": XGBClassifier(
                n_estimators=150,
                max_depth=6,
                learning_rate=0.1,
                random_state=self.random_seed,
                eval_metric="mlogloss",
                tree_method="hist",
                n_jobs=1,
            ),
        }

    def train_and_evaluate(self) -> pd.DataFrame:
        """Huấn luyện và đánh giá tất cả mô hình."""
        logger.info("=" * 60)
        logger.info("HUẤN LUYỆN VÀ ĐÁNH GIÁ MÔ HÌNH PHÂN LOẠI")
        logger.info("=" * 60)

        self._build_models()
        results_list = []

        for name, model in self.models.items():
            logger.info("\n--- %s ---", name)

            # Train
            model.fit(self.X_train, self.y_train)

            # Predict
            y_pred = model.predict(self.X_test)

            # Metrics
            f1 = f1_score(self.y_test, y_pred, average="macro")
            acc = accuracy_score(self.y_test, y_pred)

            # Cross-validation
            try:
                cv_n_jobs = 1 if name == "XGBoost" else -1
                cv_scores = cross_val_score(
                    model, self.X_train, self.y_train,
                    cv=self.cv_folds, scoring="f1_macro",
                    n_jobs=cv_n_jobs,
                )
                cv_mean = cv_scores.mean()
                cv_std = cv_scores.std()
            except Exception as e:
                logger.warning("  CV failed for %s: %s", name, e)
                cv_mean, cv_std = f1, 0.0

            logger.info("  F1-macro: %.4f | Accuracy: %.4f", f1, acc)
            logger.info("  CV F1-macro: %.4f ± %.4f", cv_mean, cv_std)

            results_list.append({
                "Model": name,
                "F1_macro": round(f1, 4),
                "Accuracy": round(acc, 4),
                "CV_F1_mean": round(cv_mean, 4),
                "CV_F1_std": round(cv_std, 4),
            })

            self.results[name] = {
                "model": model,
                "y_pred": y_pred,
                "f1": f1,
                "accuracy": acc,
            }

        results_df = pd.DataFrame(results_list).sort_values("F1_macro", ascending=False)

        # Best model
        self.best_model_name = results_df.iloc[0]["Model"]
        logger.info("\n🏆 Mô hình tốt nhất: %s (F1-macro = %.4f)",
                     self.best_model_name, results_df.iloc[0]["F1_macro"])

        return results_df

    def error_analysis(self, model_name: str = None) -> dict:
        """Phân tích lỗi chi tiết."""
        if model_name is None:
            model_name = self.best_model_name

        logger.info("=" * 60)
        logger.info("PHÂN TÍCH LỖI - %s", model_name)
        logger.info("=" * 60)

        y_pred = self.results[model_name]["y_pred"]
        y_true = self.y_test

        # Decode labels
        true_labels = self.label_encoder.inverse_transform(y_true)
        pred_labels = self.label_encoder.inverse_transform(y_pred)

        wrong_mask = true_labels != pred_labels
        n_wrong = wrong_mask.sum()
        n_total = len(y_true)
        error_rate = n_wrong / n_total

        logger.info("Tổng mẫu đoán sai: %d / %d (%.1f%%)", n_wrong, n_total, error_rate * 100)

        # DataFrame mẫu sai
        df_test_copy = self.df_test.copy()
        df_test_copy["True_Label"] = true_labels
        df_test_copy["Predicted"] = pred_labels
        wrong_df = df_test_copy[wrong_mask]

        # Phân tích lỗi theo Season
        logger.info("\n📊 Phân tích lỗi theo Mùa:")
        if "Season" in df_test_copy.columns:
            for season in ["Spring", "Summer", "Autumn", "Winter"]:
                mask = df_test_copy["Season"] == season
                total_s = mask.sum()
                wrong_s = (wrong_mask & mask.values).sum()
                pct = wrong_s / total_s * 100 if total_s > 0 else 0
                logger.info("  %s: %d sai / %d tổng (%.1f%%)", season, wrong_s, total_s, pct)

        # Giao mùa vs bình thường
        if "Month" in df_test_copy.columns:
            transition_months = [3, 6, 9, 12]
            trans_mask = df_test_copy["Month"].isin(transition_months)
            trans_total = trans_mask.sum()
            trans_wrong = (wrong_mask & trans_mask.values).sum()
            normal_total = (~trans_mask).sum()
            normal_wrong = (wrong_mask & (~trans_mask).values).sum()
            logger.info("\n🔄 So sánh lỗi: Giao mùa vs Bình thường:")
            logger.info("  Tháng giao mùa (%s): %.1f%% lỗi (%d/%d)",
                         transition_months, trans_wrong / trans_total * 100, trans_wrong, trans_total)
            logger.info("  Tháng bình thường:          %.1f%% lỗi (%d/%d)",
                         normal_wrong / normal_total * 100, normal_wrong, normal_total)

        # Cực trị
        if "Temperature (C)" in df_test_copy.columns:
            temp = df_test_copy["Temperature (C)"]
            q_low = temp.quantile(0.05)
            q_high = temp.quantile(0.95)
            extreme_mask = (temp <= q_low) | (temp >= q_high)
            ext_total = extreme_mask.sum()
            ext_wrong = (wrong_mask & extreme_mask.values).sum()
            norm_total = (~extreme_mask).sum()
            norm_wrong = (wrong_mask & (~extreme_mask).values).sum()
            logger.info("\n🌡️ Phân tích lỗi theo Thời tiết cực trị:")
            logger.info("  Nhiệt độ cực trị (≤%.1f°C hoặc ≥%.1f°C): %.1f%% lỗi (%d/%d)",
                         q_low, q_high, ext_wrong / ext_total * 100, ext_wrong, ext_total)
            logger.info("  Nhiệt độ bình thường:\n   %.1f%% lỗi (%d/%d)",
                         norm_wrong / norm_total * 100, norm_wrong, norm_total)

        return {
            "n_wrong": n_wrong,
            "n_total": n_total,
            "error_rate": error_rate,
            "wrong_predictions": wrong_df,
        }

    def plot_confusion_matrix(self, model_name: str = None, save_path: str = None) -> str:
        """Vẽ Confusion Matrix."""
        if model_name is None:
            model_name = self.best_model_name

        y_pred = self.results[model_name]["y_pred"]
        cm = confusion_matrix(self.y_test, y_pred)
        labels = self.label_encoder.classes_

        fig, ax = plt.subplots(figsize=(10, 8))
        sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
                    xticklabels=labels, yticklabels=labels, ax=ax)
        ax.set_xlabel("Dự đoán", fontsize=12)
        ax.set_ylabel("Thực tế", fontsize=12)
        ax.set_title(f"Confusion Matrix — {model_name}\n"
                     f"F1-macro = {self.results[model_name]['f1']:.4f}",
                     fontsize=14, fontweight="bold")
        plt.tight_layout()

        if save_path is None:
            fig_dir = resolve_path(self.config["outputs"]["figures_dir"])
            os.makedirs(fig_dir, exist_ok=True)
            save_path = os.path.join(fig_dir, "classification_confusion_matrix.png")
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        plt.close()
        logger.info("Đã lưu Confusion Matrix tại: %s", save_path)
        return save_path

    def plot_model_comparison(self, results_df: pd.DataFrame, save_path: str = None) -> str:
        """Vẽ biểu đồ so sánh các mô hình."""
        fig, axes = plt.subplots(1, 2, figsize=(14, 5))

        colors = ["#2196F3", "#4CAF50", "#FF9800", "#9C27B0"]
        x = range(len(results_df))
        models = results_df["Model"].values

        # F1-macro
        bars1 = axes[0].bar(x, results_df["F1_macro"], color=colors[:len(x)], alpha=0.85)
        for bar, val in zip(bars1, results_df["F1_macro"]):
            axes[0].text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.005,
                         f"{val:.4f}", ha="center", va="bottom", fontsize=10, fontweight="bold")
        axes[0].set_xticks(list(x))
        axes[0].set_xticklabels(models, rotation=20, ha="right")
        axes[0].set_ylabel("F1-macro")
        axes[0].set_title("So sánh F1-macro", fontsize=14, fontweight="bold")
        axes[0].grid(axis="y", alpha=0.3)
        axes[0].set_ylim(0, 1.0)

        # Accuracy
        bars2 = axes[1].bar(x, results_df["Accuracy"], color=colors[:len(x)], alpha=0.85)
        for bar, val in zip(bars2, results_df["Accuracy"]):
            axes[1].text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.005,
                         f"{val:.4f}", ha="center", va="bottom", fontsize=10, fontweight="bold")
        axes[1].set_xticks(list(x))
        axes[1].set_xticklabels(models, rotation=20, ha="right")
        axes[1].set_ylabel("Accuracy")
        axes[1].set_title("So sánh Accuracy", fontsize=14, fontweight="bold")
        axes[1].grid(axis="y", alpha=0.3)
        axes[1].set_ylim(0, 1.0)

        plt.suptitle("So sánh Hiệu năng các Mô hình Phân loại", fontsize=15, fontweight="bold")
        plt.tight_layout()

        if save_path is None:
            fig_dir = resolve_path(self.config["outputs"]["figures_dir"])
            os.makedirs(fig_dir, exist_ok=True)
            save_path = os.path.join(fig_dir, "classification_model_comparison.png")
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        plt.close()
        logger.info("Đã lưu biểu đồ so sánh tại: %s", save_path)
        return save_path

    def save_model(self, model_name: str = None):
        """Lưu model tốt nhất ra file."""
        if model_name is None:
            model_name = self.best_model_name

        model_dir = resolve_path(self.config["outputs"]["models_dir"])
        os.makedirs(model_dir, exist_ok=True)

        model = self.results[model_name]["model"]
        model_path = os.path.join(model_dir, "best_classifier.joblib")
        joblib.dump(model, model_path)

        # Lưu label encoder
        le_path = os.path.join(model_dir, "label_encoder.joblib")
        joblib.dump(self.label_encoder, le_path)

        # Lưu label map
        map_path = os.path.join(model_dir, "label_map.joblib")
        joblib.dump(LABEL_MAP, map_path)

        logger.info("Đã lưu model tại: %s", model_path)
        logger.info("Đã lưu label encoder tại: %s", le_path)
        return model_path

    def run(self, df: pd.DataFrame) -> dict:
        """Chạy toàn bộ pipeline phân loại."""
        # 1. Chuẩn bị (gộp nhãn)
        self.prepare_data(df)

        # 2. Train & Evaluate
        results_df = self.train_and_evaluate()

        # 3. Phân tích lỗi
        error = self.error_analysis()

        # 4. Plots
        cm_path = self.plot_confusion_matrix()
        comp_path = self.plot_model_comparison(results_df)

        # 5. Lưu model tốt nhất
        model_path = self.save_model()

        return {
            "results_df": results_df,
            "best_model": self.best_model_name,
            "error_analysis": error,
            "confusion_matrix_plot": cm_path,
            "comparison_plot": comp_path,
            "model_path": model_path,
        }

    def save_results(self, results_df: pd.DataFrame = None, output_dir: str = None):
        """Lưu kết quả ra CSV."""
        if output_dir is None:
            output_dir = resolve_path(self.config["outputs"]["tables_dir"])
        os.makedirs(output_dir, exist_ok=True)

        if results_df is not None:
            save_path = os.path.join(output_dir, "classification_results.csv")
            results_df.to_csv(save_path, index=False, encoding="utf-8-sig")
            logger.info("Đã lưu kết quả phân loại tại: %s", save_path)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(levelname)s - %(message)s")

    config = load_config()
    parquet_path = resolve_path(config["data"]["cleaned_parquet"])
    df = pd.read_parquet(parquet_path)

    classifier = WeatherClassifier(config)
    results = classifier.run(df)
    classifier.save_results(results["results_df"])

    print(f"\n✅ Classification hoàn tất! Best: {results['best_model']}")
    print(results["results_df"])
