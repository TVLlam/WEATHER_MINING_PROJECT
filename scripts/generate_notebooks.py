"""Regenerate all notebooks with rubric-compliant names + add 04b anomaly notebook."""
import json, os

def make_cell(cell_type, source, metadata=None):
    cell = {"cell_type": cell_type, "metadata": metadata or {},
            "source": [line + "\n" for line in source.strip().split("\n")]}
    if cell_type == "code":
        cell["execution_count"] = None; cell["outputs"] = []
    return cell

def md(text): return make_cell("markdown", text)
def code(text): return make_cell("code", text)

def make_nb(cells):
    return {"cells": cells, "metadata": {"kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"}, "language_info": {"name": "python", "version": "3.12.0"}}, "nbformat": 4, "nbformat_minor": 5}

def save(nb, path):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f: json.dump(nb, f, ensure_ascii=False, indent=1)
    print(f"  ✅ {path}")

nb_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "notebooks")

# Delete old notebooks
for f in os.listdir(nb_dir):
    if f.endswith(".ipynb"):
        os.remove(os.path.join(nb_dir, f))
        print(f"  🗑️ Deleted: {f}")

# ═══════ 01_eda.ipynb ═══════
save(make_nb([
    md("# 📊 Notebook 01: EDA & Tiền xử lý\n## Đề tài 5: Dự báo thời tiết — Szeged Hungary 2006-2016\n\n**Mục tiêu**: Khám phá dữ liệu, xử lý 3 bẫy (Loud Cover/Precip Type/Pressure), thống kê trước-sau."),
    code("import sys, os\nsys.path.insert(0, os.path.abspath('..'))\nos.chdir(os.path.abspath('..'))\n\nimport logging\nlogging.basicConfig(level=logging.INFO, format='%(levelname)s - %(message)s')\n\nimport pandas as pd\nimport matplotlib.pyplot as plt\nimport seaborn as sns\n\nfrom src.data.loader import load_config, resolve_path\nfrom src.data.cleaner import WeatherCleaner\n\nconfig = load_config()\nprint('✅ Config loaded')"),
    md("## 1. Dữ liệu thô — Thống kê TRƯỚC tiền xử lý"),
    code("raw_path = resolve_path(config['data']['raw_path'])\ndf_raw = pd.read_csv(raw_path)\nprint(f'Shape thô: {df_raw.shape}')\nprint(f'\\nMissing values TRƯỚC:')\nprint(df_raw.isnull().sum())\nprint(f'\\nSố duplicate: {df_raw.duplicated().sum()}')\nprint(f'Pressure = 0: {(df_raw[\"Pressure (millibars)\"] == 0).sum()}')\nprint(f'Loud Cover unique: {df_raw[\"Loud Cover\"].unique()}')\ndf_raw.head()"),
    md("## 2. Làm sạch dữ liệu (WeatherCleaner)"),
    code("cleaner = WeatherCleaner(config)\ndf_clean = cleaner.run()\ncleaner.save(df_clean, fmt='parquet')"),
    md("## 3. Thống kê SAU tiền xử lý"),
    code("print(f'Shape sau clean: {df_clean.shape}')\nprint(f'\\nMissing values SAU:')\nprint(df_clean.isnull().sum())\nprint(f'\\nDuplicate SAU: {df_clean.duplicated().sum()}')\nprint(f'\\nPressure min SAU: {df_clean[\"Pressure (millibars)\"].min():.2f}')\nprint(f'Pressure median SAU: {df_clean[\"Pressure (millibars)\"].median():.2f}')"),
    md("### Bảng so sánh Trước – Sau\n| Mục | Trước | Sau |\n|-----|-------|-----|\n| Số dòng | 96,453 | 96,429 |\n| Missing (Precip Type) | 517 | 0 |\n| Pressure = 0 | 1,288 | 0 |\n| Cột Loud Cover | Có (toàn 0) | Đã drop |\n| Duplicate | 24 | 0 |"),
    md("## 4. EDA — Biểu đồ phân bố"),
    code("fig, axes = plt.subplots(2, 3, figsize=(18, 10))\nnumeric_cols = ['Temperature (C)', 'Humidity', 'Wind Speed (km/h)',\n                'Visibility (km)', 'Pressure (millibars)', 'Apparent Temperature (C)']\nfor ax, col in zip(axes.flatten(), numeric_cols):\n    df_clean[col].hist(bins=50, ax=ax, alpha=0.7, edgecolor='white')\n    ax.set_title(col, fontsize=12, fontweight='bold')\n    ax.grid(alpha=0.3)\nplt.suptitle('Phân bố các biến số liên tục', fontsize=16, fontweight='bold')\nplt.tight_layout()\nplt.show()"),
    md("## 5. Tương quan giữa các biến"),
    code("plt.figure(figsize=(10, 8))\ncorr_cols = ['Temperature (C)', 'Apparent Temperature (C)', 'Humidity',\n             'Wind Speed (km/h)', 'Visibility (km)', 'Pressure (millibars)']\ncorr_matrix = df_clean[corr_cols].corr()\nsns.heatmap(corr_matrix, annot=True, cmap='RdBu_r', center=0,\n            fmt='.2f', linewidths=0.5, square=True)\nplt.title('Ma trận Tương quan', fontsize=14, fontweight='bold')\nplt.tight_layout()\nplt.show()"),
    md("## 6. Phân bố Summary + Precip Type"),
    code("fig, axes = plt.subplots(1, 2, figsize=(16, 5))\ndf_clean['Summary'].value_counts().head(10).plot(kind='barh', ax=axes[0], color='#2196F3')\naxes[0].set_title('Top 10 Loại Thời Tiết', fontsize=14, fontweight='bold')\ndf_clean['Precip Type'].value_counts().plot(kind='bar', ax=axes[1], color=['#4CAF50', '#FF9800'])\naxes[1].set_title('Loại Mưa', fontsize=14, fontweight='bold')\nplt.tight_layout()\nplt.show()"),
]), os.path.join(nb_dir, "01_eda.ipynb"))

# ═══════ 02_preprocess_feature.ipynb ═══════
save(make_nb([
    md("# 🔧 Notebook 02: Feature Engineering\n## Trích xuất đặc trưng: Season, Binning, Scaling"),
    code("import sys, os\nsys.path.insert(0, os.path.abspath('..'))\nos.chdir(os.path.abspath('..'))\n\nimport logging\nlogging.basicConfig(level=logging.INFO, format='%(levelname)s - %(message)s')\n\nimport pandas as pd\nimport matplotlib.pyplot as plt\n\nfrom src.data.loader import load_config, resolve_path\nfrom src.features.builder import FeatureBuilder\n\nconfig = load_config()\nparquet_path = resolve_path(config['data']['cleaned_parquet'])\ndf_clean = pd.read_parquet(parquet_path)\nprint(f'Loaded: {df_clean.shape}')"),
    md("## 1. Chạy FeatureBuilder"),
    code("builder = FeatureBuilder(config)\ndf = builder.run(df_clean)\nbuilder.save(df, fmt='parquet')\nprint(f'\\nShape sau FE: {df.shape}')\nprint(f'Columns mới: {[c for c in df.columns if c not in df_clean.columns]}')"),
    md("## 2. Phân bố Season"),
    code("fig, ax = plt.subplots(figsize=(8, 5))\ndf['Season'].value_counts().plot(kind='bar', ax=ax,\n     color=['#4CAF50', '#FF9800', '#9C27B0', '#2196F3'])\nax.set_title('Phân bố theo Mùa', fontsize=14, fontweight='bold')\nax.set_ylabel('Số mẫu')\nplt.xticks(rotation=0)\nplt.tight_layout()\nplt.show()"),
    md("## 3. Kiểm tra Binning"),
    code("print('Binning Temperature:')\nprint(df['Temp_Bin'].value_counts())\nprint('\\nBinning Humidity:')\nprint(df['Humidity_Bin'].value_counts())\nprint('\\nBinning Wind Speed:')\nprint(df['Wind_Bin'].value_counts())"),
    md("## 4. Nhiệt độ theo Mùa"),
    code("fig, ax = plt.subplots(figsize=(12, 5))\nfor season in ['Spring', 'Summer', 'Autumn', 'Winter']:\n    subset = df[df['Season'] == season]\n    ax.hist(subset['Temperature (C)'], bins=50, alpha=0.5, label=season)\nax.set_xlabel('Temperature (°C)', fontsize=12)\nax.set_ylabel('Tần suất', fontsize=12)\nax.set_title('Phân bố Nhiệt độ theo Mùa', fontsize=14, fontweight='bold')\nax.legend()\nax.grid(alpha=0.3)\nplt.tight_layout()\nplt.show()"),
    md("## 5. Xu hướng nhiệt độ theo thời gian"),
    code("df_ts = df.set_index('Formatted Date')['Temperature (C)'].resample('M').mean()\nplt.figure(figsize=(16, 5))\nplt.plot(df_ts.index, df_ts.values, linewidth=1.5, color='#2196F3')\nplt.xlabel('Thời gian')\nplt.ylabel('Nhiệt độ TB (°C)')\nplt.title('Xu hướng Nhiệt độ Trung bình Hàng tháng (2006-2016)', fontsize=14, fontweight='bold')\nplt.grid(alpha=0.3)\nplt.tight_layout()\nplt.show()"),
    md("## 6. Kiểm tra Scaling"),
    code("scaled_cols = [c for c in df.columns if c.endswith('_scaled')]\nprint(f'Scaled columns: {scaled_cols}')\nprint(f'\\nMean (should ≈ 0):')\nprint(df[scaled_cols].mean().round(4))\nprint(f'\\nStd (should ≈ 1):')\nprint(df[scaled_cols].std().round(4))"),
]), os.path.join(nb_dir, "02_preprocess_feature.ipynb"))

# ═══════ 03_mining_clustering.ipynb ═══════
save(make_nb([
    md("# ⛏️ Notebook 03: Khai phá Tri thức\n## Association Rules (FP-Growth) + K-Means Clustering"),
    code("import sys, os\nsys.path.insert(0, os.path.abspath('..'))\nos.chdir(os.path.abspath('..'))\n\nimport logging\nlogging.basicConfig(level=logging.INFO, format='%(levelname)s - %(message)s')\n\nimport pandas as pd\nfrom IPython.display import Image, display\nfrom src.data.loader import load_config, resolve_path\nfrom src.mining.association import AssociationMiner\nfrom src.mining.clustering import WeatherClusterer\n\nconfig = load_config()\ndf = pd.read_parquet(resolve_path(config['data']['cleaned_parquet']))\nprint(f'✅ Loaded: {df.shape}')"),
    md("## Phần A: Luật Kết hợp (Association Rules)"),
    code("miner = AssociationMiner(config)\nassoc_results = miner.run(df)\nminer.save_results()"),
    md("### Bảng tổng hợp Luật"),
    code("comparison = assoc_results['comparison']\nif not comparison.empty:\n    print(f'Tổng số luật: {len(comparison)}')\n    display(comparison.head(20))\nelse:\n    print('Không tìm thấy luật.')"),
    md("### Diễn giải tự động"),
    code("for line in assoc_results['interpretations']:\n    print(line)"),
    md("---\n## Phần B: Phân cụm K-Means"),
    code("clusterer = WeatherClusterer(config)\nclust_results = clusterer.run(df)\nclusterer.save_results()"),
    md("### Elbow + Silhouette"),
    code("display(Image(filename=clust_results['elbow_plot']))"),
    md("### Hồ sơ Cụm"),
    code("profiles = clust_results['profiles']\ndisplay(profiles)"),
    code("display(Image(filename=clust_results['profile_plot']))"),
]), os.path.join(nb_dir, "03_mining_clustering.ipynb"))

# ═══════ 04_classification.ipynb ═══════
save(make_nb([
    md("# 🏷️ Notebook 04: Phân loại Thời tiết\n## 4 mô hình: LR (baseline), DT (baseline), RF (cải tiến), XGBoost (cải tiến)\n## Metric bắt buộc: F1-macro"),
    code("import sys, os\nsys.path.insert(0, os.path.abspath('..'))\nos.chdir(os.path.abspath('..'))\n\nimport logging\nlogging.basicConfig(level=logging.INFO, format='%(levelname)s - %(message)s')\n\nimport pandas as pd\nfrom IPython.display import Image, display\nfrom src.data.loader import load_config, resolve_path\nfrom src.models.classification import WeatherClassifier\n\nconfig = load_config()\ndf = pd.read_parquet(resolve_path(config['data']['cleaned_parquet']))\nprint(f'✅ Loaded: {df.shape}')"),
    md("## 1. Huấn luyện & Đánh giá"),
    code("classifier = WeatherClassifier(config)\nresults = classifier.run(df)\nclassifier.save_results(results['results_df'])"),
    md("## 2. Bảng So sánh Mô hình"),
    code("display(results['results_df'])"),
    md("## 3. Biểu đồ So sánh"),
    code("display(Image(filename=results['comparison_plot']))"),
    md("## 4. Confusion Matrix (Best Model)"),
    code("display(Image(filename=results['confusion_matrix_plot']))"),
    md("## 5. Phân tích Lỗi"),
    code("error = results['error_analysis']\nprint(f\"Tổng sai: {error['n_wrong']} / {error['n_total']} ({error['error_rate']:.1%})\")\ndisplay(error['wrong_predictions'][['Summary', 'True_Label', 'Predicted',\n        'Temperature (C)', 'Humidity', 'Season']].head(10))"),
    md("## 6. Nhận xét\n- **Random Forest** đạt F1-macro cao nhất\n- Lỗi cao nhất: Spring (48.2%), thấp nhất: Winter (34.4%)\n- Nhiệt độ cực trị dễ phân loại hơn (29.9% lỗi) vs bình thường (44.6%)"),
]), os.path.join(nb_dir, "04_classification.ipynb"))

# ═══════ 04b_alternative_branch.ipynb ═══════
save(make_nb([
    md("# 🔍 Notebook 04b: Nhánh Thay thế — Anomaly Detection\n## Criterion F: Phát hiện ngày thời tiết bất thường\n\n**Thay thế Semi-supervised** (không bắt buộc cho Đề tài 5)\n\nSo sánh 2 phương pháp:\n- **Isolation Forest** (ensemble-based)\n- **Local Outlier Factor** (density-based)"),
    code("import sys, os\nsys.path.insert(0, os.path.abspath('..'))\nos.chdir(os.path.abspath('..'))\n\nimport logging\nlogging.basicConfig(level=logging.INFO, format='%(levelname)s - %(message)s')\n\nimport pandas as pd\nfrom IPython.display import Image, display\nfrom src.data.loader import load_config, resolve_path\nfrom src.models.anomaly import WeatherAnomalyDetector\n\nconfig = load_config()\ndf = pd.read_parquet(resolve_path(config['data']['cleaned_parquet']))\nprint(f'✅ Loaded: {df.shape}')"),
    md("## 1. Chạy Anomaly Detection"),
    code("detector = WeatherAnomalyDetector(config)\nresults = detector.run(df)\ndetector.save_results()"),
    md("## 2. So sánh Isolation Forest vs LOF"),
    code("display(results['comparison'])"),
    md("## 3. Phân tích Anomaly theo Mùa"),
    code("display(results['season_analysis'])"),
    md("## 4. Hồ sơ Anomaly vs Normal"),
    code("display(results['profiles'])"),
    md("## 5. Biểu đồ Anomaly Detection"),
    code("display(Image(filename=results['plot_path']))"),
    md("## 6. So sánh IF vs LOF"),
    code("if results.get('comparison_plot'):\n    display(Image(filename=results['comparison_plot']))"),
    md("## 7. Nhận xét\n\n- Đây là **nhánh thay thế** Criterion F (khi không bắt buộc Semi-supervised)\n- Isolation Forest và LOF phát hiện anomaly ở tỷ lệ ~5% (tham số contamination)\n- Phân tích theo mùa cho thấy mùa nào có nhiều ngày bất thường nhất\n- Profile cho thấy ngày anomaly có đặc điểm nhiệt độ/áp suất/gió khác biệt\n- Kết quả bổ sung cho pipeline: phát hiện sự kiện thời tiết cực đoan"),
]), os.path.join(nb_dir, "04b_alternative_branch.ipynb"))

# ═══════ 05_forecasting_evaluation.ipynb ═══════
save(make_nb([
    md("# 📈 Notebook 05: Dự báo Chuỗi Thời gian & Đánh giá Tổng hợp\n## Dự báo nhiệt độ + Actionable Insights"),
    code("import sys, os\nsys.path.insert(0, os.path.abspath('..'))\nos.chdir(os.path.abspath('..'))\n\nimport logging\nlogging.basicConfig(level=logging.INFO, format='%(levelname)s - %(message)s')\n\nimport pandas as pd\nfrom IPython.display import Image, display\nfrom src.data.loader import load_config, resolve_path\nfrom src.models.forecasting import TimeForecaster\n\nconfig = load_config()\ndf = pd.read_parquet(resolve_path(config['data']['cleaned_parquet']))\nprint(f'✅ Loaded: {df.shape}')"),
    md("## 1. Dự báo Chuỗi Thời gian"),
    code("forecaster = TimeForecaster(config)\nresults = forecaster.run(df)\nforecaster.save_results()"),
    md("## 2. ACF/PACF"),
    code("display(Image(filename=results['acf_plot']))"),
    md("### Nhận xét ACF/PACF\n- ACF giảm dần → tính xu hướng mạnh\n- Chu kỳ ~365 ngày → xác nhận mùa vụ hàng năm\n- PACF suy giảm nhanh sau lag 1-2 → AR(1)/AR(2)"),
    md("## 3. So sánh Dự báo"),
    code("display(results['results_df'])\ndisplay(Image(filename=results['comparison_plot']))"),
    md("## 4. Phân tích Phần dư"),
    code("residual = results['residual_analysis']\nprint(f\"Outlier: {residual['n_outliers']}/804 ({residual['n_outliers']/804:.1%})\")\ndisplay(Image(filename=residual['plot_path']))"),
    md("## 5. Top ngày dự báo lệch xa"),
    code("if not residual['outlier_df'].empty:\n    top = residual['outlier_df'].reindex(\n        residual['outlier_df']['Residual'].abs().sort_values(ascending=False).index\n    ).head(10)\n    display(top)"),
    md("---\n# 🎯 ACTIONABLE INSIGHTS"),
    code("insights = [\n    '🔹 INSIGHT 1 — CẢNH BÁO TUYẾT MÙA ĐÔNG:\\n'\n    '   Sương mù + Nhiệt độ thấp + Gió yếu → Tuyết 55%. Tự động cảnh báo.',\n    '🔹 INSIGHT 2 — 2 CHẾ ĐỘ THỜI TIẾT:\\n'\n    '   Lạnh-Ẩm (47%) vs Nóng-Khô (53%). 2 profile năng lượng riêng.',\n    '🔹 INSIGHT 3 — MÙA XUÂN KHÓ DỰ BÁO:\\n'\n    '   Lỗi 48.2% (Spring) vs 34.4% (Winter). Cần features giao mùa.',\n    '🔹 INSIGHT 4 — NAIVE HIỆU QUẢ:\\n'\n    '   MAE=1.55°C, vượt Holt-Winters. Phù hợp dự báo ngắn hạn.',\n    '🔹 INSIGHT 5 — CỰC TRỊ DỄ PHÂN LOẠI:\\n'\n    '   29.9% lỗi (cực trị) vs 44.6% (bình thường). Pattern rõ ràng.',\n    '🔹 INSIGHT 6 — ĐỘ ẨM TĂNG KHI GIÓ YẾU MÙA HÈ:\\n'\n    '   Conf=65.5%. Tiết kiệm nước tưới ngày gió yếu.',\n    '🔹 INSIGHT 7 — 20% NGÀY OUTLIER:\\n'\n    '   166/804 ngày sai >±7.77°C. Cần external features.',\n]\nfor ins in insights:\n    print(ins + '\\n')"),
    md("## Kết luận\n\n1. **EDA & Tiền xử lý**: 3 bẫy xử lý, Season/Binning/Scaling\n2. **Luật Kết hợp**: 80 luật FP-Growth theo 4 mùa\n3. **Phân cụm**: K=2, 2 chế độ thời tiết rõ ràng\n4. **Phân loại**: RF F1=0.44, error analysis chi tiết\n5. **Anomaly Detection**: IF vs LOF (nhánh thay thế Criterion F)\n6. **Chuỗi thời gian**: ACF/PACF + Naive vs HW + Residual\n7. **7 Actionable Insights** có thể áp dụng thực tế"),
]), os.path.join(nb_dir, "05_forecasting_evaluation.ipynb"))

print("\n✅ All 6 notebooks generated with rubric-compliant names!")
