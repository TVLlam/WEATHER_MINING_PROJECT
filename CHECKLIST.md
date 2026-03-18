# ✅ DANH SÁCH TÍNH NĂNG — WEATHER MINING PROJECT
## Đề tài 5: Dự báo thời tiết Szeged Hungary 2006-2016

> Checklist theo Rubric chấm điểm BTL (tổng 11.0đ + bonus)

---

## A. Bài toán + Mô tả dữ liệu + Data Dictionary (1.0đ)

- [x] Mục tiêu dự án rõ ràng — `README.md` phần 1
- [x] Nguồn dữ liệu + link Kaggle — `README.md` phần 2
- [x] Data Dictionary đầy đủ 12 cột — `README.md` phần 3
- [x] Phân tích rủi ro: mất cân bằng lớp (27 lớp Summary)
- [x] Phân tích rủi ro: missing values (Precip Type 517 NaN)
- [x] Phân tích rủi ro: outlier (Pressure = 0, 1288 giá trị)
- [x] Phân tích rủi ro: data leakage (split theo thời gian)
- [x] Cột vô nghĩa: Loud Cover = 0 toàn bộ

---

## B. EDA & Tiền xử lý (1.5đ)

- [x] Module `src/data/cleaner.py` — class `WeatherCleaner`
- [x] Xử lý BẪY 1: Drop cột Loud Cover — `drop_loud_cover()`
- [x] Xử lý BẪY 2: Fill Missing Precip Type — `handle_missing_precip_type()`
- [x] Xử lý BẪY 3: Pressure = 0 → NaN → Median — `handle_pressure_outliers()`
- [x] Loại bỏ dòng trùng lặp — `drop_duplicates()`
- [x] Pipeline hóa: `cleaner.run()` chạy tuần tự tất cả bước
- [x] Logging shape trước/sau để kiểm soát
- [x] Lưu kết quả ra Parquet/CSV — `cleaner.save()`
- [x] Notebook EDA: `notebooks/01_eda.ipynb`
- [x] Notebook Tiền xử lý: `notebooks/02_preprocess_feature.ipynb`

---

## C. Data Mining Core — Khai phá Tri thức (2.0đ)

### C1. Luật kết hợp (Association Rules)
- [x] Module `src/mining/association.py` — class `AssociationMiner`
- [x] Rời rạc hóa: Temp_Bin, Humidity_Bin, Wind_Bin — `builder.py`
- [x] Transaction Encoding (one-hot) — `_prepare_transactions()`
- [x] Thuật toán FP-Growth / Apriori — `_mine_rules()`
- [x] **Tách theo mùa** — `mine_rules_by_season()`
- [x] So sánh luật giữa 4 mùa — `compare_seasons()`
- [x] Diễn giải tự động bằng ngôn ngữ tự nhiên — `interpret_rules()`
- [x] Tham số cấu hình: min_support=0.05, min_confidence=0.5, min_lift=1.0
- [x] Lưu kết quả CSV — `save_results()`

### C2. Phân cụm K-Means (Clustering)
- [x] Module `src/mining/clustering.py` — class `WeatherClusterer`
- [x] Chuẩn hóa StandardScaler cho 7 biến số — cột `_scaled`
- [x] Tìm K tối ưu: Elbow Method + Silhouette Score — `find_optimal_k()`
- [x] Sampling cho Silhouette (max 10K mẫu) — tránh quá tải RAM
- [x] Huấn luyện K-Means — `fit()`
- [x] **Hồ sơ cụm** (Mean/Median/Mode) — `cluster_profiling()`
- [x] Tự động gợi ý tên cụm (Lạnh/Nóng/Ẩm/Khô...) — logic rule-based
- [x] Biểu đồ Elbow + Silhouette — `plot_elbow()`
- [x] Biểu đồ Profile cụm — `plot_cluster_profile()`
- [x] Lưu hồ sơ cụm CSV — `save_results()`
- [x] Notebook: `notebooks/03_mining_clustering.ipynb`

---

## D. Mô hình hóa + Baseline so sánh ≥ 2 (2.0đ)

### D1. Phân lớp loại thời tiết (Classification)
- [x] Module `src/models/classification.py` — class `WeatherClassifier`
- [x] **Label Grouping**: 27 nhãn Summary → 5 nhóm — `LABEL_MAP`
- [x] Baseline 1: Logistic Regression (class_weight='balanced')
- [x] Baseline 2: Decision Tree (class_weight='balanced')
- [x] Cải tiến 1: Random Forest (200 cây, class_weight='balanced')
- [x] Cải tiến 2: XGBoost (150 cây, learning_rate=0.1)
- [x] Metric chính: F1-macro
- [x] Cross-Validation 5-fold — `cross_val_score()`
- [x] Stratified Train/Test Split (80/20)
- [x] LabelEncoder cho target
- [x] Tự động chọn model tốt nhất — `best_model_name`
- [x] Confusion Matrix — `plot_confusion_matrix()`
- [x] So sánh 4 mô hình — `plot_model_comparison()`
- [x] Lưu model joblib — `save_model()`
- [x] Notebook: `notebooks/04_classification.ipynb`

### D2. Dự báo chuỗi thời gian (Time Series Forecasting)
- [x] Module `src/models/forecasting.py` — class `TimeForecaster`
- [x] Chuyển Hourly → Daily (resample mean) — `prepare_time_series()`
- [x] **Split theo THỜI GIAN** (không shuffle!) — 80% train / 20% test
- [x] ACF/PACF — `plot_acf_pacf()`
- [x] Baseline: Naive Forecast (y_pred = y(t-1)) — `naive_forecast()`
- [x] Holt-Winters Exponential Smoothing (seasonal_periods=365) — `arima_forecast()`
- [x] Fallback: SimpleExpSmoothing khi HW thất bại
- [x] Metric: MAE + RMSE — cả Naive và HW đều tính
- [x] So sánh Naive vs HW — `plot_forecast_comparison()`
- [x] Notebook: `notebooks/05_forecasting_evaluation.ipynb`

---

## E. Thiết kế Thực nghiệm + Metric đúng (1.0đ)

- [x] Seed cố định: `random_seed: 42` — `configs/params.yaml`
- [x] Stratified split (Classification) — tránh bias
- [x] Temporal split (Forecasting) — tránh data leakage
- [x] Cross-Validation 5-fold (Classification)
- [x] Metric phù hợp: F1-macro (classification), MAE/RMSE (forecasting), Silhouette (clustering)
- [x] Tất cả tham số tập trung trong `configs/params.yaml`

---

## F. Nhánh thay thế — Anomaly Detection (1.0đ)

> Đề tài 5 KHÔNG yêu cầu bán giám sát. Thay thế bằng Anomaly Detection.

- [x] Module `src/models/anomaly.py` — class `WeatherAnomalyDetector`
- [x] Isolation Forest (200 cây, contamination=5%) — `run_isolation_forest()`
- [x] Local Outlier Factor (n_neighbors=20) — `run_lof()`
- [x] So sánh IF vs LOF: overlap, consensus — `compare_methods()`
- [x] Phân tích anomaly theo mùa — `analyze_anomalies_by_season()`
- [x] Top 20 ngày bất thường nhất — `get_top_anomalies()`
- [x] Biểu đồ scatter IF vs LOF — `plot_anomaly_scatter()`
- [x] Biểu đồ anomaly theo mùa — `plot_anomaly_by_season()`
- [x] Biểu đồ timeline anomaly — `plot_anomaly_timeline()`
- [x] Insight: so sánh đặc trưng anomaly vs normal — `_log_insights()`
- [x] Lưu kết quả CSV (3 files) — `save_results()`
- [x] Tham số cấu hình — `configs/params.yaml` (anomaly_detection section)
- [x] Notebook: `notebooks/04b_alternative_branch.ipynb`

---

## G. Đánh giá, Phân tích lỗi & Insight hành động (1.5đ)

- [x] Confusion Matrix cho model tốt nhất — `plot_confusion_matrix()`
- [x] **Phân tích lỗi theo Season** — `error_analysis()`
- [x] **Phân tích lỗi giao mùa** (tháng 3,6,9,12) vs bình thường
- [x] **Phân tích lỗi cực trị nhiệt độ** (quantile 5%–95%)
- [x] Residual Analysis: mean, std, max, outlier > 2σ — `residual_analysis()`
- [x] 4 biểu đồ phần dư (line, histogram, actual vs predicted, scatter)
- [x] Insight hành động trong Streamlit: giao thông (Cụm 0), nông nghiệp (Cụm 1)
- [x] Insight dự báo: cảnh báo đợt rét đậm, khuyến nghị SARIMAX
- [x] Insight luật kết hợp: diễn giải khuyến nghị theo mùa

---

## H. Repo GitHub chuẩn + Reproducible (1.0đ)

- [x] `README.md` — chi tiết, đầy đủ hướng dẫn
- [x] `requirements.txt` — dependencies
- [x] `.gitignore` — loại trừ data + outputs lớn
- [x] `configs/params.yaml` — tập trung tham số
- [x] Cấu trúc thư mục đúng mẫu: src/, notebooks/, scripts/, outputs/, configs/
- [x] `scripts/run_pipeline.py` — chạy toàn bộ 4 phase
- [x] `scripts/generate_notebooks.py` — sinh notebooks
- [x] Notebooks theo thứ tự 01→05 đúng pipeline
- [x] Module `src/data/loader.py` — config loader + path resolver
- [x] Module `src/evaluation/metrics.py` — hàm tính metric dùng chung
- [x] Module `src/evaluation/report.py` — tổng hợp kết quả pipeline
- [x] Module `src/visualization/plots.py` — hàm vẽ dùng chung

---

## I. Điểm thưởng — GUI/Demo App (Bonus)

- [x] `app.py` — Streamlit Dashboard (669 dòng)
- [x] Custom CSS: Google Fonts Inter, gradient cards, dark sidebar
- [x] **Trang 1**: 🌍 Bảng điều khiển Thời tiết
  - [x] 4 KPI Metric Cards (Nhiệt độ, Độ ẩm, Gió, Tổng bản ghi)
  - [x] Plotly interactive: Nhiệt độ + Độ ẩm theo thời gian (range slider)
  - [x] Biểu đồ phân bố theo Mùa
- [x] **Trang 2**: 🧠 Khai phá Tri thức
  - [x] Tab 1: Luật Kết Hợp — chọn mùa, phiên dịch Tiếng Việt, insight chuyên gia
  - [x] Tab 2: Phân Cụm — scatter Temp vs Humidity, hồ sơ cụm, actionable insights
- [x] **Trang 3**: 🔮 Cỗ máy AI Dự báo
  - [x] Tab 1: Dự đoán loại thời tiết (7 sliders + form submit + kết quả lớn)
  - [x] Tab 2: Dự báo chuỗi thời gian (Holt-Winters + MAE/RMSE + Plotly + fallback Naive)
  - [x] Residual Analysis + Actionable Insights (expander)
- [x] @cache_data / @cache_resource cho performance

---

## 📊 TỔNG KẾT

| Tiêu chí | Điểm tối đa | Trạng thái |
|----------|:-----------:|:----------:|
| A. Bài toán + Data Dictionary | 1.0 | ✅ Đạt |
| B. EDA & Tiền xử lý | 1.5 | ✅ Đạt |
| C. Mining Core (Association + Clustering) | 2.0 | ✅ Đạt |
| D. Mô hình + Baseline ≥ 2 | 2.0 | ✅ Đạt |
| E. Thực nghiệm + Metric | 1.0 | ✅ Đạt |
| F. Nhánh thay thế (Anomaly Detection) | 1.0 | ✅ Đạt |
| G. Phân tích lỗi + Insight | 1.5 | ✅ Đạt |
| H. Repo chuẩn + Reproducible | 1.0 | ✅ Đạt |
| **TỔNG** | **11.0** | **✅ ĐẦY ĐỦ** |
| Bonus: GUI App (Streamlit) | +bonus | ✅ Đạt |

---

## 📁 Cấu trúc File đã hoàn thành

```
WEATHER_MINING_PROJECT/
├── README.md ✅
├── requirements.txt ✅
├── .gitignore ✅
├── configs/
│   └── params.yaml ✅ (8 sections config)
├── notebooks/
│   ├── 01_eda.ipynb ✅
│   ├── 02_preprocess_feature.ipynb ✅
│   ├── 03_mining_clustering.ipynb ✅
│   ├── 04_classification.ipynb ✅
│   ├── 04b_alternative_branch.ipynb ✅
│   └── 05_forecasting_evaluation.ipynb ✅
├── src/
│   ├── data/
│   │   ├── loader.py ✅ (Config loader)
│   │   └── cleaner.py ✅ (WeatherCleaner)
│   ├── features/
│   │   └── builder.py ✅ (FeatureBuilder)
│   ├── mining/
│   │   ├── association.py ✅ (AssociationMiner)
│   │   └── clustering.py ✅ (WeatherClusterer)
│   ├── models/
│   │   ├── classification.py ✅ (WeatherClassifier)
│   │   ├── forecasting.py ✅ (TimeForecaster)
│   │   └── anomaly.py ✅ (WeatherAnomalyDetector) ← MỚI
│   ├── evaluation/
│   │   ├── metrics.py ✅ (Hàm metric dùng chung)
│   │   └── report.py ✅ (PipelineReporter) ← MỚI
│   └── visualization/
│       └── plots.py ✅ (Hàm vẽ dùng chung)
├── scripts/
│   ├── run_pipeline.py ✅
│   └── generate_notebooks.py ✅
├── outputs/
│   ├── figures/ (tự tạo khi chạy pipeline)
│   ├── tables/ ✅
│   └── models/ (tự tạo khi chạy pipeline)
└── app.py ✅ (Streamlit Dashboard)
```
