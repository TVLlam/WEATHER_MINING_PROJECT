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

## 🔍 MÔ TẢ CHI TIẾT CHỨC NĂNG TỪNG TIÊU CHÍ

### A. Bài toán + Data Dictionary (1.0đ)
**Mục đích:** Xác định rõ bài toán cần giải quyết, mô tả nguồn dữ liệu và ý nghĩa từng cột.

| Chức năng | Mô tả hoạt động | Vị trí |
|-----------|-----------------|--------|
| Mục tiêu dự án | Xây dựng pipeline khai phá dữ liệu thời tiết Szeged Hungary: khai phá luật, phân cụm, phân loại, dự báo, phát hiện bất thường | `README.md` phần 1 |
| Data Dictionary | Bảng mô tả chi tiết 12 cột dữ liệu: tên cột, kiểu dữ liệu, ý nghĩa, vai trò (feature/target/bẫy) | `README.md` phần 3 |
| Phân tích rủi ro | Liệt kê 6 rủi ro dữ liệu (mất cân bằng lớp, missing, outlier, cột vô nghĩa, leakage, high cardinality) kèm giải pháp | `README.md` phần 4 |

---

### B. EDA & Tiền xử lý (1.5đ)
**Mục đích:** Làm sạch dữ liệu thô, xử lý 3 "bẫy" chính, chuẩn bị dữ liệu cho các module tiếp theo.

| Chức năng | Mô tả hoạt động | Module/File |
|-----------|-----------------|-------------|
| Drop Loud Cover | Xóa cột `Loud Cover` vì toàn bộ 96K dòng = 0, không có ý nghĩa phân tích | `cleaner.py` → `drop_loud_cover()` |
| Fill Missing Precip Type | Điền 517 giá trị NaN ở cột `Precip Type` bằng mode ("rain") | `cleaner.py` → `handle_missing_precip_type()` |
| Fix Pressure Outlier | Thay 1,288 giá trị Pressure = 0 (lỗi sensor) → NaN → Median (1016.55 mbar) | `cleaner.py` → `handle_pressure_outliers()` |
| Feature Engineering | Trích xuất Year/Month/Day/Hour/Season, tạo Temp_Bin/Humidity_Bin/Wind_Bin, chuẩn hóa StandardScaler | `builder.py` → `run()` |

**Trên Web App:** Trang 🌍 "Bảng điều khiển" hiển thị dữ liệu đã được làm sạch — 4 KPI cards (Nhiệt độ TB, Độ ẩm TB, Tốc độ gió TB, Tổng bản ghi) + biểu đồ Plotly interactive.

---

### C. Mining Core — Khai phá Tri thức (2.0đ)
**Mục đích:** Phát hiện quy luật ẩn trong dữ liệu bằng luật kết hợp và phân cụm.

| Chức năng | Mô tả hoạt động | Module/File |
|-----------|-----------------|-------------|
| Luật kết hợp theo mùa | Dùng FP-Growth tìm các điều kiện thời tiết đồng xuất hiện, tách riêng 4 mùa để so sánh | `association.py` → `mine_rules_by_season()` |
| Top luật + Lift | Lọc luật theo min_support=0.05, min_confidence=0.5, min_lift=1.0, sắp xếp theo Lift giảm dần | `association.py` → `_mine_rules()` |
| Diễn giải tự nhiên | Chuyển luật dạng "A → B" thành câu diễn giải có nghĩa bằng tiếng Việt | `association.py` → `interpret_rules()` |
| Phân cụm K-Means | Nhóm 96K bản ghi thành K cụm theo 7 đặc trưng số, tìm K tối ưu bằng Elbow + Silhouette | `clustering.py` → `find_optimal_k()` + `fit()` |
| Hồ sơ cụm | Tính Mean/Median/Mode cho mỗi cụm, tự động đặt tên ("Lạnh, Ẩm ướt" / "Nóng, Hanh khô") | `clustering.py` → `cluster_profiling()` |

**Trên Web App:** Trang 🧠 "Khai phá Tri thức":
- **Tab Luật Kết Hợp:** Chọn mùa → hiển thị top 5 luật được phiên dịch Tiếng Việt + insight chuyên gia
- **Tab Phân Cụm:** Biểu đồ scatter Nhiệt độ vs Độ ẩm có màu theo cụm + bảng hồ sơ cụm + actionable insights

---

### D. Mô hình hóa + Baseline ≥ 2 (2.0đ)
**Mục đích:** Xây dựng mô hình dự đoán loại thời tiết và dự báo nhiệt độ, có ≥ 2 baseline để so sánh.

| Chức năng | Mô tả hoạt động | Module/File |
|-----------|-----------------|-------------|
| Label Grouping 27→5 | Gộp 27 nhãn Summary gốc thành 5 nhóm chính: Clear, Cloudy, Foggy, Rain, Windy | `classification.py` → `LABEL_MAP` |
| 4 mô hình phân loại | Logistic Regression, Decision Tree (baseline), Random Forest, XGBoost (cải tiến) | `classification.py` → `_build_models()` |
| So sánh mô hình | Bảng F1-macro + Accuracy + CV 5-fold, tự động chọn model tốt nhất (RF: F1=0.755) | `classification.py` → `train_and_evaluate()` |
| Naive Forecast | Baseline dự báo: nhiệt độ ngày mai = nhiệt độ hôm nay (MAE=1.55°C) | `forecasting.py` → `naive_forecast()` |
| Holt-Winters | Mô hình chuỗi thời gian với seasonal_periods=365, áp dụng trên dữ liệu daily mean | `forecasting.py` → `arima_forecast()` |

**Trên Web App:** Trang 🔮 "Cỗ máy AI Dự báo":
- **Tab Dự đoán Loại TT:** Người dùng kéo 7 sliders (Nhiệt độ, Độ ẩm, Gió...) → nhấn Submit → model Random Forest dự đoán loại thời tiết → hiển thị kết quả lớn + emoji
- **Tab Dự báo Chuỗi TG:** Chọn số ngày dự báo → Holt-Winters forecast → biểu đồ Plotly + MAE/RMSE + thống kê (TB/Max/Min) + actionable insights

---

### E. Thiết kế Thực nghiệm + Metric đúng (1.0đ)
**Mục đích:** Đảm bảo thực nghiệm khoa học, có thể tái lập, metric phù hợp với từng bài toán.

| Chức năng | Mô tả hoạt động | Vị trí |
|-----------|-----------------|--------|
| Seed cố định | `random_seed: 42` cho tất cả mô hình — đảm bảo chạy lại luôn ra cùng kết quả | `configs/params.yaml` |
| Stratified Split | Chia train/test 80/20 giữ đúng tỷ lệ các lớp (rất quan trọng khi lớp mất cân bằng) | `classification.py` |
| Temporal Split | Chia theo thời gian (train=trước, test=sau) — KHÔNG shuffle — tránh data leakage | `forecasting.py` |
| Metric phù hợp | F1-macro (phân loại mất cân bằng), MAE/RMSE (dự báo), Silhouette (phân cụm) | `evaluation/metrics.py` |

---

### F. Nhánh thay thế — Anomaly Detection (1.0đ)
**Mục đích:** Phát hiện các ngày thời tiết bất thường (cực đoan) bằng 2 phương pháp và so sánh.

| Chức năng | Mô tả hoạt động | Module/File |
|-----------|-----------------|-------------|
| Isolation Forest | Xây 200 cây phân tách ngẫu nhiên, mẫu nào bị cô lập nhanh → anomaly (contamination=5%) | `anomaly.py` → `run_isolation_forest()` |
| LOF | So sánh mật độ cục bộ của mỗi điểm với 20 láng giềng, mật độ thấp bất thường → anomaly | `anomaly.py` → `run_lof()` |
| So sánh IF vs LOF | Tính overlap: 1,058 ngày cả hai đồng thuận (1.1%), 3,764 chỉ IF, 3,764 chỉ LOF | `anomaly.py` → `compare_methods()` |
| Phân tích theo mùa | Mùa Đông có anomaly cao nhất (IF=10.0%), mùa Thu thấp nhất (IF=2.0%) | `anomaly.py` → `analyze_anomalies_by_season()` |
| Top ngày bất thường | Lấy 20 ngày có combined score cao nhất + đặc trưng chi tiết (ngày, mùa, nhiệt độ...) | `anomaly.py` → `get_top_anomalies()` |
| 3 biểu đồ | Scatter IF vs LOF, bar chart anomaly theo mùa, timeline anomaly markers trên chuỗi thời gian | `anomaly.py` → `plot_*()` |

**Trên Pipeline:** Chạy ở Phase 4, output ra `outputs/tables/anomaly_summary.csv`, `anomaly_by_season.csv`, `anomaly_top_days.csv` + 3 biểu đồ PNG.

---

### G. Đánh giá, Phân tích lỗi & Insight hành động (1.5đ)
**Mục đích:** Phân tích sâu các trường hợp model đoán sai, rút ra insight có thể hành động.

| Chức năng | Mô tả hoạt động | Module/File |
|-----------|-----------------|-------------|
| Confusion Matrix | Ma trận nhầm lẫn 5×5 cho model tốt nhất — thấy rõ lớp nào hay bị nhầm | `classification.py` → `plot_confusion_matrix()` |
| Lỗi theo Season | So sánh % lỗi giữa 4 mùa — phát hiện Summer sai nhiều nhất (13.7%) | `classification.py` → `error_analysis()` |
| Lỗi giao mùa | So sánh tháng giao mùa (3,6,9,12) vs tháng bình thường — kiểm tra giả thuyết "giao mùa khó đoán" | `classification.py` → `error_analysis()` |
| Lỗi cực trị nhiệt độ | So sánh sai số ở nhiệt độ cực trị (quantile 5%–95%) vs bình thường | `classification.py` → `error_analysis()` |
| Residual Analysis | Phân tích phần dư dự báo: mean=-1.5°C (bias), std=3.57°C, 55 ngày outlier > 2σ | `forecasting.py` → `residual_analysis()` |

**Trên Web App:**
- Trang 🧠: Actionable insights cho mỗi cụm (khuyến nghị giao thông, nông nghiệp, du lịch)
- Trang 🔮: Insights dự báo (cảnh báo đợt rét, khuyến nghị mô hình SARIMAX)
- Insight luật kết hợp: diễn giải khuyến nghị theo mùa bằng tiếng Việt

---

### H. Repo GitHub chuẩn + Reproducible (1.0đ)
**Mục đích:** Người khác có thể clone repo, cài đặt, chạy pipeline và thu được cùng kết quả.

| Chức năng | Mô tả hoạt động | File |
|-----------|-----------------|------|
| README.md | Hướng dẫn đầy đủ: mục tiêu, data dictionary, rủi ro, cấu trúc, cài đặt, chạy, kết quả | `README.md` |
| requirements.txt | Liệt kê 12 dependencies chính (pandas, scikit-learn, xgboost, streamlit, plotly...) | `requirements.txt` |
| params.yaml | Tập trung TẤT CẢ tham số (8 sections) — thay đổi 1 chỗ, ảnh hưởng toàn pipeline | `configs/params.yaml` |
| run_pipeline.py | Chạy `python scripts/run_pipeline.py` → tự động chạy 4 phase → tạo tất cả outputs trong ~330s | `scripts/run_pipeline.py` |
| Cấu trúc module hóa | Code tách thành src/ (logic) + notebooks/ (trình bày) + configs/ (tham số) + outputs/ (kết quả) | Toàn bộ repo |

---

### I. Bonus: GUI App Streamlit (+điểm thưởng)
**Mục đích:** Demo trực quan toàn bộ kết quả dự án qua giao diện web tương tác.

| Trang | Chức năng trên Web | Hoạt động |
|-------|-------------------|-----------|
| 🌍 Bảng điều khiển | Dashboard tổng quan thời tiết | 4 KPI cards → biểu đồ Plotly Nhiệt độ + Độ ẩm (kéo zoom range slider) → phân bố theo Mùa |
| 🧠 Khai phá Tri thức | Luật kết hợp + Phân cụm | Chọn mùa → top 5 luật bằng Tiếng Việt + insight chuyên gia ‖ Scatter plot phân cụm + hồ sơ cụm + khuyến nghị |
| 🔮 Cỗ máy AI Dự báo | Dự đoán + Dự báo tương tác | 7 sliders → nhấn Dự đoán → kết quả loại thời tiết ‖ Chọn số ngày → Holt-Winters forecast → biểu đồ + MAE/RMSE + insights |

**Cách chạy:** `streamlit run app.py` → mở `http://localhost:8501`

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
