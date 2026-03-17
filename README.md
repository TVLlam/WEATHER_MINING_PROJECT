# 🌦️ WEATHER MINING PROJECT
## Đề tài 5: Dự báo thời tiết — Szeged Hungary 2006-2016
### Học phần: Khai phá Dữ liệu — HK2 2025-2026

---

## 1. Mục tiêu Dự án

Xây dựng pipeline khai phá dữ liệu thời tiết toàn diện từ bộ dữ liệu **Weather in Szeged, Hungary (2006-2016)**:
- **Khai phá luật kết hợp** phát hiện quy luật đồng xuất hiện các điều kiện thời tiết theo mùa
- **Phân cụm K-Means** nhóm các ngày có kiểu thời tiết tương tự
- **Phân loại** loại thời tiết (Summary) bằng nhiều mô hình ML
- **Dự báo chuỗi thời gian** nhiệt độ hàng ngày
- **Phát hiện bất thường** ngày thời tiết cực đoan bằng Isolation Forest & LOF

---

## 2. Nguồn Dữ liệu

- **Dataset**: [Weather in Szeged 2006-2016](https://www.kaggle.com/datasets/budincsevity/szeged-weather)
- **Kích thước**: 96,453 hàng × 12 cột (dữ liệu theo giờ, ~10 năm)
- **Định dạng**: CSV
- **Lưu trữ**: `data/raw/weather_raw.csv` (không commit lên GitHub — xem hướng dẫn tải bên dưới)

---

## 3. Data Dictionary

| # | Cột | Kiểu | Mô tả | Vai trò |
|---|-----|------|-------|---------|
| 1 | `Formatted Date` | datetime | Ngày giờ quan trắc (UTC+1) | Index thời gian |
| 2 | `Summary` | categorical | Tóm tắt thời tiết (27 lớp: Partly Cloudy, Mostly Cloudy, Overcast, Clear, Foggy, ...) | **Target phân loại** |
| 3 | `Precip Type` | categorical | Loại mưa: rain / snow / NaN | Feature |
| 4 | `Temperature (C)` | float | Nhiệt độ thực tế (°C) | **Target dự báo** |
| 5 | `Apparent Temperature (C)` | float | Nhiệt độ cảm nhận (°C) | Feature |
| 6 | `Humidity` | float | Độ ẩm tương đối (0.0 – 1.0) | Feature |
| 7 | `Wind Speed (km/h)` | float | Tốc độ gió (km/h) | Feature |
| 8 | `Wind Bearing (degrees)` | float | Hướng gió (0° – 360°) | Feature |
| 9 | `Visibility (km)` | float | Tầm nhìn xa (km) | Feature |
| 10 | `Loud Cover` | float | **Luôn = 0** → Cột vô nghĩa → Drop | ⚠️ Bẫy 1 |
| 11 | `Pressure (millibars)` | float | Áp suất khí quyển (mbar) — **có 1,288 giá trị = 0 (lỗi máy đo)** | ⚠️ Bẫy 3 |
| 12 | `Daily Summary` | text | Mô tả dài thời tiết trong ngày | Không dùng |

---

## 4. Phân tích Rủi ro Dữ liệu

| Rủi ro | Mô tả | Giải pháp |
|--------|-------|-----------|
| **Mất cân bằng lớp** | `Summary` có 27 lớp; 4 lớp chiếm 90% (Partly Cloudy: 33%, Mostly Cloudy: 29%). 6 lớp có < 10 mẫu | Loại bỏ lớp hiếm < 10 mẫu; dùng F1-macro thay accuracy |
| **Missing Values** | `Precip Type`: 517 NaN (~0.5%) | Fill bằng "rain" (mode) — ⚠️ Bẫy 2 |
| **Outlier** | `Pressure = 0`: 1,288 giá trị — lỗi sensor | Replace 0 → NaN → Median (1016.55 mbar) — ⚠️ Bẫy 3 |
| **Cột vô nghĩa** | `Loud Cover` = 0 cho tất cả 96K dòng | Drop cột — ⚠️ Bẫy 1 |
| **Data Leakage** | Dữ liệu time series → Không được shuffle khi split | Forecasting: split theo thời gian (80% train trước, 20% test sau) |
| **High cardinality** | 27 lớp Summary khó dự đoán | Xem xét gộp lớp hoặc chấp nhận F1-macro thấp |

---

## 5. Cấu trúc Dự án

```
WEATHER_MINING_PROJECT/
├── README.md                          # File này
├── requirements.txt                   # Dependencies
├── .gitignore
├── configs/
│   └── params.yaml                    # Toàn bộ tham số
├── data/
│   ├── raw/                           # Dữ liệu gốc (không commit)
│   └── processed/                     # Parquet sau tiền xử lý
├── notebooks/
│   ├── 01_eda.ipynb                   # EDA + Tiền xử lý
│   ├── 02_preprocess_feature.ipynb    # Feature Engineering chi tiết
│   ├── 03_mining_clustering.ipynb     # Association Rules + K-Means
│   ├── 04_classification.ipynb        # Phân loại thời tiết
│   ├── 04b_alternative_branch.ipynb   # Anomaly Detection (nhánh thay thế)
│   └── 05_forecasting_evaluation.ipynb # Chuỗi thời gian + Insights
├── src/
│   ├── data/
│   │   ├── loader.py                  # Config loader
│   │   └── cleaner.py                 # WeatherCleaner (3 bẫy)
│   ├── features/
│   │   └── builder.py                 # FeatureBuilder
│   ├── mining/
│   │   ├── association.py             # AssociationMiner (FP-Growth)
│   │   └── clustering.py             # WeatherClusterer (K-Means)
│   ├── models/
│   │   ├── classification.py          # WeatherClassifier (4 models)
│   │   ├── forecasting.py            # TimeForecaster (Naive + HW)
│   │   └── anomaly.py                # WeatherAnomalyDetector (IF + LOF)
│   ├── evaluation/
│   │   └── metrics.py                 # Hàm tính metric dùng chung
│   └── visualization/
│       └── plots.py                   # Hàm vẽ dùng chung
├── scripts/
│   ├── run_pipeline.py                # Chạy toàn bộ pipeline
│   └── generate_notebooks.py
├── outputs/
│   ├── figures/                       # Biểu đồ PNG
│   ├── tables/                        # Bảng CSV
│   ├── models/                        # Model artifacts
│   └── reports/
└── app.py                             # Streamlit Demo App
```

---

## 6. Hướng dẫn Cài đặt & Chạy

### 6.1 Cài đặt
```bash
git clone <repo-url>
cd WEATHER_MINING_PROJECT
pip install -r requirements.txt
```

### 6.2 Tải dữ liệu
Tải dataset từ [Kaggle](https://www.kaggle.com/datasets/budincsevity/szeged-weather) và lưu vào `data/raw/weather_raw.csv`.

### 6.3 Chạy toàn bộ pipeline
```bash
python scripts/run_pipeline.py
```

### 6.4 Chạy Streamlit Demo
```bash
streamlit run app.py
```

### 6.5 Mở Jupyter Notebooks
```bash
jupyter notebook notebooks/
```
Chạy notebook theo thứ tự: `01` → `02` → `03` → `04` → `04b` → `05`

---

## 7. Kết quả Chính

| Module | Kết quả |
|--------|---------|
| Association Mining | 80 luật kết hợp, 4 mùa (Lift tới 4.25) |
| Clustering | K=2, Silhouette=0.245 (Lạnh-Ẩm vs Nóng-Khô) |
| Classification | Random Forest F1-macro=0.4423 (best of 4 models) |
| Forecasting | Naive MAE=1.55°C, Holt-Winters MAE=4.88°C |
| Anomaly Detection | Isolation Forest + LOF phát hiện ngày bất thường |

---

## 8. Thành viên nhóm

| STT | Họ tên | MSSV | Vai trò |
|-----|--------|------|---------|
| 1 | [Tên SV 1] | [MSSV] | Trưởng nhóm |
| 2 | [Tên SV 2] | [MSSV] | Thành viên |
