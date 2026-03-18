"""
🌦️ Weather Mining Project — Interactive Dashboard
Đề tài 5: Dự báo thời tiết Szeged Hungary 2006-2016

Usage: streamlit run app.py
"""
import sys, os

project_root = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, project_root)
os.chdir(project_root)

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import joblib
from PIL import Image

# ──────────── CONFIG ────────────
st.set_page_config(
    page_title="🌦️ Thời tiết Szeged",
    page_icon="🌦️",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ──────────── CUSTOM CSS ────────────
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700;800&display=swap');
    html, body, [class*="css"] { font-family: 'Inter', sans-serif; }

    /* ── Title Styles ── */
    .main-title {
        font-size: 2.6rem; font-weight: 800; text-align: center;
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        -webkit-background-clip: text; -webkit-text-fill-color: transparent;
        margin-bottom: 0.2rem; padding-top: 0.5rem;
    }
    .sub-title {
        font-size: 1.05rem; text-align: center; color: #888;
        margin-bottom: 2rem; font-weight: 400;
    }
    .section-header {
        font-size: 1.3rem; font-weight: 700; margin: 1.5rem 0 0.8rem 0;
        border-left: 4px solid #667eea; padding-left: 12px;
    }

    /* ── KPI Metric Cards ── */
    div[data-testid="stMetric"] {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        border: none; border-radius: 14px; padding: 18px 22px;
        box-shadow: 0 4px 15px rgba(102, 126, 234, 0.3);
    }
    div[data-testid="stMetric"] label[data-testid="stMetricLabel"] p {
        color: rgba(255,255,255,0.85) !important; font-weight: 600;
    }
    div[data-testid="stMetric"] div[data-testid="stMetricValue"] > div {
        color: #ffffff !important; font-weight: 800;
    }
    div[data-testid="stMetric"] div[data-testid="stMetricDelta"] {
        color: rgba(255,255,255,0.9) !important;
    }
    div[data-testid="stMetric"] div[data-testid="stMetricDelta"] svg {
        fill: rgba(255,255,255,0.9);
    }

    /* ── Big Prediction Box ── */
    .big-prediction {
        font-size: 3rem; font-weight: 800; text-align: center;
        padding: 1.2rem; border-radius: 16px; margin: 1rem 0;
    }

    /* ── Tabs ── */
    .stTabs [data-baseweb="tab-list"] { gap: 8px; }
    .stTabs [data-baseweb="tab"] {
        border-radius: 8px; padding: 8px 20px; font-weight: 600;
    }

    /* ── Sidebar ── */
    section[data-testid="stSidebar"] {
        background: linear-gradient(180deg, #1a1a2e 0%, #16213e 100%);
    }
    section[data-testid="stSidebar"] * {
        color: #e0e0e0 !important;
    }
</style>
""", unsafe_allow_html=True)

# ──────────── LOAD DATA ────────────
@st.cache_data
def load_data():
    try:
        from src.data.loader import load_config, resolve_path
        config = load_config()
        path = resolve_path(config["data"]["cleaned_parquet"])
        df = pd.read_parquet(path)
        for col in df.select_dtypes(include=["datetimetz"]).columns:
            df[col] = df[col].dt.tz_localize(None)
        return df, config
    except Exception as e:
        st.warning(f"Không load được dữ liệu: {e}")
        return None, {}

@st.cache_data
def load_rules():
    try:
        return pd.read_csv("outputs/tables/association_rules_by_season.csv")
    except: return None

@st.cache_data
def load_cluster_profiles():
    try:
        return pd.read_csv("outputs/tables/cluster_profiles.csv")
    except: return None

@st.cache_data
def load_anomaly_results():
    try:
        summary = pd.read_csv("outputs/tables/anomaly_summary.csv")
        by_season = pd.read_csv("outputs/tables/anomaly_by_season.csv")
        top_days = pd.read_csv("outputs/tables/anomaly_top_days.csv")
        return summary, by_season, top_days
    except:
        return None, None, None

@st.cache_resource
def load_classifier():
    try:
        model = joblib.load("outputs/models/best_classifier.joblib")
        le = joblib.load("outputs/models/label_encoder.joblib")
        results = pd.read_csv("outputs/tables/classification_results.csv")
        f1 = results.iloc[0]["F1_macro"]
        return model, le, f1
    except Exception as e:
        return None, None, 0.0

# ──────────── SIDEBAR ────────────
with st.sidebar:
    st.markdown("## 🌦️ Weather Mining")
    st.markdown("**Szeged, Hungary**")
    st.markdown("*2006 — 2016*")
    st.markdown("---")
    page = st.radio(
        "📋 ĐIỀU HƯỚNG",
        [
            "🌍 Bảng điều khiển Thời tiết",
            "🧠 Khai phá Tri thức",
            "🔮 Cỗ máy AI Dự báo",
        ],
        label_visibility="collapsed",
    )
    st.markdown("---")
    st.caption("Phần của Lâm · Nhã · Đông")

df, config = load_data()

# ═══════════════════════════════════════════════════════════════
# TRANG 1: BẢNG ĐIỀU KHIỂN THỜI TIẾT
# ═══════════════════════════════════════════════════════════════
if page == "🌍 Bảng điều khiển Thời tiết":
    st.markdown('<p class="main-title">DỰ BÁO THỜI TIẾT SZEGED</p>', unsafe_allow_html=True)
    st.markdown('<p class="sub-title">Phân tích dữ liệu khí tượng 10 năm (2006 — 2016) · Hungary</p>', unsafe_allow_html=True)

    if df is not None:
        # ── KPI Metrics ──
        col1, col2, col3, col4 = st.columns(4)
        col1.metric("🌡️ Nhiệt độ TB", f"{df['Temperature (C)'].mean():.1f}°C",
                     delta=f"Min {df['Temperature (C)'].min():.0f}°C")
        col2.metric("💧 Độ ẩm TB", f"{df['Humidity'].mean():.0%}",
                     delta=f"Max {df['Humidity'].max():.0%}")
        col3.metric("💨 Gió TB", f"{df['Wind Speed (km/h)'].mean():.1f} km/h",
                     delta=f"Max {df['Wind Speed (km/h)'].max():.0f} km/h")
        col4.metric("📊 Tổng quan trắc", f"{len(df):,}",
                     delta="96K+ bản ghi")

        # ── KPI Row 2: Thêm thống kê ──
        st.markdown("")
        c5, c6, c7, c8 = st.columns(4)
        c5.metric("🌡️ Khoảng nhiệt độ",
                  f"{df['Temperature (C)'].max() - df['Temperature (C)'].min():.1f}°C",
                  delta=f"{df['Temperature (C)'].min():.0f}°C → {df['Temperature (C)'].max():.0f}°C")
        c6.metric("📊 Áp suất TB", f"{df['Pressure (millibars)'].mean():.0f} mb",
                  delta=f"Min {df['Pressure (millibars)'].min():.0f} mb")
        c7.metric("👁️ Tầm nhìn TB", f"{df['Visibility (km)'].mean():.1f} km",
                  delta=f"Min {df['Visibility (km)'].min():.1f} km")
        if "Year" in df.columns:
            n_years = df["Year"].nunique()
            n_days = int(len(df) / 24) if len(df) > 8760 else len(df)
            c8.metric("📅 Phạm vi thời gian", f"{n_years} năm", delta=f"~{n_days:,} ngày")
        else:
            c8.metric("📅 Số cột dữ liệu", f"{df.shape[1]} cột")

        st.markdown("")

        # ── Interactive Plotly Chart ──
        st.markdown('<div class="section-header">📈 Biến động Nhiệt độ & Độ ẩm theo thời gian</div>', unsafe_allow_html=True)

        if "Formatted Date" in df.columns:
            df_monthly = df.set_index("Formatted Date").resample("ME").agg({
                "Temperature (C)": "mean",
                "Humidity": "mean",
            }).reset_index()

            fig = go.Figure()
            fig.add_trace(go.Scatter(
                x=df_monthly["Formatted Date"],
                y=df_monthly["Temperature (C)"],
                name="🌡️ Nhiệt độ (°C)",
                line=dict(color="#FF6B6B", width=2.5),
                fill="tozeroy",
                fillcolor="rgba(255,107,107,0.1)",
            ))
            fig.add_trace(go.Scatter(
                x=df_monthly["Formatted Date"],
                y=df_monthly["Humidity"] * 40,
                name="💧 Độ ẩm (×40)",
                line=dict(color="#4ECDC4", width=2.5),
                yaxis="y2",
            ))
            fig.update_layout(
                template="plotly_dark",
                height=500,
                margin=dict(l=20, r=20, t=40, b=20),
                xaxis=dict(title="", rangeslider=dict(visible=True)),
                yaxis=dict(title="Nhiệt độ (°C)", side="left", color="#FF6B6B"),
                yaxis2=dict(title="Độ ẩm", side="right", overlaying="y",
                            color="#4ECDC4", tickformat=".0%",
                            range=[0, 1]),
                legend=dict(orientation="h", yanchor="bottom", y=1.02,
                            xanchor="center", x=0.5),
                hovermode="x unified",
            )
            st.plotly_chart(fig, use_container_width=True)

        # ── 2 cột: Phân bố nhiệt độ + Loại thời tiết ──
        col_left, col_right = st.columns(2)

        with col_left:
            st.markdown('<div class="section-header">🌡️ Phân bố Nhiệt độ</div>', unsafe_allow_html=True)
            fig_hist = px.histogram(
                df, x="Temperature (C)", nbins=60,
                color_discrete_sequence=["#FF6B6B"],
                template="plotly_dark",
                labels={"Temperature (C)": "Nhiệt độ (°C)", "count": "Số bản ghi"},
            )
            fig_hist.update_layout(height=350, margin=dict(l=20, r=20, t=20, b=20),
                                    showlegend=False, yaxis_title="Số bản ghi")
            fig_hist.add_vline(x=df["Temperature (C)"].mean(), line_dash="dash",
                               line_color="yellow", annotation_text=f"TB: {df['Temperature (C)'].mean():.1f}°C")
            st.plotly_chart(fig_hist, use_container_width=True)

        with col_right:
            st.markdown('<div class="section-header">🌤️ Phân bố Loại Thời tiết</div>', unsafe_allow_html=True)
            if "Summary" in df.columns:
                summary_counts = df["Summary"].value_counts().head(8).reset_index()
                summary_counts.columns = ["Loại", "Số lượng"]
                fig_donut = px.pie(
                    summary_counts, names="Loại", values="Số lượng",
                    hole=0.45, template="plotly_dark",
                    color_discrete_sequence=px.colors.qualitative.Set2,
                )
                fig_donut.update_layout(height=350, margin=dict(l=20, r=20, t=20, b=20),
                                         legend=dict(font=dict(size=10)))
                fig_donut.update_traces(textposition="inside", textinfo="percent+label",
                                         textfont_size=10)
                st.plotly_chart(fig_donut, use_container_width=True)

        # ── 2 cột: Nhiệt độ theo năm + Phân bố mùa ──
        col_a, col_b = st.columns(2)

        with col_a:
            st.markdown('<div class="section-header">📊 Xu hướng Nhiệt độ TB theo Năm</div>', unsafe_allow_html=True)
            if "Year" in df.columns:
                yearly = df.groupby("Year").agg(
                    Temp_Mean=("Temperature (C)", "mean"),
                    Temp_Min=("Temperature (C)", "min"),
                    Temp_Max=("Temperature (C)", "max"),
                ).reset_index()
                fig_year = go.Figure()
                fig_year.add_trace(go.Scatter(
                    x=yearly["Year"], y=yearly["Temp_Max"],
                    name="Cao nhất", line=dict(color="#FF6B6B", width=1),
                    fill=None, mode="lines",
                ))
                fig_year.add_trace(go.Scatter(
                    x=yearly["Year"], y=yearly["Temp_Min"],
                    name="Thấp nhất", line=dict(color="#4ECDC4", width=1),
                    fill="tonexty", fillcolor="rgba(78,205,196,0.15)", mode="lines",
                ))
                fig_year.add_trace(go.Bar(
                    x=yearly["Year"], y=yearly["Temp_Mean"],
                    name="Trung bình", marker_color="#667eea", opacity=0.7,
                ))
                fig_year.update_layout(template="plotly_dark", height=350,
                                        margin=dict(l=20, r=20, t=20, b=20),
                                        yaxis_title="°C", xaxis_title="",
                                        legend=dict(orientation="h", y=1.08, x=0.5, xanchor="center"),
                                        barmode="overlay")
                st.plotly_chart(fig_year, use_container_width=True)

        with col_b:
            st.markdown('<div class="section-header">🍂 Phân bố theo Mùa</div>', unsafe_allow_html=True)
            if "Season" in df.columns:
                season_data = df["Season"].value_counts().reset_index()
                season_data.columns = ["Mùa", "Số bản ghi"]
                fig_season = px.bar(
                    season_data, x="Mùa", y="Số bản ghi",
                    color="Mùa",
                    color_discrete_map={"Spring": "#4CAF50", "Summer": "#FF9800",
                                        "Autumn": "#9C27B0", "Winter": "#2196F3"},
                    template="plotly_dark",
                )
                fig_season.update_layout(height=350, showlegend=False,
                                         margin=dict(l=20, r=20, t=20, b=20))
                st.plotly_chart(fig_season, use_container_width=True)

        # ── Correlation Heatmap ──
        st.markdown('<div class="section-header">🔥 Ma trận Tương quan (Correlation Heatmap)</div>', unsafe_allow_html=True)
        num_cols = ["Temperature (C)", "Apparent Temperature (C)", "Humidity",
                    "Wind Speed (km/h)", "Wind Bearing (degrees)", "Visibility (km)", "Pressure (millibars)"]
        available_cols = [c for c in num_cols if c in df.columns]
        if len(available_cols) >= 3:
            corr = df[available_cols].corr()
            # Shorten labels
            short_labels = {"Temperature (C)": "Temp", "Apparent Temperature (C)": "Cảm nhận",
                            "Humidity": "Độ ẩm", "Wind Speed (km/h)": "Gió",
                            "Wind Bearing (degrees)": "Hướng gió", "Visibility (km)": "Tầm nhìn",
                            "Pressure (millibars)": "Áp suất"}
            corr = corr.rename(index=short_labels, columns=short_labels)
            fig_corr = px.imshow(
                corr, text_auto=".2f", color_continuous_scale="RdBu_r",
                zmin=-1, zmax=1, template="plotly_dark",
                aspect="auto",
            )
            fig_corr.update_layout(height=450, margin=dict(l=20, r=20, t=20, b=20))
            st.plotly_chart(fig_corr, use_container_width=True)

        # ── Phân tích Mưa / Tuyết theo tháng ──
        if "Precip Type" in df.columns and "Month" in df.columns:
            st.markdown('<div class="section-header">🌧️ Phân bố Mưa & Tuyết theo Tháng</div>', unsafe_allow_html=True)
            precip_month = df.groupby(["Month", "Precip Type"]).size().reset_index(name="Count")
            month_names = {1:"T1",2:"T2",3:"T3",4:"T4",5:"T5",6:"T6",
                           7:"T7",8:"T8",9:"T9",10:"T10",11:"T11",12:"T12"}
            precip_month["Tháng"] = precip_month["Month"].map(month_names)
            fig_precip = px.bar(
                precip_month, x="Tháng", y="Count", color="Precip Type",
                barmode="group", template="plotly_dark",
                color_discrete_map={"rain": "#4ECDC4", "snow": "#A5B4FC"},
                labels={"Count": "Số bản ghi", "Precip Type": "Loại"},
            )
            fig_precip.update_layout(height=350, margin=dict(l=20, r=20, t=20, b=20),
                                      legend=dict(orientation="h", y=1.08, x=0.5, xanchor="center"))
            st.plotly_chart(fig_precip, use_container_width=True)

    else:
        st.warning("⚠️ Chưa có dữ liệu. Chạy `python scripts/run_pipeline.py` trước.")


# ═══════════════════════════════════════════════════════════════
# TRANG 2: KHAI PHÁ TRI THỨC
# ═══════════════════════════════════════════════════════════════
elif page == "🧠 Khai phá Tri thức":
    st.markdown('<p class="main-title">KHAI PHÁ TRI THỨC</p>', unsafe_allow_html=True)
    st.markdown('<p class="sub-title">Phát hiện luật kết hợp & phân cụm mô hình thời tiết</p>', unsafe_allow_html=True)

    tab1, tab2, tab3 = st.tabs(["🔗 Luật Kết Hợp", "🎯 Phân Cụm K-Means", "🔍 Phát hiện Bất thường"])

    # ── TAB 1: LUẬT KẾT HỢP ──
    with tab1:
        # ═══ HÀM PHIÊN DỊCH NGÔN NGỮ TỰ NHIÊN ═══
        def translate_rule_to_human(rule_str: str) -> tuple:
            """Chuyển đổi luật thô thành Tiếng Việt dễ hiểu."""
            import re

            # Summary mapping
            summary_vn = {
                "Partly Cloudy": "Có mây rải rác", "Mostly Cloudy": "Nhiều mây",
                "Overcast": "U ám", "Clear": "Quang đãng", "Foggy": "Sương mù",
                "Breezy": "Gió nhẹ", "Breezy and Mostly Cloudy": "Gió nhẹ & nhiều mây",
                "Breezy and Overcast": "Gió nhẹ & u ám",
                "Breezy and Partly Cloudy": "Gió nhẹ & ít mây",
            }

            def translate_part(part: str) -> str:
                part = part.strip()
                # Summary=...
                m = re.match(r"Summary=(.+)", part)
                if m:
                    val = m.group(1).strip()
                    return f"Thời tiết: {summary_vn.get(val, val)}"
                # Precip Type
                if "Precip Type=rain" in part:
                    return "Có mưa 🌧️"
                if "Precip Type=snow" in part:
                    return "Có tuyết ❄️"
                # _Bin variables
                part = part.replace("_Bin", "")
                part = part.replace("Humidity", "Độ ẩm")
                part = part.replace("Temp", "Nhiệt độ")
                part = part.replace("Wind", "Sức gió")
                part = part.replace("Pressure", "Áp suất")
                part = part.replace("Visibility", "Tầm nhìn")
                part = part.replace("=Low", ": Thấp")
                part = part.replace("=Normal", ": Bình thường")
                part = part.replace("=High", ": Cao")
                return part

            # Tách vế trái -> vế phải
            if " -> " in rule_str:
                left, right = rule_str.split(" -> ", 1)
            elif "→" in rule_str:
                left, right = rule_str.split("→", 1)
            else:
                return translate_part(rule_str), "", rule_str

            left_parts = [translate_part(p) for p in left.split(", ")]
            right_parts = [translate_part(p) for p in right.split(", ")]

            left_vn = " **VÀ** ".join(f"[{p}]" for p in left_parts)
            right_vn = " **VÀ** ".join(f"[{p}]" for p in right_parts)

            full = f"{left_vn}  ➡️  DẪN ĐẾN:  {right_vn}"
            return full, left_vn, right_vn

        # ═══ LOAD & RENDER ═══
        rules = load_rules()
        if rules is not None:
            st.markdown('<div class="section-header">🔍 Khám phá quy luật thời tiết theo Mùa</div>', unsafe_allow_html=True)

            season_map = {
                "🌸 Xuân (Spring)": ("Spring", "Xuân"),
                "☀️ Hạ (Summer)": ("Summer", "Hạ"),
                "🍂 Thu (Autumn)": ("Autumn", "Thu"),
                "❄️ Đông (Winter)": ("Winter", "Đông"),
            }
            selected = st.selectbox("🔍 Chọn mùa để phân tích:", list(season_map.keys()))
            season_key, season_vn = season_map[selected]

            filtered = rules[rules["Season"] == season_key].sort_values("lift", ascending=False).head(5)

            if not filtered.empty:
                # ── Format bảng đẹp ──
                display_df = filtered.copy().reset_index(drop=True)
                display_df["Quy luật thời tiết"] = display_df["rule"].apply(
                    lambda r: translate_rule_to_human(r)[0]
                )
                display_df["Độ phổ biến (Support)"] = display_df["support"].apply(lambda x: f"{x:.4f}")
                display_df["Độ tin cậy (Conf)"] = display_df["confidence"].apply(lambda x: f"{x*100:.2f}%")
                display_df["Độ mạnh (Lift)"] = display_df["lift"].apply(lambda x: f"{x:.2f}")

                st.dataframe(
                    display_df[["Quy luật thời tiết", "Độ phổ biến (Support)", "Độ tin cậy (Conf)", "Độ mạnh (Lift)"]],
                    use_container_width=True, hide_index=True,
                )

                # ── Storytelling Insight ──
                top = filtered.iloc[0]
                full_rule, left_vn, right_vn = translate_rule_to_human(top["rule"])
                conf_pct = top["confidence"] * 100
                lift_val = top["lift"]

                st.markdown(f"""
> 💡 **Góc nhìn chuyên gia (Mùa {season_vn}):** Dữ liệu lịch sử 10 năm chỉ ra một quy luật
> thời tiết rất đáng chú ý. Khi hiện tượng {left_vn} xuất hiện cùng lúc,
> thì có tới **{conf_pct:.1f}%** khả năng ngay sau đó sẽ xảy ra {right_vn}.
> Sự cộng hưởng này có tính quy luật cao gấp **{lift_val:.1f} lần** so với những ngày bình thường.
>
> 🎯 **Khuyến nghị thực tế:** Phát hiện này cực kỳ hữu ích để dự báo các đợt biến động
> nhiệt độ bất thường, hỗ trợ người dân chuẩn bị trang phục phù hợp hoặc nông dân
> lên kế hoạch gieo trồng theo mùa.
                """)
            else:
                st.warning(f"Không tìm thấy luật cho mùa {season_vn}.")

            st.metric("📊 Tổng số luật đã phát hiện", f"{len(rules)} luật")
        else:
            st.warning("⚠️ Chưa có kết quả. Chạy pipeline trước.")

    # ── TAB 2: PHÂN CỤM ──
    with tab2:
        profiles = load_cluster_profiles()

        if df is not None and profiles is not None:
            # ═══ SCATTER PLOT (sampled 1000 rows) ═══
            st.markdown('<div class="section-header">🔬 Phân bố cụm: Nhiệt độ vs Độ ẩm</div>', unsafe_allow_html=True)

            try:
                from sklearn.cluster import KMeans
                scaled_cols = [c for c in df.columns if c.endswith("_scaled")]
                if len(scaled_cols) >= 2:
                    # Sample 1000 dòng để web không treo
                    df_sample = df.sample(n=min(1000, len(df)), random_state=42).copy()
                    X_sample = df_sample[scaled_cols].values
                    km = KMeans(n_clusters=len(profiles), random_state=42, n_init=10)
                    km.fit(df[scaled_cols].values[:5000])  # fit trên 5K, predict trên 1K
                    df_sample["Cluster"] = km.predict(X_sample)

                    cluster_names = {
                        0: "❄️ Cụm 0: Lạnh & Ẩm ướt",
                        1: "☀️ Cụm 1: Nóng & Hanh khô",
                    }
                    df_sample["Tên cụm"] = df_sample["Cluster"].map(
                        lambda c: cluster_names.get(c, f"Cụm {c}")
                    )

                    fig_scatter = px.scatter(
                        df_sample,
                        x="Temperature (C)", y="Humidity",
                        color="Tên cụm",
                        labels={"Temperature (C)": "🌡️ Nhiệt độ (°C)", "Humidity": "💧 Độ ẩm"},
                        color_discrete_map={
                            "❄️ Cụm 0: Lạnh & Ẩm ướt": "#3B82F6",
                            "☀️ Cụm 1: Nóng & Hanh khô": "#F97316",
                        },
                        opacity=0.6,
                        template="plotly_dark",
                    )
                    fig_scatter.update_layout(
                        height=500,
                        margin=dict(l=20, r=20, t=40, b=20),
                        legend=dict(orientation="h", yanchor="bottom", y=1.02,
                                    xanchor="center", x=0.5, font=dict(size=13)),
                    )
                    fig_scatter.update_traces(marker=dict(size=7))
                    st.plotly_chart(fig_scatter, use_container_width=True)
            except Exception as e:
                st.warning(f"Lỗi vẽ scatter: {e}")

            # ═══ HỒ SƠ CỤM ═══
            st.markdown('<div class="section-header">📋 Hồ sơ cụm (Cluster Profile)</div>', unsafe_allow_html=True)

            for _, row in profiles.iterrows():
                cluster_id = row.get("Cluster", row.name)
                name = row.get("Cluster_Name", f"Cụm {cluster_id}")
                temp = row.get("Temperature (C)", 0)
                hum = row.get("Humidity", 0)
                wind = row.get("Wind Speed (km/h)", 0)
                vis = row.get("Visibility (km)", 0)
                prs = row.get("Pressure (millibars)", 0)
                count = row.get("Count", 0)

                details = (
                    f"🌡️ Nhiệt độ TB: **{temp:.1f}°C** · "
                    f"💧 Độ ẩm: **{hum:.0%}** · "
                    f"💨 Gió: **{wind:.1f} km/h** · "
                    f"👁️ Tầm nhìn: **{vis:.1f} km** · "
                    f"📊 Áp suất: **{prs:.0f} mb** · "
                    f"📈 Số mẫu: **{int(count):,}**"
                )

                if temp < 10:
                    st.info(f"❄️ **{name}**\n\n{details}")
                else:
                    st.success(f"☀️ **{name}**\n\n{details}")

            # ═══ ACTIONABLE INSIGHTS ═══
            st.markdown('<div class="section-header">💡 ỨNG DỤNG THỰC TIỄN (Actionable Insights)</div>', unsafe_allow_html=True)

            st.markdown("""
Thuật toán **K-Means** đã tự động phân tách khí hậu Szeged thành **2 hình thái đối lập**
mà không cần con người gán nhãn trước. Từ hồ sơ cụm này, chúng ta có thể xây dựng
một **Hệ thống Cảnh báo sớm (Early Warning System)**:

---

🚦 **Giao thông & Cảnh báo an toàn (Dành cho Cụm 0 — Lạnh & Ẩm ướt):**
Với đặc trưng *"Tầm nhìn thấp, Ẩm ướt"*, cụm này chứa đựng rủi ro tai nạn liên hoàn cực cao.
**👉 Khuyến nghị:** Tích hợp dữ liệu phân cụm này vào hệ thống Smart City.
Khi AI nhận diện thời tiết tiệm cận vào Cụm 0, hệ thống biển báo điện tử trên cao tốc
phải tự động giảm tốc độ giới hạn và bật đèn sương mù.

---

🌾 **Nông nghiệp & Chuỗi cung ứng (Dành cho Cụm 1 — Nóng & Hanh khô):**
Đặc trưng *"Nóng, Hanh khô"* làm tăng tốc độ bốc hơi nước bề mặt, dễ gây sốc nhiệt cho cây trồng.
**👉 Khuyến nghị:** Các trang trại nông nghiệp cần thiết lập hệ thống tưới tiêu tự động
tăng công suất khi bước vào các ngày Cụm 1. Đồng thời, hệ thống siêu thị bán lẻ
cần đẩy mạnh dự trữ nước giải khát và thiết bị làm mát.
            """)
        else:
            st.warning("⚠️ Chưa có kết quả phân cụm.")

    # ── TAB 3: ANOMALY DETECTION ──
    with tab3:
        st.markdown('<div class="section-header">🔍 Phát hiện Ngày Thời Tiết Bất Thường</div>', unsafe_allow_html=True)
        st.markdown("So sánh **Isolation Forest** vs **Local Outlier Factor (LOF)** để tìm ngày thời tiết cực đoan.")

        anom_summary, anom_season, anom_top = load_anomaly_results()

        if anom_summary is not None:
            # ── KPI Cards ──
            a1, a2, a3 = st.columns(3)
            if len(anom_summary) >= 3:
                a1.metric("🌲 Isolation Forest", f"{int(anom_summary.iloc[0]['N_Anomalies']):,} ngày",
                          delta=f"{anom_summary.iloc[0]['Pct_Anomalies']:.1f}%")
                a2.metric("🔵 LOF", f"{int(anom_summary.iloc[1]['N_Anomalies']):,} ngày",
                          delta=f"{anom_summary.iloc[1]['Pct_Anomalies']:.1f}%")
                a3.metric("🎯 Consensus (cả hai)", f"{int(anom_summary.iloc[2]['N_Anomalies']):,} ngày",
                          delta=f"{anom_summary.iloc[2]['Pct_Anomalies']:.1f}%")

            st.markdown("")

            # ── Anomaly theo mùa ──
            if anom_season is not None and not anom_season.empty:
                st.markdown('<div class="section-header">📊 Phân bố Anomaly theo Mùa</div>', unsafe_allow_html=True)

                fig_as = go.Figure()
                fig_as.add_trace(go.Bar(
                    x=anom_season["Season"], y=anom_season["IF_Pct"],
                    name="Isolation Forest", marker_color="#FF6B6B",
                ))
                fig_as.add_trace(go.Bar(
                    x=anom_season["Season"], y=anom_season["LOF_Pct"],
                    name="LOF", marker_color="#FF9800",
                ))
                fig_as.add_trace(go.Bar(
                    x=anom_season["Season"], y=anom_season["Consensus_Pct"],
                    name="Consensus", marker_color="#9C27B0",
                ))
                fig_as.update_layout(
                    template="plotly_dark", height=400, barmode="group",
                    yaxis_title="% Ngày bất thường",
                    margin=dict(l=20, r=20, t=20, b=20),
                    legend=dict(orientation="h", y=1.08, x=0.5, xanchor="center"),
                )
                st.plotly_chart(fig_as, use_container_width=True)

            # ── Top ngày bất thường ──
            if anom_top is not None and not anom_top.empty:
                st.markdown('<div class="section-header">🚨 Top 10 Ngày Bất Thường Nhất</div>', unsafe_allow_html=True)
                display_cols = [c for c in ["Formatted Date", "Consensus", "Season",
                                "Temperature (C)", "Humidity", "Wind Speed (km/h)",
                                "Visibility (km)", "Pressure (millibars)"] if c in anom_top.columns]
                st.dataframe(anom_top[display_cols].head(10), use_container_width=True, hide_index=True)

            # ── Insight ──
            st.markdown("""
> 💡 **Insight:** Mùa Đông có tỷ lệ anomaly **cao nhất** — những ngày cực lạnh, độ ẩm
> cực thấp (0%), gió mạnh và tầm nhìn gần 0 km tạo thành các ngày thời tiết bất thường.
> Phát hiện này hữu ích cho **hệ thống cảnh báo sớm** và **đánh giá rủi ro khí hậu**.
            """)
        else:
            st.warning("⚠️ Chưa có kết quả anomaly. Chạy pipeline Phase 4 trước.")


# ═══════════════════════════════════════════════════════════════
# TRANG 3: CỖ MÁY AI DỰ BÁO
# ═══════════════════════════════════════════════════════════════
elif page == "🔮 Cỗ máy AI Dự báo":
    st.markdown('<p class="main-title">CỖ MÁY AI DỰ BÁO</p>', unsafe_allow_html=True)
    st.markdown('<p class="sub-title">Dự đoán thời tiết & dự báo chuỗi thời gian bằng Machine Learning</p>', unsafe_allow_html=True)

    tab_cls, tab_fc = st.tabs(["🏷️ Dự đoán Loại Thời Tiết", "📈 Dự báo Chuỗi Thời Gian"])

    # ── TAB 1: DỰ ĐOÁN LOẠI THỜI TIẾT ──
    with tab_cls:
        model, le, f1_score_val = load_classifier()

        st.markdown('<div class="section-header">Nhập thông số để AI dự đoán</div>', unsafe_allow_html=True)

        with st.form("prediction_form"):
            col1, col2 = st.columns(2)
            with col1:
                temp = st.slider("🌡️ Nhiệt độ (°C)", -20.0, 40.0, 15.0, 0.5)
                humidity = st.slider("💧 Độ ẩm", 0.0, 1.0, 0.65, 0.01)
                wind_speed = st.slider("💨 Tốc độ gió (km/h)", 0.0, 60.0, 12.0, 0.5)
            with col2:
                apparent_temp = st.slider("🌡️ Nhiệt độ cảm nhận (°C)", -25.0, 45.0, 13.0, 0.5)
                visibility = st.slider("👁️ Tầm nhìn (km)", 0.0, 16.0, 10.0, 0.5)
                pressure = st.slider("📊 Áp suất (mbar)", 950.0, 1050.0, 1015.0, 1.0)
                wind_bearing = st.slider("🧭 Hướng gió (°)", 0.0, 360.0, 180.0, 5.0)

            submitted = st.form_submit_button("🔮 DỰ ĐOÁN!", use_container_width=True, type="primary")

        if submitted:
            if model is not None and le is not None:
                features = np.array([[temp, apparent_temp, humidity, wind_speed,
                                      wind_bearing, visibility, pressure]])
                prediction = model.predict(features)
                label = le.inverse_transform(prediction)[0]

                # Vietnamese + Emoji map
                vn_map = {
                    "Clear": ("TRỜI QUANG ĐÃN", "☀️"),
                    "Cloudy": ("TRỜI NHIỀU MÂY", "☁️"),
                    "Rain": ("TRỜI MƯA", "🌧️"),
                    "Foggy": ("TRỜI SƯƠNG MÙ", "🌫️"),
                    "Windy": ("TRỜI CÓ GIÓ MẠNH", "💨"),
                }
                vn_label, emoji = vn_map.get(label, (label.upper(), "🌤️"))

                # ═══ KẾT QUẢ DỰ BÁO — THẬT TO ═══
                st.markdown(
                    f"<h2 style='text-align:center; margin-top:1rem;'>"
                    f"{emoji} KẾT QUẢ DỰ BÁO {emoji}</h2>"
                    f"<div class='big-prediction' style='background:linear-gradient(135deg,#667eea,#764ba2);color:white;'>"
                    f"{emoji} {vn_label} {emoji}</div>"
                    f"<p style='text-align:center;color:#888;font-size:0.9rem;'>Phân loại gốc: <b>{label}</b></p>",
                    unsafe_allow_html=True,
                )

                # Bảng tham số
                st.markdown("**Tham số đã nhập:**")
                params_df = pd.DataFrame({
                    "Tham số": ["Nhiệt độ", "Cảm nhận", "Độ ẩm", "Gió", "Hướng gió", "Tầm nhìn", "Áp suất"],
                    "Giá trị": [f"{temp}°C", f"{apparent_temp}°C", f"{humidity:.0%}",
                                f"{wind_speed} km/h", f"{wind_bearing}°", f"{visibility} km", f"{pressure} mbar"],
                })
                st.table(params_df)

                st.info(f"ℹ️ Mô hình: **Random Forest** · F1-macro = **{f1_score_val:.4f}** · Số nhóm thời tiết: **5**")

                # ── Bảng so sánh 4 mô hình ──
                try:
                    cls_results = pd.read_csv("outputs/tables/classification_results.csv")
                    if not cls_results.empty:
                        st.markdown('<div class="section-header">📊 So sánh Hiệu năng 4 Mô hình</div>', unsafe_allow_html=True)
                        fig_comp = go.Figure()
                        fig_comp.add_trace(go.Bar(
                            x=cls_results["Model"], y=cls_results["F1_macro"],
                            name="F1-macro", marker_color="#667eea",
                            text=cls_results["F1_macro"].apply(lambda x: f"{x:.4f}"),
                            textposition="outside",
                        ))
                        fig_comp.add_trace(go.Bar(
                            x=cls_results["Model"], y=cls_results["Accuracy"],
                            name="Accuracy", marker_color="#4ECDC4",
                            text=cls_results["Accuracy"].apply(lambda x: f"{x:.4f}"),
                            textposition="outside",
                        ))
                        fig_comp.update_layout(
                            template="plotly_dark", height=400, barmode="group",
                            yaxis_title="Score", yaxis_range=[0, 1.05],
                            margin=dict(l=20, r=20, t=20, b=20),
                            legend=dict(orientation="h", y=1.08, x=0.5, xanchor="center"),
                        )
                        st.plotly_chart(fig_comp, use_container_width=True)
                except:
                    pass
            else:
                st.error("❌ Chưa có model. Chạy `python src/models/classification.py` trước!")

    # ── TAB 2: DỰ BÁO CHUỖI THỜI GIAN ──
    with tab_fc:
        st.markdown('<div class="section-header">📈 Dự báo nhiệt độ tương lai (Holt-Winters)</div>', unsafe_allow_html=True)

        forecast_days = st.slider("📅 Số ngày dự báo tương lai:", 7, 60, 30, 1)

        if st.button("📈 Chạy AI Dự báo!", use_container_width=True, type="primary"):
            if df is not None and "Formatted Date" in df.columns:
                with st.spinner("🔄 Đang xử lý dữ liệu & dự báo..."):
                    try:
                        # ═══ FIX 1: Interpolation — lấp đầy ngày thiếu ═══
                        ts_raw = df.set_index("Formatted Date")["Temperature (C)"]
                        ts = ts_raw.resample("D").mean()           # Tạo grid ngày đầy đủ
                        ts = ts.interpolate(method="time")         # Nội suy ngày bị thiếu
                        ts = ts.dropna()

                        # Holt-Winters forecast (cần >= 2 chu kỳ)
                        from statsmodels.tsa.holtwinters import ExponentialSmoothing

                        train_data = ts[-730:]  # 2 năm cuối
                        model_hw = ExponentialSmoothing(
                            train_data,
                            seasonal_periods=365,
                            trend="add",
                            seasonal="add",
                            use_boxcox=False,
                        ).fit(optimized=True)

                        forecast = model_hw.forecast(forecast_days)

                        # ═══ FIX 3: Tính MAE/RMSE bằng backtesting ═══
                        backtest_days = 30
                        if len(ts) > 730 + backtest_days:
                            train_bt = ts[-(730 + backtest_days):-backtest_days]
                            actual_bt = ts[-backtest_days:]
                            model_bt = ExponentialSmoothing(
                                train_bt, seasonal_periods=365,
                                trend="add", seasonal="add", use_boxcox=False,
                            ).fit(optimized=True)
                            pred_bt = model_bt.forecast(backtest_days)
                            mae = float(np.abs(actual_bt.values - pred_bt.values).mean())
                            rmse = float(np.sqrt(((actual_bt.values - pred_bt.values) ** 2).mean()))
                        else:
                            # Tính từ fitted values
                            residuals = model_hw.resid.dropna()
                            mae = float(np.abs(residuals).mean())
                            rmse = float(np.sqrt((residuals ** 2).mean()))

                        # ── MAE / RMSE Metrics ──
                        m1, m2 = st.columns(2)
                        m1.metric("📏 Sai số tuyệt đối TB (MAE)", f"{mae:.2f} °C")
                        m2.metric("📐 Sai số RMSE", f"{rmse:.2f} °C")

                        # ═══ FIX 1+2: Vẽ biểu đồ liền mạch ═══
                        history_show = ts.tail(150)  # 150 ngày gần nhất, đã interpolated

                        # FIX 2: Nối liền — thêm điểm cuối lịch sử vào đầu forecast
                        last_hist_date = history_show.index[-1]
                        last_hist_val = history_show.values[-1]
                        forecast_seamless = pd.concat([
                            pd.Series([last_hist_val], index=[last_hist_date]),
                            forecast,
                        ])

                        fig_fc = go.Figure()
                        fig_fc.add_trace(go.Scatter(
                            x=history_show.index,
                            y=history_show.values,
                            name="📊 Dữ liệu lịch sử (150 ngày)",
                            line=dict(color="#4ECDC4", width=2),
                            mode="lines",
                        ))
                        fig_fc.add_trace(go.Scatter(
                            x=forecast_seamless.index,
                            y=forecast_seamless.values,
                            name=f"🔮 Dự báo {forecast_days} ngày",
                            line=dict(color="#FF6B6B", width=3, dash="dot"),
                            fill="tozeroy",
                            fillcolor="rgba(255,107,107,0.08)",
                            mode="lines",
                        ))

                        fig_fc.update_layout(
                            template="plotly_dark",
                            height=500,
                            margin=dict(l=20, r=20, t=40, b=20),
                            xaxis_title="",
                            yaxis_title="Nhiệt độ (°C)",
                            hovermode="x unified",
                            legend=dict(orientation="h", yanchor="bottom", y=1.02,
                                        xanchor="center", x=0.5),
                        )
                        st.plotly_chart(fig_fc, use_container_width=True)

                        # ── Forecast Stats ──
                        c1, c2, c3 = st.columns(3)
                        c1.metric("📈 TB dự báo", f"{forecast.mean():.1f}°C")
                        c2.metric("🔺 Cao nhất", f"{forecast.max():.1f}°C")
                        c3.metric("🔻 Thấp nhất", f"{forecast.min():.1f}°C")

                        # ═══ FIX 3: Residual Analysis + Actionable Insights ═══
                        with st.expander("🔎 Đánh giá Mô hình & Cảnh báo (Actionable Insights)", expanded=True):
                            st.markdown(f"""
📊 **Đánh giá Sai số (Residual Analysis):** Mô hình Holt-Winters đạt sai số tuyệt đối
trung bình (MAE) khoảng **{mae:.2f} °C** và RMSE = **{rmse:.2f} °C**. Phân tích phần dư
cho thấy AI nắm bắt tốt chu kỳ mùa vụ nhưng thường bị **"trễ nhịp" (lag)** tại các
điểm cực trị (Outliers) — đặc biệt khi có đợt rét đậm hoặc nắng nóng bất thường.

🚨 **Cảnh báo Thời tiết:** Trong chu kỳ dự báo **{forecast_days} ngày** tới, nhiệt độ
có khả năng chạm mức thấp nhất là **{forecast.min():.1f} °C** và cao nhất
là **{forecast.max():.1f} °C**.

👉 **Khuyến nghị thực tiễn:** Các hệ thống lưới điện và nông nghiệp cần đề phòng sai số
dự báo hụt khi có các đợt **không khí lạnh tràn về đột ngột** (Frigid air outbreaks)
không tuân theo quy luật lịch sử. Hướng phát triển tương lai là tích hợp thêm biến
ngoại sinh **"Áp suất"** và **"Hướng gió"** vào mô hình **SARIMAX** để khử nhiễu phần dư.
                            """)

                    except Exception as e:
                        st.warning(f"⚠️ Lỗi dự báo Holt-Winters: {e}. Sử dụng Naive Forecast.")

                        # Fallback: Naive (same period last year)
                        ts_daily = df.set_index("Formatted Date")["Temperature (C)"].resample("D").mean()
                        ts_daily = ts_daily.interpolate(method="time").dropna()
                        history_show = ts_daily.tail(150)
                        naive_fc = ts_daily[-forecast_days:].values
                        dates = pd.date_range(ts_daily.index[-1] + pd.Timedelta(days=1), periods=forecast_days)

                        # Seamless
                        fc_vals = np.concatenate([[history_show.values[-1]], naive_fc])
                        fc_dates = pd.DatetimeIndex([history_show.index[-1]]).append(dates)

                        fig_naive = go.Figure()
                        fig_naive.add_trace(go.Scatter(
                            x=history_show.index, y=history_show.values,
                            name="Lịch sử", line=dict(color="#4ECDC4", width=2),
                        ))
                        fig_naive.add_trace(go.Scatter(
                            x=fc_dates, y=fc_vals,
                            name=f"Naive {forecast_days}d", line=dict(color="#FF6B6B", width=3, dash="dot"),
                        ))
                        fig_naive.update_layout(template="plotly_dark", height=450)
                        st.plotly_chart(fig_naive, use_container_width=True)
            else:
                st.warning("⚠️ Chưa có dữ liệu.")
