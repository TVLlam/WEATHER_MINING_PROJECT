

## HƯỚNG DẪN THỰC HIỆN BÀI TẬP LỚN
## HỌC PHẦN DỮ LIỆU LỚN, KHAI PHÁ DỮ LIỆU; KHAI PHÁ DỮ LIỆU -
## HỌC KÌ II NĂM HỌC 2025–2026
ThS. Lê Thị Thùy Trang
## 2025-12-23
1 Mục tiêu của bài tập lớn
1.1 Về mặt kiến thức:
Sinh viên vận dụng kiến thức về khám phá dữ liệu (EDA), tiền xử lý, trích xuất/thiết kế đặc trưng, và các kỹ thuật khai phá
dữ liệu đã học để xây dựng pipeline khai phá và đánh giá kết quả một cách khoa học.
Bài tập lớn tập trung vào các nhóm bài toán
- Khai phá mẫu/tri thức (association patterns)
- Phân cụm (clustering + diễn giải cụm)
- Phân lớp (classification) và bán giám sát khi thiếu nhãn
- Hồi quy/chuỗi thời gian (nếu đề tài thuộc nhóm dự báo)
1.2 Về mặt kỹ năng:
•Rèn luyện kỹ năng cốt lõi của một người làm dữ liệu: tiền xử lý, tạo đặc trưng, thử nghiệm–so sánh phương pháp và trình
bày kết quả theo lập luận khoa học.
- Rèn luyện kỹ năng làm việc nhóm, nghiên cứu, viết báo cáo và thuyết trình dự án.
2 Yêu cầu chung
Sinh viên làm việc theo nhóm từ2đến4thành viên.
Đề tài bài tập lớndo giảng viên chỉ định theo danh sách đề bài ở phần1danh sách đề tài.
Dataset sử dụng đúng link được cung cấp trong bảng đề tài; nhóm phải ghi rõ nguồn dữ liệu, mô tả cột, ý nghĩa nhãn/target
(data dictionary).
Dự ánbắt buộcthực hiện theo tinh thần “project repo”: có cấu trúc thư mục rõ ràng, module hoá, chạy lại được
(reproducible), KHÔNG sử dụng notebook rời rạc.
## 1

Final Project Data miningLê Thị Thùy Trang
2.1 Quy trình khai phá dữ liệu tổng quát
Quy trình tổng quát tuân theo logic “Nguồn dữ liệu → Tiền xử lý → Đặc trưng/biểu diễn → Mô hình → Đánh giá” như
sau:
- Data Source (Nguồn dữ liệu): Dữ liệu dạng bảng/văn bản/ảnh/chuỗi thời gian/đồ thị... tùy đề tài.
2.Preprocessing (Tiền xử lý): Làm sạch, xử lý thiếu, chuẩn hoá, mã hoá biến phân loại, cân bằng lớp (nếu cần), tạo session
(log), resample (time series)...
3.Feature / Representation (Đặc trưng/biểu diễn): TF-IDF/embeddings (text), đặc trưng ảnh, đặc trưng hành vi (RFM), lag
features (chuỗi thời gian), graph features, vector hoá giỏ hàng...
- Mining / Modeling (Khai phá, mô hình hoá)
- Khai phá tri thức (bắt buộc): pattern mining / clustering / anomaly / rule extraction...
•Mô hình dự đoán (nếu đề tài có supervised): classification/regression/forecasting; ghi rõ hyperparams, thời gian
train, thiết lập thực nghiệm
5.Evaluation and Results (Đánh giá, kết quả): Dùng metric phù hợp, lập bảng/biểu đồ so sánh mô hình, kèm insight và
khuyến nghị hành động.
- Nhánh bổ sung: Bán giám sát (Semi-supervised) – chỉ áp dụng khi đề tài là phân lớp
- Với đề tài phân lớp, nhóm phải thêm một nhánh thực nghiệm “thiếu nhãn”:
- Giữ lại p% nhãn (p = 5/10/20), phần còn lại coi là unlabeled
•So sánh Supervised-only (ít nhãn) vs Semi-supervised (self-training/pseudo-label hoặc label spreading/propagation)
- Báo cáo learning curve theo % nhãn và phân tích rủi ro pseudo-label
Figure 1: Pipeline project
2.2 Nội dung báo cáo (bắt buộc, theo đúng thứ tự)
Báo cáo phải có đủ các phần như sau:
- Đặt vấn đề và phân tích yêu cầu: bối cảnh, mục tiêu, tiêu chí thành công; mô tả dữ liệu và EDA.
- Thiết kế giải pháp và quy trình khai phá: mô tả pipeline; tiền xử lý; đặc trưng; lý do chọn kỹ thuật.
## 2

Final Project Data miningLê Thị Thùy Trang
3.Phân tích mã nguồn và chức năng: tập trung mô tả kiến trúc repo, các module/class chính (DataCleaner/Feature-
Builder/Miner/Trainer/Evaluator...).
- Thử nghiệm và kết quả: metric phù hợp; bảng/biểu đồ so sánh các phương án.
- Thảo luận và so sánh: so sánh ưu/nhược; giải thích vì sao phương án A tốt hơn B; nêu thách thức gặp phải.
- Tổng kết và hướng phát triển: tóm tắt kết quả và đề xuất cải tiến.
Ghi chú:Với nhóm có bán giám sát, phần (4)–(5) phải có thêm mục “thiếu nhãn: supervised vs semi-supervised”.
2.3 Hình thức báo cáo
Báo cáo viết bằng LaTeX theo template do giảng viên cung cấp hoặc.docxtheo mẫu của trường
Khuyến khích: trình bày nhiều hình/bảng, ít “dán code”; code nằm trong repo.
2.4 Sử dụng Github trong quá trình làm bài tập lớn
Sinh viên phải tổ chức project nhóm theo cấu trúc repo rõ ràng để đảm bảo dễ đọc – dễ chạy lại – dễ đánh giá. Repo cần
tách bạch giữa dữ liệu, notebook báo cáo, mã nguồn (module), cấu hình tham số và kết quả đầu ra.
2.4.1 Cấu trúc repo mẫu
## DATA_MINING_PROJECT/
README.md
requirements.txt# hoặc environment.yml
## .gitignore
configs/
params.yaml# tham số: seed, split, paths, hyperparams...
data/
raw/# dữ liệu gốc (không commit nếu quá lớn)
processed/# dữ liệu sau tiền xử lý (ưu tiên parquet/csv)
notebooks/
## 01_eda.ipynb
## 02_preprocess_feature.ipynb
## 03_mining_or_clustering.ipynb
## 04_modeling.ipynb
04b_semi_supervised.ipynb# CHỈ áp dụng cho đề tài có bán giám sát
## 05_evaluation_report.ipynb
src/
## __init__.py
data/
## __init__.py
loader.py# đọc dữ liệu, kiểm tra schema
cleaner.py# xử lý thiếu, outlier, encoding cơ bản
features/
## __init__.py
builder.py# feature engineering (RFM, TF-IDF, lag, ...)
mining/
## 3

Final Project Data miningLê Thị Thùy Trang
## __init__.py
association.py# (nếu có) luật kết hợp / pattern
clustering.py# KMeans/HAC/DBSCAN + profiling
anomaly.py# (nếu có) outlier/anomaly
models/
## __init__.py
supervised.py# train/predict cho classification/regression
semi_supervised.py# CHỈ áp dụng cho đề tài có bán giám sát
forecasting.py# (nếu có) time series
evaluation/
## __init__.py
metrics.py# accuracy, f1, auc, rmse, mae, ...
report.py# tổng hợp bảng/biểu đồ kết quả
visualization/
## __init__.py
plots.py# hàm vẽ dùng chung
scripts/
run_pipeline.py# chạy toàn bộ pipeline (khuyến khích)
run_papermill.py# (khuyến khích) chạy notebook bằng papermill
outputs/
figures/
tables/
models/
reports/
final_report.pdf
2.4.2 Quy ước đặt tên luồng pipeline
Notebook đặt theo thứ tự 01 → 05 để người chấm chạy/đọc theo pipeline.
src/chứa toàn bộ logic chính; notebook chỉ gọi hàm/lớp và trình bày kết quả.
Tất cả đường dẫn và tham số quan trọng đặt trong configs/params.yaml.
2.4.3 Quy định về dữ liệu và lưu trữ
Không commit dữ liệu lớn vào GitHub.
Nếu dataset nặng, nhóm phải cung cấp:
- link dataset + hướng dẫn tải trongREADME.md, hoặc
- script tải dữ liệu trongscripts/(nếu có thể).
2.4.4 Yêu cầu tái lập
Repo được coi là đạt khi người khác có thể:
1.pip install -r requirements.txt
- Cập nhật đường dẫn dữ liệu trong configs/params.yaml
## 4

- Chạypython scripts/run_papermill.py
- Thu đượcoutputs/và hình/bảng đúng như báo cáo.
2.5 Điểm thưởng
Có GUI/Demo app (Streamlit/Gradio/web nhỏ): cộng điểm theo mức hoàn thiện
3 Danh sách đề tài bài tập lớn

Đề tàiLink datasetLuật kết hợpPhân cụmPhân lớpBán giám sátHồi quy / Chuỗi thời gian
- Phân tích
doanh số siêu thị
Kaggle:Superstore
## Sales Dataset
Chuẩn hoá giỏ hàng theo
hoá đơn.
## Chọn
min_support/min_conf,
xem lift.
Top luật + gợi ý
combo/cross-sell.
Đặc trưng RFM/giá
trị/nhóm hàng.
Chuẩn hoá.
Hồ sơ cụm + insight.
Dự đoán phân khúc
(LogReg/DT/RF).
Đánh giá
F1/ROC-AUC + phân
tích lỗi.
Không thực hiện đối với
đề tài này
Dự báo doanh số theo thời
gian: split theo thời gian,
baseline (naive/MA),
ARIMA/Holt-
Winters/Prophet,
MAE/RMSE/sMAPE, phân
tích residual/seasonality.
- Dự đoán bệnh
tim
UCI/Kaggle:Heart
## Disease Dataset
Rời rạc hoá chỉ số theo
ngưỡng.
Apriori tìm tổ hợp triệu
chứng.
Báo cáo support/conf/lift
+ diễn giải.
Chuẩn hoá đặc trưng.
Mô tả cụm nguy cơ.
Phân lớp nguy cơ
## (SVM/RF/XGB).
## Imbalance:
class_weight/SMOTE.
Metric ưu tiên
PR-AUC/F1 + phân
tích lỗi.
Tuỳ chọn (cân nhắc)vì
dữ liệu thường không
quá lớn.
Nếu làm: giả lập
10–30% nhãn,
self-training với ngưỡng
tin cậy cao, so sánh
PR-AUC theo % nhãn,
phân tích nhóm khó.
Hồi quy chỉ số (ví dụ huyết
áp) theo yếu tố nguy cơ:
Linear/Ridge vs
XGBRegressor, MAE/RMSE,
kiểm tra outlier/leakage;
Không bắt buộc áp dụng
chuỗi thời gian.
- Phân tích cảm
xúc khách hàng
Kaggle:Amazon
## Reviews Dataset
Tách review tích cực/tiêu
cực.
Apriori trên tập từ
khoá/phrase.
Top luật + diễn giải chủ
đề.
TF-IDF/embeddings.
## Topic/cluster
(K-means/HDBSCAN).
Đặt tên cụm theo top
terms.
Silhouette + ví dụ đại
diện.
Phân lớp sentiment
(NB/Lo-
gReg/SVM/LSTM).
F1-macro + confusion
matrix + phân tích lỗi
(sarcasm, review
ngắn).
Giả lập ít nhãn, label
spreading/self-training,
learning curve theo %
nhãn, phân tích
pseudo-label sai theo độ
dài/độ hiếm từ.
Hồi quy rating từ nội dung:
baseline (Ridge/Linear) + mô
hình mạnh (SVR/XGB),
## MAE/RMSE.
Không bắt buộc áp dụng
chuỗi thời gian
- Phân tích giao
dịch ngân hàng
UCI: Bank
## Marketing Dataset
Tạo giỏ sản phẩm/dịch
vụ theo khách.
Top luật cross-sell.
Đề xuất gói sản phẩm.
Phân cụm hồ sơ tài
chính.
Chuẩn hoá.
Đặt tên cụm.
Dự đoán đăng ký term
deposit
(LogReg/RF/XGB).
## Imbalance.
PR-AUC/F1 + giải
thích đặc trưng.
Giả lập 10–30% nhãn,
self-training, so sánh
PR-AUC/lift@k theo %
nhãn, phân tích
pseudo-label sai theo
campaign/nhóm tuổi.
Hồi quy số dư/tài chính theo
tuổi/thu nhập: Linear/Ridge
vs XGBRegressor,
## MAE/RMSE.
Không bắt buộc áp dụng
chuỗi thời gian
- Dự báo thời
tiết
Kaggle:Weather
## Dataset
Rời rạc hoá điều kiện.
Tìm điều kiện đồng xuất
hiện.
So sánh theo mùa.
Top luật + diễn giải.
Phân cụm ngày kiểu thời
tiết.
Chuẩn hoá.
Hồ sơ cụm.
Phân lớp loại thời tiết
## (RF/XGB).
F1-macro + phân tích
lỗi giao mùa/cực trị.
Không bắt buộc áp dụng
học bán giám sát
Sử dụng chuỗi thời gian dự
báo nhiệt độ/độ ẩm:
## ACF/PACF,
ARIMA/Holt-Winters,
MAE/RMSE, residual +
outlier.
- Phân tích hiệu
suất nhân viên
Kaggle:HR
## Analytics Dataset
Rời rạc hoá yếu tố.
Tìm tổ hợp dẫn đến nghỉ
việc.
So sánh lift giữa stay vs
leave.
Gợi ý chính sách.
Phân cụm nhân viên.
Chuẩn hoá.
Profiling cụm.
Phân lớp nghỉ việc
## (XGB/RF).
PR-AUC/F1 + giải
thích SHAP.
Giả lập 10–30% nhãn,
self-training, so sánh
PR-AUC theo % nhãn,
phân tích rủi ro gán nhãn
sai (tác động chính sách).
Hồi quy mức độ hài
lòng/điểm hiệu suất (nếu có):
MAE/RMSE, kiểm tra
leakage.
Chuỗi thời gian thường
không phù hợp
- Dự báo năng
suất cây trồng
Kaggle:Crop
Yield datasets
Rời rạc hoá điều kiện
(đất/thời tiết/giống).
Apriori/ FP Growth tìm
điều kiện đi kèm năng
suất cao.
Top luật + khuyến nghị
canh tác (mức mô tả).
Phân cụm vùng trồng
theo điều kiện.
Chuẩn hoá.
Profiling cụm + so sánh
năng suất.
Sử dụng cộtYield
(biến liên tục) làm
nhãn:ép kiểu số cho
Yield, loại bản ghi
Yield bị thiếu/0 bất
thường.
Baseline + mô hình
mạnh.
F1 + phân tích lỗi vùng
hiếm.
Không yêu cầu áp dụng
học bán giám sát
Dự báo năng suất cây trồng:
Baseline Linear/Ridge +
RF/XGBRegressor.
## MAE/RMSE.
Chuỗi thời gian: split theo
năm, đánh giá
drift/seasonality.
- Phát hiện gian
lận
Kaggle:Credit
## Card Fraud
Rời rạc hoá
(amount/time bins).
Tìm pattern thường gặp
ở fraud.
Top luật + lift (chỉ dùng
như insight/feature).
Phân cụm/anomaly
cluster
(HDBSCAN/K-means).
Chuẩn hoá.
Đánh giá, mô tả cụm bất
thường.
Phân lớp fraud/legit.
Imbalance mạnh.
PR-AUC + chọn
ngưỡng theo chi phí
sai.
Phân tích false
positive/negative.
## Semi-supervised
anomaly + thiếu nhãn.
Giả lập ít nhãn fraud,
self-training hoặc
one-class/PU-learning
(mức cơ bản), learning
curve, phân tích rủi ro
pseudo-label.
Không yêu cầu sử dụng hồi
quy và chuỗi thời gian
- Phân tích chất
lượng nước
Kaggle:Water
quality datasets
Rời rạc hoá chỉ số ô
nhiễm.
Apriori tìm chỉ số hay
cùng tăng.
Top luật + diễn giải theo
ngưỡng an toàn.
Phân cụm nguồn nước
theo profile.
Chuẩn hoá.
## Chọn K.
Profiling cụm + cảnh báo
vùng rủi ro.
Phân lớp an
toàn/không an toàn
(nếu có nhãn/chuẩn).
F1/PR-AUC + phân
tích lỗi gần ngưỡng.
Giả lập ít nhãn, label
spreading k-NN graph,
learning curve, phân tích
pseudo-label sai ở vùng
ít mẫu.
Dự báo WQI hoặc chỉ số liên
tục.
## MAE/RMSE.
Dự báo các chỉ số chất lượng
nước.
- Dự báo nhu
cầu năng lượng
UCI: Household
## Power
## Consumption
Nếu có thiết bị: basket
thiết bị dùng cùng.
Nếu không: rời rạc hoá
trạng thái (peak/off-peak)
rồi Apriori tìm đồng xuất
hiện điều kiện.
Phân cụm hộ theo profile
tiêu thụ.
Chuẩn hoá.
Phân cụm và Profiling
## (night-owl,
peak-heavy...).
Không áp dụng học
bán giám sát
F1 + phân tích lỗi theo
mùa.
Không áp dụng hồi quy
Nhánh thay thế: anomaly
mining (phát hiện ngày
bất thường) kèm đánh
giá.
Dự báo nhu cầu năng lượng
Split theo thời gian.
Baseline seasonal-naive.
ARIMA/ETS/Holt-Winters.
MAE/RMSE/sMAPE +
residual + outlier.

Đề tàiLink datasetLuật kết hợpPhân cụmPhân lớpBán giám sátHồi quy / Chuỗi thời gian
- Phân tích
đánh giá khách
sạn & chủ đề
dịch vụ
Kaggle:Hotel
reviews datasets
Phân tích luật kết hợp
trên từ khoá/khía cạnh
(aspect) sau khi rời rạc
hoá.
Top luật dịch vụ đi kèm
phàn nàn/khen.
Vector hoá.
Phân cụm chủ đề (topic
clusters).
Đặt tên cụm + review đại
diện.
Phân lớp
sentiment/aspect.
F1-macro + phân tích
lỗi review đa chủ đề.
Thiếu nhãn
aspect/sentiment theo
domain.
Giả lập ít nhãn, label
spreading/self-training,
learning curve, phân tích
pseudo-label sai ở review
ngắn.
Hồi quy rating.
## MAE/RMSE.
Không yêu cầu áp dụng chuỗi
thời gian
- Dự đoán huỷ
đặt phòng
Kaggle:Hotel
## Booking Demand
dataset
Rời rạc hoá lead
time/channel/country.
Luật kết hợp tìm combo
thuộc tính liên quan huỷ.
So sánh theo mùa/quốc
gia.
Phân cụm booking theo
hành vi.
Chuẩn hoá.
Profiling cụm rủi ro huỷ
cao.
Phân lớp huỷ/không
huỷ.
## Imbalance.
## PR-AUC/F1.
Kiểm tra leakage
(thông tin sau đặt
phòng).
Giả lập ít nhãn,
self-training ngưỡng cao,
learning curve, phân tích
pseudo-label sai theo
lead time dài.
Không yêu cầu áp dụng hồi
quy.
## MAE/RMSE.
Chuỗi thời gian dự báo
cancellation rate theo tháng.
- Dự đoán trả
hàng TMĐT
## Kaggle:
## E-commerce
returns datasets
Rời rạc hoá cate-
gory/shipping/promo.
Luật kết hợp tìm combo
liên quan trả hàng.
Top luật + insight chính
sách đổi trả.
Phân cụm khách/sản
phẩm theo return rate.
Chuẩn hoá.
Profiling cụm rủi ro trả
cao.
Phân lớp
return/no-return.
## Imbalance.
## PR-AUC.
Phân tích lỗi theo
category.
Nhãn trả hàng có thể
thiếu/đến muộn.
Giả lập ít nhãn,
self-training, learning
curve, phân tích
pseudo-label sai do
promo/season.
Không yêu cầu áp dụng hồi
quy.
Chuỗi thời gian dự báo return
rate theo tuần/tháng.
- Dự báo
giá/biến động
crypto
## Kaggle:
## Cryptocurrency
## Historical Prices
dataset
Rời rạc hoá trạng thái
(up/down/high-vol bins).
Luật kết hợp tìm coin
biến động cùng.
Top luật + so sánh theo
regime.
Phân cụm coin theo
volatility/return/indica-
tors.
Chuẩn hoá.
Profiling cụm
## (majors/high-risk...).
Tuỳ chọn: phân lớp
trend up/down.
ROC-AUC/F1 + phân
tích lỗi khi tin tức sốc.
Không yêu cầu áp dụng
học bán giám sát
Chuỗi thời gian dự báo
price/return.
Split theo thời gian,
walk-forward.
Baseline naive.
## ARIMA/ETS.
MAE/RMSE + residual +
volatility regime.
- Phân nhóm
học tập & dự
đoán trượt môn
UCI: Student
## Performance
Rời rạc hoá hành vi
(activity bins).
Apriori tìm pattern hành
vi đi kèm trượt/đỗ.
Top luật + gợi ý can
thiệp sớm.
Phân cụm sinh viên theo
hành vi.
Chuẩn hoá.
## Chọn K.
Profiling cụm + khuyến
nghị học tập.
Phân lớp pass/fail.
## PR-AUC/F1.
Phân tích lỗi theo tuần
học/nhóm yếu.
Rất phù hợp: nhãn kết
quả chỉ có cuối kỳ, thiếu
nhãn sớm.
Giả lập ít nhãn, label
spreading/self-training,
learning curve, phân tích
pseudo-label sai ở tuần
đầu.
Phù hợp (chuỗi thời gian)với
OULAD (theo tuần).
Dự báo điểm/engagement
theo tuần: split theo thời gian,
baseline, ARIMA/ETS (hoặc
mô hình đơn giản),
MAE/sMAPE.
- Phân tích lỗi
sản xuất & dự
đoán lỗi
## UCI: AI4I 2020
## Predictive
## Maintenance
Rời rạc hoá trạng thái
máy (bin sensor/setting).
Apriori tìm combo điều
kiện liên quan lỗi.
Top luật + insight vận
hành.
Phân cụm máy/chu kỳ
theo hành vi.
Chuẩn hoá.
Profiling cụm + lịch bảo
trì.
Phân lớp lỗi/không lỗi
(hoặc loại lỗi).
## Imbalance.
## PR-AUC/F1.
Phân tích lỗi theo loại
lỗi.
Áp dụng trong trường
hợp nhãn lỗi hiếm/khó
gán.
Giả lập ít nhãn,
self-training ngưỡng cao,
learning curve, phân tích
rủi ro pseudo-label (false
alarm).
Sử dụng mô hình hồi quy dự
đoán biến liên tục nhưTool
wear [min](hoặc tự tạo
RUL/time-to-failuretừ
Machine failure).
## MAE/RMSE.
Dataset không có timestamp,
chỉ cóUIDtheo thứ tự; coi
như time index để
ARIMA/lag-features khi chấp
nhận giả định “UID≈thời
gian”.
Chia train/test theo thứ tự
quan sát (không shuffle),
đánh giá MAE/RMSE.
Table 1: Danh sách đề tài bài tập lớn

Final Project Data miningLê Thị Thùy Trang
4 Rubic chấm điểm
Tiêu chíĐiểm tối đaĐánh giá (Đạt / Trung bình / Chưa đạt)
A. Bài toán + mô tả dữ liệu + data
dictionary
1.0Đạt: Mục tiêu rõ ràng; mô tả nguồn dữ liệu; giải thích
cột/nhãn/target; có data dictionary; nêu rủi ro như mất cân bằng lớp,
thiếu dữ liệu, data leakage (nếu có).
Trung bình: Có mô tả nhưng thiếu 1–2 ý quan trọng (ví dụ thiếu data
dictionary hoặc chưa nói về leakage/imbalance).
Chưa đạt: Mơ hồ; thiếu nguồn dữ liệu; không nêu target/label hoặc
không giải thích dữ liệu.
B. EDA & tiền xử lý1.5Đạt: EDA có ít nhất 3 biểu đồ kèm diễn giải; xử lý
missing/outlier/duplicate; encoding/scaling hợp lý; có thống kê
trước–sau hoặc pipeline hoá.
Trung bình: Có EDA và tiền xử lý nhưng phân tích còn nông; thiếu
kiểm soát tham số hoặc thiếu thống kê trước–sau.
Chưa đạt: EDA sơ sài hoặc chỉ chụp hình; tiền xử lý tuỳ tiện hoặc
sai.
C. Data Mining core (pattern/clus-
ter/anomaly/rule/graph)
2.0Đạt: Có “khai phá tri thức” đúng chất (phân
cụm/pattern/anomaly/rule/graph); trình bày tham số; có đánh giá
(silhouette/DBI/coverage/runtime...); rút insight rõ ràng.
Trung bình: Có mining nhưng hời hợt; thiếu đánh giá hoặc thiếu
diễn giải kết quả.
Chưa đạt: Không có phần mining; chỉ huấn luyện mô hình dự đoán.
D. Mô hình hoá + baseline so sánh
(>= 2 baseline)
2.0Đạt: Có ít nhất 2 baseline và 1 mô hình cải tiến; giải thích lựa chọn;
có so sánh rõ ràng.
Trung bình: Có baseline nhưng chưa rõ vai trò/thiết lập; thiếu so
sánh hoặc mô hình cải tiến chưa thuyết phục.
Chưa đạt: Chỉ 1 mô hình hoặc không có so sánh baseline.
E. Thiết kế thực nghiệm + metric
đúng
1.0Đạt: Split/CV hợp lý; đặt seed; tránh leakage; chọn metric phù hợp
(F1/PR-AUC/ROC-AUC; RMSE/MAE/sMAPE; silhouette/DBI...).
Trung bình: Có thực nghiệm nhưng thiếu kiểm soát (seed/leakage);
metric đúng nhưng giải thích chưa rõ.
Chưa đạt: Thiết kế thực nghiệm sai; metric không phù hợp hoặc
không nêu rõ.
F. Bán giám sát hoặc nhánh thay
thế tương đương
1.0Đạt:Nếu đề tài phân lớp:có kịch bản thiếu nhãn (10–30% labeled),
so sánh supervised-only vs semi-supervised (self-training/label
spreading), có learning curve theo % nhãn và phân tích pseudo-label
sai.
Trung bình: Có triển khai nhưng thiếu một phần (thiếu learning
curve/thiếu phân tích lỗi/thiếu so sánh).
Chưa đạt: Không thực hiện semi-supervised (khi bắt buộc) hoặc
không có nhánh thay thế (khi không áp dụng).
G. Đánh giá, phân tích lỗi & insight
hành động
1.5Đạt: Có phân tích lỗi (confusion matrix/residual); nêu dạng sai phổ
biến; có ít nhất 5 insight “có hành động” (actionable) gắn với kết
quả.
Trung bình: Có insight nhưng chung chung; phân tích lỗi còn nông.
Chưa đạt: Không phân tích lỗi; insight mơ hồ hoặc không có khuyến
nghị.
H. Repo GitHub chuẩn + chạy lại
được (reproducible)
1.0Đạt: Repo đúng cấu trúc; có README, requirements/environment;
có configs/outputs; chạy lại tạo ra kết quả; notebook “sạch” (gọi
code từ src).
Trung bình: Repo tương đối ổn nhưng thiếu 1–2 phần (ví dụ thiếu
script chạy pipeline hoặc thiếu hướng dẫn); vẫn còn nhiều code
trong notebook.
Chưa đạt: Repo lộn xộn; thiếu hướng dẫn; không chạy lại được.
Table 2: Thang đánh giá đề tài bài tập lớn
## 8