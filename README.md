# Diabetes Risk Prediction using Machine Learning
## Mô tả dự án
Đây là đồ án chuyên ngành của sinh viên năm 4 ngành Khoa học Dữ liệu – Đại học Nguyễn Tất Thành. Dự án xây dựng hệ thống dự đoán nguy cơ tiểu đường của người dùng dựa trên dữ liệu về **thói quen sinh hoạt** và **yếu tố di truyền** bằng các mô hình học máy (ML), đồng thời triển khai ứng dụng web giúp người dùng tự kiểm tra.
## 🎯 Mục tiêu
- Tiền xử lý và phân tích dữ liệu liên quan đến bệnh tiểu đường
- Áp dụng các mô hình ML như **Logistic Regression**, **Random Forest**, **XGBoost**, **LightGBM**, **VotingClassifier**
- Sử dụng **Ensemble Learning** để tăng độ chính xác dự đoán
- Triển khai mô hình thành ứng dụng web sử dụng Flask
## 📁 Cấu trúc thư mục
```
Diabetes_Prediction/
│
├── app.py                     # Flask web app
├── Model.ipynb               # Notebook xử lý dữ liệu và huấn luyện mô hình
├── models/
│   ├── scaler.pkl            # File lưu MinMaxScaler
│   ├── encoder.pkl           # File lưu OneHotEncoder
│   └── ensemble_model.pkl    # Mô hình VotingClassifier đã train
│
├── static/                   # CSS & ảnh
│
├── templates/
│   └── index.html            # Giao diện web người dùng
│
├── data/
│   └── diabetes_data.csv     # Dữ liệu gốc
│
└── requirements.txt          # Các thư viện cần cài
```
## 🧪 Các mô hình đã sử dụng
- Logistic Regression
- Random Forest (GridSearchCV)
- XGBoost (GridSearchCV)
- LightGBM (GridSearchCV)
- Ensemble (VotingClassifier)
**Độ chính xác cuối cùng (trên test set):** ~90%  
**ROC AUC:** ~0.93
## ⚙️ Cách chạy ứng dụng
```bash
# Cài thư viện
pip install -r requirements.txt
# Chạy Flask app
python app.py
```
Sau đó truy cập `http://127.0.0.1:5000/` để sử dụng ứng dụng dự đoán.

## 📌 Ghi chú

- File mô hình đã được lưu bằng `joblib` trong thư mục `models/`
- Đảm bảo file `scaler.pkl` và `encoder.pkl` khớp với dữ liệu đầu vào
- Web sử dụng Flask thuần + HTML/CSS đơn giản