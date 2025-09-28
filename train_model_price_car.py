import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import r2_score

# ----- Bước 1: Tạo dữ liệu giả lập -----
np.random.seed(42)
n_samples = 200

year = np.random.randint(2000, 2023, n_samples)   # năm sản xuất
km_driven = np.random.randint(5000, 200000, n_samples)  # số km đã đi
engine = np.random.randint(800, 3000, n_samples)  # dung tích động cơ (cc)
seats = np.random.choice([4, 5, 7], n_samples)    # số chỗ ngồi

# Giá xe = công thức giả lập (đã fix lỗi)
price = (
    (2025 - year) * (-15000) +        # xe càng mới thì giá càng cao
    (engine * 200) +                  # động cơ mạnh thì giá cao
    (seats * 5000) +                  # nhiều chỗ ngồi thì đắt hơn
    (-0.05 * km_driven) +             # đi nhiều km thì giá giảm
    np.random.randint(-20000, 20000, n_samples)   # thêm nhiễu
)

# Tạo DataFrame
data = pd.DataFrame({
    "year": year,
    "km_driven": km_driven,
    "engine": engine,
    "seats": seats,
    "price": price
})

print("📊 Dữ liệu mẫu:")
print(data.head())

# ----- Bước 2: Tách dữ liệu -----
X = data.drop(columns=["price"])
y = data["price"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# ----- Bước 3: Train model -----
# Hồi quy tuyến tính
lr = LinearRegression()
lr.fit(X_train, y_train)
y_pred_lr = lr.predict(X_test)

# Random Forest
rf = RandomForestRegressor(n_estimators=100, random_state=42)
rf.fit(X_train, y_train)
y_pred_rf = rf.predict(X_test)

# ----- Bước 4: Đánh giá -----
print("\n🎯 Kết quả đánh giá:")
print("Linear Regression R2:", r2_score(y_test, y_pred_lr))
print("Random Forest R2:", r2_score(y_test, y_pred_rf))
