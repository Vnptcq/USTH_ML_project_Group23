from src.train import train_model
from src.evaluate import evaluate_model
from src.predict import predict_expense
import pandas as pd

# Huấn luyện mô hình
model = train_model('data/processed/expenses.csv')

# Đánh giá mô hình
X_test, y_test = pd.read_csv('data/processed/test_data.csv')
rmse, r2 = evaluate_model(model, X_test, y_test)

print(f"RMSE: {rmse}")
print(f"R^2: {r2}")

# Dự đoán chi tiêu
new_data = pd.read_csv('data/processed/new_user.csv')
prediction = predict_expense(model, new_data)
print(f"Predicted Expense: {prediction}")
