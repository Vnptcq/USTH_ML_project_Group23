from fastapi import FastAPI
from pydantic import BaseModel
import joblib
import os
import numpy as np

# Khởi tạo app FastAPI
app = FastAPI()

# Load mô hình đã train
model_path = os.path.join('models', 'xgboost_expense_model.pkl')
model = joblib.load(model_path)

# Định nghĩa input schema
class ExpenseInput(BaseModel):
    category: int
    payment_method: int
    location: int
    essential: int
    emotion: int
    day_of_week: int
    month: int
    dummy_feature: int = 0  # Trong trường hợp thiếu feature, tạm set default

@app.get("/")
def read_root():
    return {"message": "Welcome to the Expense Prediction API 🚀"}

@app.post("/predict")
def predict_expense(input_data: ExpenseInput):
    input_array = np.array([
        input_data.category,
        input_data.payment_method,
        input_data.location,
        input_data.essential,
        input_data.emotion,
        input_data.day_of_week,
        input_data.month,
        input_data.dummy_feature
    ]).reshape(1, -1)

    prediction = model.predict(input_array)[0]
    return {
        "predicted_expense": round(prediction, 2)
    }
