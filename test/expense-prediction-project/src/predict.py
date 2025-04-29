import pandas as pd
import joblib
import os

def predict_single(model, input_features):
    prediction = model.predict([input_features])
    return prediction[0]

if __name__ == "__main__":
    model_path = os.path.join('models', 'xgboost_expense_model.pkl')
    model = joblib.load(model_path)

    # Ví dụ input (phải đúng thứ tự và định dạng features sau preprocessing)
    example_input = [2, 0, 0, 1, 3, 4, 1, 2]  # Giả sử một giao dịch mới

    predicted_amount = predict_single(model, example_input)
    print(f"✅ Predicted expense: {predicted_amount:.2f} VND")
