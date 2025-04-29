import pandas as pd
import xgboost as xgb
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
import joblib
import os

def train_model(df):
    X = df.drop(['amount', 'date', 'notes'], axis=1)
    y = df['amount']

    X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)

    model = xgb.XGBRegressor(
        n_estimators=100,
        learning_rate=0.1,
        max_depth=5,
        random_state=42
    )
    model.fit(X_train, y_train)

    # Đánh giá model
    preds = model.predict(X_val)
    rmse = mean_squared_error(y_val, preds, squared=False)
    print(f"✅ Validation RMSE: {rmse:.2f}")

    return model

if __name__ == "__main__":
    input_path = os.path.join('data', 'processed_data.csv')
    model_output_path = os.path.join('models', 'xgboost_expense_model.pkl')

    df = pd.read_csv(input_path)
    model = train_model(df)

    # Lưu model
    os.makedirs('models', exist_ok=True)
    joblib.dump(model, model_output_path)
    print(f"✅ Model saved to {model_output_path}")
