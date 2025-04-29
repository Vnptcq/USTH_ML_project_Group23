import os

# Cấu trúc dự án
project_structure = {
    "expense-prediction-project": {
        "data": {
            "raw": [],
            "processed": []
        },
        "notebooks": [
            "eda.ipynb",
            "feature_engineering.ipynb",
            "model_evaluation.ipynb"
        ],
        "src": {
            "__init__.py": None,
            "data_preprocessing.py": """import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

def preprocess_data(file_path):
    # Đọc dữ liệu
    data = pd.read_csv(file_path)
    
    # Tiền xử lý dữ liệu
    data['full_time'] = data['full_time'].astype(int)
    data['part_time'] = data['part_time'].astype(int)
    data['parents'] = data['parents'].astype(int)
    
    # Chia dữ liệu thành X và y
    X = data.drop(columns=['monthly_expense'])
    y = data['monthly_expense']
    
    # Chia dữ liệu thành tập huấn luyện và kiểm tra
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # Chuẩn hóa dữ liệu
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    return X_train_scaled, X_test_scaled, y_train, y_test
""",
            "feature_engineering.py": """def create_features(data):
    data['savings'] = data['income'] - data['total_expense']
    data['spending_ratio'] = data['total_expense'] / data['income']
    return data
""",
            "model.py": """import xgboost as xgb

def create_model():
    model = xgb.XGBRegressor(objective='reg:squarederror', n_estimators=100, max_depth=6, learning_rate=0.1)
    return model
""",
            "train.py": """from src.model import create_model
from src.data_preprocessing import preprocess_data

def train_model(data_file):
    X_train, X_test, y_train, y_test = preprocess_data(data_file)
    model = create_model()
    model.fit(X_train, y_train)
    return model
""",
            "evaluate.py": """from sklearn.metrics import mean_squared_error, r2_score

def evaluate_model(model, X_test, y_test):
    y_pred = model.predict(X_test)
    rmse = mean_squared_error(y_test, y_pred, squared=False)
    r2 = r2_score(y_test, y_pred)
    return rmse, r2
""",
            "predict.py": """def predict_expense(model, new_data):
    prediction = model.predict(new_data)
    return prediction
""",
        },
        "requirements.txt": """xgboost==1.7.1
pandas==1.3.5
scikit-learn==0.24.2
matplotlib==3.4.3
flask==2.0.1
streamlit==1.3.0
""",
        "config.yaml": """model:
  n_estimators: 100
  max_depth: 6
  learning_rate: 0.1
  objective: 'reg:squarederror'

data:
  file_path: 'data/processed/expenses.csv'

training:
  test_size: 0.2
  random_state: 42
""",
        "app.py": """# File này sẽ dùng nếu bạn muốn triển khai ứng dụng web với Flask hoặc Streamlit
""",
        "run.py": """from src.train import train_model
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
""",
        "README.md": """# Dự Án Dự Báo Chi Tiêu
Mô hình dự báo chi tiêu theo hành vi chi tiêu của người dùng. Dự án này sử dụng XGBoost để dự đoán chi tiêu của người dùng dựa trên dữ liệu về thu nhập, chi tiêu và các yếu tố khác.
""",
    }
}

# Hàm tạo cấu trúc thư mục và tệp
def create_structure(base_path, structure):
    for key, value in structure.items():
        path = os.path.join(base_path, key)
        if isinstance(value, dict):
            os.makedirs(path, exist_ok=True)
            create_structure(path, value)
        elif isinstance(value, list):
            for file_name in value:
                file_path = os.path.join(base_path, file_name)
                with open(file_path, 'w') as f:
                    f.write('')
        elif isinstance(value, str):
            file_path = os.path.join(base_path, key)
            with open(file_path, 'w') as f:
                f.write(value)

# Tạo dự án
create_structure(".", project_structure)

print("Cấu trúc dự án đã được tạo thành công!")
