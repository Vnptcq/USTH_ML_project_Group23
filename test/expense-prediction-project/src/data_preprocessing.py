import pandas as pd
from sklearn.preprocessing import LabelEncoder
import os

def load_data(file_path):
    return pd.read_csv(file_path)

def preprocess_data(df):
    # Chuyển đổi ngày tháng
    df['date'] = pd.to_datetime(df['date'])

    # Rút trích thêm đặc trưng từ ngày
    df['day_of_week'] = df['date'].dt.dayofweek
    df['month'] = df['date'].dt.month

    # Mã hóa categorical features
    categorical_cols = ['category', 'payment_method', 'location', 'emotion']
    encoder = LabelEncoder()
    for col in categorical_cols:
        df[col] = encoder.fit_transform(df[col])

    # Đảm bảo cột boolean là số
    df['essential'] = df['essential'].astype(int)

    return df

def save_preprocessed_data(df, output_path):
    df.to_csv(output_path, index=False)
    print(f"✅ Preprocessed data saved to {output_path}")

if __name__ == "__main__":
    input_path = os.path.join('data', 'data_sample.csv')
    output_path = os.path.join('data', 'processed_data.csv')
    
    df = load_data(input_path)
    df = preprocess_data(df)
    save_preprocessed_data(df, output_path)
