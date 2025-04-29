import pandas as pd

def load_data(path='data/data_sample.csv'):
    data = pd.read_csv(path)
    return data

if __name__ == "__main__":
    df = load_data()
    print(df.head())
