import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os

def plot_expense_by_category(df):
    plt.figure(figsize=(10,6))
    sns.barplot(data=df, x='category', y='amount', estimator=sum)
    plt.title('Tổng chi tiêu theo loại')
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    data_path = os.path.join('data', 'processed_data.csv')
    df = pd.read_csv(data_path)

    plot_expense_by_category(df)
