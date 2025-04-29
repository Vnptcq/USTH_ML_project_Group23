# Import necessary libraries
import matplotlib
matplotlib.use('Agg')  # Use a non-interactive backend to avoid tkinter dependency
import pandas as pd
import numpy as np
from matplotlib import pyplot as plt
from sklearn.model_selection import train_test_split, GridSearchCV, cross_val_score
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OrdinalEncoder, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor  # Import Random Forest Regressor
from sklearn.ensemble import GradientBoostingRegressor  # Import Gradient Boosting Regressor
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
from sklearn.base import BaseEstimator, RegressorMixin  # Import necessary classes for custom regressor
from sklearn.base import TransformerMixin  # Add this line
import warnings  # Add this line to handle warnings
warnings.filterwarnings('ignore', category=FutureWarning)

# Constants
TRAIN_TEST_SPLIT_RATIO = 0.8

# Functions
def calculate_income(row):
    """
    Calculate total income based on work and parental support.
    """
    total_income = 0
    # If working full-time without part-time or parental support
    if row['fulltime'] == 1 and row['parttime'] == 0 and row['parents'] == 0:
        total_income += row['monthly_salary']
    # If working part-time without full-time or parental support
    elif row['fulltime'] == 0 and row['parttime'] == 1 and row['parents'] == 0:
        total_income += row['hourly_rate'] * row['working_hours']
    # If working full-time and receiving parental support
    elif row['fulltime'] == 1 and row['parttime'] == 0 and row['parents'] == 1:
        total_income += row['monthly_salary']
        if not pd.isna(row['monthly_allowance']):
            total_income += row['monthly_allowance']
        else:
            total_income += row['weekly_allowance'] * 4  # Assume 4 weeks in a month
    # If working part-time and receiving parental support
    elif row['fulltime'] == 0 and row['parttime'] == 1 and row['parents'] == 1:
        total_income += row['hourly_rate'] * row['working_hours']
        if not pd.isna(row['monthly_allowance']):
            total_income += row['monthly_allowance']
        else:
            total_income += row['weekly_allowance'] * 4
    # If working both full-time and part-time without parental support
    elif row['fulltime'] == 1 and row['parttime'] == 1 and row['parents'] == 0:
        total_income += row['monthly_salary'] + (row['hourly_rate'] * row['working_hours'])
    # If working both full-time and part-time and receiving parental support
    elif row['fulltime'] == 1 and row['parttime'] == 1 and row['parents'] == 1:
        total_income += row['monthly_salary'] + (row['hourly_rate'] * row['working_hours'])
        if not pd.isna(row['monthly_allowance']):
            total_income += row['monthly_allowance']
        else:
            total_income += row['weekly_allowance'] * 4
    return total_income

def predict_future_expenses(model, future_data):
    """
    Predict future expenses based on the individual's spending habits.
    """
    predictions = model.predict(future_data)
    future_data['predicted_expenses'] = predictions
    return future_data

def analyze_individual_habits(data):
    """
    Analyze individual spending habits and provide personalized insights.
    """
    insights = []
    # Identify the most consistent spending category
    std_dev = data[['rent', 'food', 'transport', 'others']].std()
    most_consistent_category = std_dev.idxmin()
    insights.append(f"The most consistent spending category is '{most_consistent_category}'. This indicates stable spending in this area.")
    # Identify the most volatile spending category
    most_volatile_category = std_dev.idxmax()
    insights.append(f"The most volatile spending category is '{most_volatile_category}'. Consider reviewing spending in this area for better control.")
    # Check if the individual is overspending compared to their income
    overspending_months = data[data['total_expenses'] > data['income']]
    if not overspending_months.empty:
        insights.append(f"You overspent your income in {len(overspending_months)} months. Consider adjusting your budget to avoid financial strain.")
    # Analyze saving habits
    avg_saving_rate = data['saving_rate'].mean()
    if avg_saving_rate < 0.2:
        insights.append("Your average saving rate is below 20%. Try to increase your savings to build financial security.")
    else:
        insights.append(f"Your average saving rate is {avg_saving_rate:.2%}. Keep up the good work!")
    return insights

def generate_additional_insights(data):
    """
    Generate additional insights based on the extended dataset.
    """
    insights = []
    # Analyze the percentage of income spent on rent
    avg_rent_percentage = data['rent_percentage'].mean()
    if avg_rent_percentage > 0.3:
        insights.append("You are spending more than 30% of your income on rent. Consider finding ways to reduce housing costs.")
    else:
        insights.append(f"Your average rent spending is {avg_rent_percentage:.2%}, which is within a healthy range.")
    # Analyze the savings gap
    if data['savings_gap'].mean() < 0:
        insights.append("You are consistently saving less than your target. Review your expenses to close the savings gap.")
    else:
        insights.append("You are meeting or exceeding your savings target. Keep up the good work!")
    return insights

def add_interaction_and_lagged_features(data):
    """
    Add interaction terms and lagged features to the dataset.
    """
    # Interaction terms
    data['rent_income_interaction'] = data['rent_percentage'] * data['income']
    data['food_income_interaction'] = data['food_percentage'] * data['income']

    # Lagged features
    data['lagged_expenses'] = data['total_expenses'].shift(1).fillna(0)
    return data

def augment_data(data, num_samples=100):
    """
    Generate synthetic data to increase variability in the dataset.
    """
    synthetic_data = data.sample(num_samples, replace=True).copy()
    synthetic_data['rent'] *= np.random.uniform(0.9, 1.1, size=num_samples)
    synthetic_data['food'] *= np.random.uniform(0.9, 1.1, size=num_samples)
    synthetic_data['transport'] *= np.random.uniform(0.9, 1.1, size=num_samples)
    synthetic_data['others'] *= np.random.uniform(0.9, 1.1, size=num_samples)
    synthetic_data['total_expenses'] *= np.random.uniform(0.9, 1.1, size=num_samples)
    return pd.concat([data, synthetic_data], ignore_index=True)

# Custom transformer for multi-label features
class MultiLabelEncoder(BaseEstimator, TransformerMixin):
    """
    Custom transformer to encode multi-label features.
    """
    def __init__(self):
        self.encoder = OneHotEncoder(handle_unknown='ignore', sparse_output=False)

    def fit(self, X, y=None):
        self.encoder.fit(X)
        return self

    def transform(self, X):
        return self.encoder.transform(X)

class BoundedRegressor(BaseEstimator, RegressorMixin):
    """
    A wrapper for regressors to bound predictions within a specified range.
    """
    def __init__(self, base_regressor, y_min=None, y_max=None):
        self.base_regressor = base_regressor
        self.y_min = y_min
        self.y_max = y_max

    def fit(self, X, y):
        self.base_regressor.fit(X, y)
        self.y_min = y.min() if self.y_min is None else self.y_min
        self.y_max = y.max() if self.y_max is None else self.y_max
        return self

    def predict(self, X):
        predictions = self.base_regressor.predict(X)
        return np.clip(predictions, self.y_min, self.y_max)

def tune_random_forest(X_train, y_train):
    """
    Perform hyperparameter tuning for RandomForestRegressor using GridSearchCV.
    """
    param_grid = {
        'n_estimators': [50, 100, 200],
        'max_depth': [None, 10, 20, 30],
        'min_samples_split': [2, 5, 10],
        'min_samples_leaf': [1, 2, 4]
    }
    grid_search = GridSearchCV(
        estimator=RandomForestRegressor(random_state=42),
        param_grid=param_grid,
        scoring='neg_mean_absolute_error',
        cv=5,
        n_jobs=-1
    )
    grid_search.fit(X_train, y_train)
    print("\nBest Parameters for RandomForestRegressor:")
    print(grid_search.best_params_)
    return grid_search.best_estimator_

# Load data
data = pd.read_csv('student_expenses.csv')  # Load the dataset containing individual spending data
city_indices = pd.read_csv('expense_of_each_city.csv')  # Load city-specific expense indices

# Data preprocessing
data['income'] = data.apply(calculate_income, axis=1)
data['target_saving'] = data['income'] * data['saving_rate']
data['expected_expenses'] = data['income'] - data['target_saving']
data['total_expenses'] = data[['rent', 'food', 'transport', 'others']].sum(axis=1)
data['year_month'] = pd.to_datetime(data[['Year', 'Month']].assign(day=1))
data['year_month_str'] = data['Year'].astype(str) + '-' + data['Month'].astype(str).str.zfill(2)
data.sort_values('year_month', inplace=True)
data = data.merge(city_indices, left_on='City', right_on='City', how='left')

# Add new features
data['rent_percentage'] = data['rent'] / data['income']
data['food_percentage'] = data['food'] / data['income']
data['transport_percentage'] = data['transport'] / data['income']
data['others_percentage'] = data['others'] / data['income']
data['total_working_hours'] = data['working_hours'] + (data['fulltime'] * 160)
data['savings_gap'] = data['target_saving'] - (data['income'] - data['total_expenses'])

# Add interaction terms and lagged features
data = add_interaction_and_lagged_features(data)

# Augment the dataset with synthetic data
data = augment_data(data)

# Simplify feature set with new features
X = data[['rent', 'food', 'transport', 'others', 'income', 'lagged_expenses']]  # Refined feature set
y = data['total_expenses']  # Define the target variable explicitly

# Split data into training and testing sets
train_size = int(len(data) * TRAIN_TEST_SPLIT_RATIO)
X_train, X_test = X[:train_size], X[train_size:]
y_train, y_test = y[:train_size], y[train_size:]

# Add a constant for model selection
MODEL_TYPE = "RandomForest"  # Options: "LinearRegression", "RandomForest"

# Define the base regressor based on the selected model type
if MODEL_TYPE == "GradientBoosting":
    base_regressor = GradientBoostingRegressor(n_estimators=200, learning_rate=0.1, max_depth=3, random_state=42)
elif MODEL_TYPE == "RandomForest":
    base_regressor = tune_random_forest(X_train, y_train)  # Use tuned RandomForest
elif MODEL_TYPE == "LinearRegression":
    base_regressor = LinearRegression()
else:
    raise ValueError("Invalid MODEL_TYPE. Choose 'LinearRegression', 'RandomForest', or 'GradientBoosting'.")

# Wrap the base regressor with BoundedRegressor
bounded_regressor = BoundedRegressor(base_regressor=base_regressor)

# Define preprocessing steps
num_transformer = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='median')),
    ('scaler', StandardScaler())
])

# Since we removed multi-label features, we only need to handle categorical features
cat_transformer = Pipeline(steps=[
    ('encoder', OneHotEncoder(handle_unknown='ignore'))
])

# Define the preprocessor
preprocessor = ColumnTransformer(transformers=[
    ("num", num_transformer, ['rent', 'food', 'transport', 'others', 'income', 'lagged_expenses']),
    ("cat", cat_transformer, [])  # No categorical features in the simplified feature set
])

# Define the pipeline with preprocessing and the bounded regressor
model = Pipeline(steps=[
    ("preprocessor", preprocessor),
    ("regressor", bounded_regressor)
])

# Train the model
model.fit(X_train, y_train)

# Evaluate the model
y_pred = model.predict(X_test)
mae = mean_absolute_error(y_test, y_pred)
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

# Normalize MAE and MSE
mean_target = y_test.mean()
range_target = y_test.max() - y_test.min()
normalized_mae = mae / mean_target
normalized_mse = mse / (range_target ** 2)

# Print evaluation metrics
print("\nModel Evaluation Metrics:")
print(f"Mean Absolute Error (MAE): {mae:,.2f}")
print(f"Normalized MAE: {normalized_mae:.2%}")
print(f"Mean Squared Error (MSE): {mse:,.2f}")
print(f"Normalized MSE: {normalized_mse:.2%}")
print(f"R-squared (R²): {r2:.2f}")

# Residual Analysis
residuals = y_test - y_pred
plt.figure(figsize=(10, 6))
plt.scatter(y_pred, residuals, alpha=0.6, color='blue')
plt.axhline(y=0, color='red', linestyle='--', linewidth=2)
plt.xlabel("Predicted Values", fontsize=12)
plt.ylabel("Residuals", fontsize=12)
plt.title("Residual Analysis", fontsize=14)
plt.grid(True)
plt.tight_layout()
plt.savefig("residual_analysis.png")

# Generate insights
individual_insights = analyze_individual_habits(data)
print("\nPersonalized Spending Insights:")
for insight in individual_insights:
    print(f"- {insight}")

additional_insights = generate_additional_insights(data)
print("\nAdditional Insights:")
for insight in additional_insights:
    print(f"- {insight}")

# Analyze trends and relationships
print("\nMonthly Trends (Average):")
monthly_trends = data.groupby(['Year', 'Month'])[['total_expenses', 'income', 'target_saving']].mean()
print(monthly_trends.style.format("{:,.2f}").to_string())

print("\nImpact of Saving Rates:")
saving_rate_impact = data.groupby('saving_rate')[['income', 'total_expenses', 'savings_gap']].mean()
print(saving_rate_impact.style.format("{:,.2f}").to_string())

print("\nWorking Hours vs Income and Expenses:")
working_hours_analysis = data.groupby('total_working_hours')[['income', 'total_expenses']].mean()
print(working_hours_analysis.style.format("{:,.2f}").to_string())

# Use time-series cross-validation
from sklearn.model_selection import TimeSeriesSplit

tscv = TimeSeriesSplit(n_splits=5)
cv_scores = cross_val_score(model, X, y, cv=tscv, scoring='neg_mean_absolute_error')
print("\nTime-Series Cross-Validation Results:")
print(f"Mean Absolute Error (MAE): {-np.mean(cv_scores):,.2f} ± {np.std(cv_scores):,.2f}")

# Save insights and analysis results to a file
with open("analysis_results.txt", "w") as file:
    # Write evaluation metrics
    file.write("Model Evaluation Metrics:\n")
    file.write(f"Mean Absolute Error (MAE): {mae:,.2f}\n")
    file.write(f"Mean Squared Error (MSE): {mse:,.2f}\n")
    file.write(f"R-squared (R²): {r2:.2f}\n\n")

    # Write personalized spending insights
    file.write("Personalized Spending Insights:\n")
    for insight in individual_insights:
        file.write(f"- {insight}\n")
    file.write("\n")

    # Write additional insights
    file.write("Additional Insights:\n")
    for insight in additional_insights:
        file.write(f"- {insight}\n")
    file.write("\n")

    # Write monthly trends
    file.write("Monthly Trends (Average):\n")
    file.write(monthly_trends.style.format("{:,.2f}").to_string())
    file.write("\n\n")

    # Write impact of saving rates
    file.write("Impact of Saving Rates:\n")
    file.write(saving_rate_impact.style.format("{:,.2f}").to_string())
    file.write("\n\n")

    # Write working hours analysis
    file.write("Working Hours vs Income and Expenses:\n")
    file.write(working_hours_analysis.style.format("{:,.2f}").to_string())
    file.write("\n")
