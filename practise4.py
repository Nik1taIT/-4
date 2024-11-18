# Частина 1: Дослідницький аналіз даних (EDA)

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.datasets import fetch_california_housing

california = fetch_california_housing()
df = pd.DataFrame(california.data, columns=california.feature_names)
df['target'] = california.target

print(df.describe())

print(df.isnull().sum())

print(df.dtypes)

for column in df.columns:
    plt.figure(figsize=(10, 5))
    df[column].hist(bins=50)
    plt.title(f'Histogram for {column}')
    plt.show()

    plt.figure(figsize=(10, 5))
    plt.boxplot(df[column])
    plt.title(f'Boxplot for {column}')
    plt.show()

corr_matrix = df.corr()
plt.figure(figsize=(12, 10))
sns.heatmap(corr_matrix, annot=True, cmap='coolwarm')
plt.show()

for column in california.feature_names:
    plt.figure(figsize=(10, 5))
    plt.scatter(df[column], df['target'], alpha=0.5)
    plt.title(f'Scatter plot between target and {column}')
    plt.xlabel(column)
    plt.ylabel('Target')
    plt.show()

# Частина 2: Підготовка даних
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import joblib

X = df.drop(columns=['target'])
y = df['target']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)
joblib.dump(scaler, 'scaler.pkl')

# Частина 3: Побудова моделей
from sklearn.linear_model import LinearRegression, Lasso
from sklearn.metrics import mean_squared_error, r2_score

X_train_simple = X_train_scaled[:, 0].reshape(-1, 1)
X_test_simple = X_test_scaled[:, 0].reshape(-1, 1)

simple_model = LinearRegression()
simple_model.fit(X_train_simple, y_train)
y_pred_simple = simple_model.predict(X_test_simple)

plt.figure(figsize=(10, 5))
plt.scatter(X_test_simple, y_test, color='blue', label='Actual')
plt.plot(X_test_simple, y_pred_simple, color='red', label='Predicted')
plt.legend()
plt.xlabel('MedInc')
plt.ylabel('Target')
plt.title('Simple Linear Regression')
plt.show()

print(f'MSE (Simple): {mean_squared_error(y_test, y_pred_simple)}')
print(f'R-squared (Simple): {r2_score(y_test, y_pred_simple)}')

multi_model = LinearRegression()
multi_model.fit(X_train_scaled, y_train)
y_pred_multi = multi_model.predict(X_test_scaled)

coeff_df = pd.DataFrame(multi_model.coef_, X.columns, columns=['Coefficient'])
print(coeff_df)

print(f'MSE (Multiple): {mean_squared_error(y_test, y_pred_multi)}')
print(f'R-squared (Multiple): {r2_score(y_test, y_pred_multi)}')

lasso_model = Lasso(alpha=0.1)
lasso_model.fit(X_train_scaled, y_train)
y_pred_lasso = lasso_model.predict(X_test_scaled)

print(f'MSE (Lasso): {mean_squared_error(y_test, y_pred_lasso)}')
print(f'R-squared (Lasso): {r2_score(y_test, y_pred_lasso)}')

# Частина 4: Оцінка моделей
import math

def print_metrics(y_test, y_pred, model_name):
    mse = mean_squared_error(y_test, y_pred)
    rmse = math.sqrt(mse)
    r2 = r2_score(y_test, y_pred)
    print(f'{model_name} MSE: {mse}')
    print(f'{model_name} RMSE: {rmse}')
    print(f'{model_name} R-squared: {r2}')

print_metrics(y_test, y_pred_simple, 'Simple Linear Regression')
print_metrics(y_test, y_pred_multi, 'Multiple Linear Regression')
print_metrics(y_test, y_pred_lasso, 'Lasso Regression')

plt.figure(figsize=(10, 5))
plt.scatter(y_test, y_pred_multi, alpha=0.5)
plt.xlabel('Actual Values')
plt.ylabel('Predicted Values')
plt.title('Actual vs Predicted (Multiple Regression)')
plt.show()

# Частина 5: Інтерпретація результатів

def predict_price(features, scaler, model):
    features_scaled = scaler.transform([features])
    return model.predict(features_scaled)[0]

features = X.iloc[0].values
predicted_price = predict_price(features, scaler, multi_model)
print(f'Predicted price: {predicted_price}')