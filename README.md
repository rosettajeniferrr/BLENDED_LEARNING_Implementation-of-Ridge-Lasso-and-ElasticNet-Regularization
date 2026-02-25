# BLENDED_LEARNING
# Implementation of Ridge, Lasso, and ElasticNet Regularization for Predicting Car Price

## AIM:
To implement Ridge, Lasso, and ElasticNet regularization models using polynomial features and pipelines to predict car price.

## Equipments Required:
1. Hardware – PCs
2. Anaconda – Python 3.7 Installation / Jupyter notebook

## Algorithm
1. Load the encoded car dataset and preprocess it by separating features (X) and target variable (price).
2. Apply StandardScaler to normalize both input features and target values.
3. Split the dataset into training and testing sets using train_test_split.
4. Create Polynomial Regression pipelines with Ridge regression, Lasso regression, and Elastic Net models and train them on the training data.
5. Evaluate each model using MSE, MAE, and R² score, then compare results using bar charts.
 

## Program:
```
/*
Program to implement Ridge, Lasso, and ElasticNet regularization using pipelines.
Developed by: Rosetta Jenifer.C
RegisterNumber: 212225230230
*/
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns 
from sklearn.model_selection import train_test_split
from sklearn.linear_model import Ridge, Lasso, ElasticNet
from sklearn.preprocessing import PolynomialFeatures, StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
data=pd.read_csv("encoded_car_data (1).csv")
data.head()
data = pd.get_dummies(data, drop_first=True)
x=data.drop('price',axis=1)
y=data['price']
scaler = StandardScaler()
x = scaler.fit_transform(x)
y = scaler.fit_transform(y.values.reshape(-1, 1))
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42)
models = {
    "Ridge": Ridge(alpha=1.0),
    "Lasso": Lasso(alpha=1.0),
    "ElasticNet": ElasticNet (alpha=1.0, l1_ratio=0.5)
}
result= {}
for name,model in models.items():
    pipeline = Pipeline([('poly',PolynomialFeatures(degree=2)),
    ('regressor',model)
    ])
pipeline.fit(x_train, y_train)
predictions = pipeline.predict(x_test)
mse = mean_squared_error(y_test, predictions)
mae = mean_absolute_error(y_test, predictions)
r2 = r2_score(y_test, predictions)
result[name] = {'MSE': mse, 'MAE': mae, 'R² Score': r2}
print('Name: Rosetta Jenifer.C')
print('Reg. No: 212225230230')
for model_name, metrics in result.items():
    print (f"{model_name} - Mean Squared Error: {metrics['MSE']:.2f}, Mean Absolute Error: {metrics['MAE']:.2f}, R² Score: {metrics['R² Score']:.2f}")
results_df = pd.DataFrame(result).T
results_df.reset_index(inplace=True)
results_df.rename(columns={'index': 'Model'}, inplace=True)
plt.figure(figsize=(12, 5))
plt.subplot(1, 2, 1)
sns.barplot(x='Model', y='MSE', data=results_df, palette='viridis')
plt.title('Mean Squared Error (MSE)')
plt.ylabel('MSE')
plt.xticks(rotation=45)
plt.subplot(1, 2, 2)
sns.barplot(x='Model', y='R² Score', data=results_df, palette='viridis')
plt.title('R2 Score')
plt.ylabel('R2 Score')
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()
plt.subplot(1, 2, 1)
sns.barplot(x='Model', y='MAE', data=results_df, palette='viridis')
plt.title('Mean Absolute Error (MAE)')
plt.ylabel('MAE')
plt.xticks(rotation=45)
```

## Output:
<img width="451" height="162" alt="image" src="https://github.com/user-attachments/assets/b99e44e4-562b-4746-8d54-aeeb55663bde" />
<img width="907" height="92" alt="image" src="https://github.com/user-attachments/assets/a19a15ba-c63c-45c6-a6bc-1d9c3c1d6b1d" />
<img width="556" height="98" alt="image" src="https://github.com/user-attachments/assets/75ca19c1-e3c1-4b89-b0e6-1fb784d6e386" />
<img width="838" height="746" alt="image" src="https://github.com/user-attachments/assets/737d4c1e-3481-43fa-8160-da7b3454d319" />
<img width="653" height="723" alt="image" src="https://github.com/user-attachments/assets/522b0e72-8631-41bf-b2cf-d4e5e01cb1c5" />
<img width="647" height="737" alt="image" src="https://github.com/user-attachments/assets/736cc9be-e759-406e-8e4b-2616a6740e8f" />


## Result:
Thus, Ridge, Lasso, and ElasticNet regularization models were implemented successfully to predict the car price and the model's performance was evaluated using R² score and Mean Squared Error.
