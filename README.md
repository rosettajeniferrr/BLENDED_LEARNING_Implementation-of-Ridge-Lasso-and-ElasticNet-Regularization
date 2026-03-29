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
print('Name: Rosetta Jenifer C')
print('Reg. No: 212225230230')
for model_name, metrics in result.items():
    print (f"{model_name} - Mean Squared Error: {metrics['MSE']:.2f}, Mean Absolute Error: {metrics['MAE']:.2f}, R² Score: {metrics['R² Score']:.2f}")4
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
```

## Output:
<img width="1512" height="321" alt="image" src="https://github.com/user-attachments/assets/b3600195-a77b-49b4-a13c-c38338f751c1" />
<img width="960" height="125" alt="image" src="https://github.com/user-attachments/assets/b719aa26-012f-4fe0-8868-ebbc941e648f" />
<img width="1436" height="587" alt="image" src="https://github.com/user-attachments/assets/03a574b2-b80b-4ddc-bdf6-722803615f45" />




## Result:
Thus, Ridge, Lasso, and ElasticNet regularization models were implemented successfully to predict the car price and the model's performance was evaluated using R² score and Mean Squared Error.
