# SGD-Regressor-for-Multivariate-Linear-Regression

## AIM:
To write a program to predict the price of the house and number of occupants in the house with SGD regressor.

## Equipments Required:
1. Hardware – PCs
2. Anaconda – Python 3.7 Installation / Jupyter notebook

## Algorithm
1.Import the dataset and separate the number of occupants and house price.
2.Create an SGDRegressor model and train it using the data.
3.Give the number of occupants as input and predict the house price.
4.Display the predicted house price.

## Program:
```
/*
Program to implement the multivariate linear regression model for predicting the price of the house and number of occupants in the house with SGD regressor.
Developed by: DHAYAL ABISEK R
RegisterNumber:  212225060061
*/
```
```
import pandas as pd
from sklearn.linear_model import SGDRegressor
from sklearn.preprocessing import StandardScaler

# Create dataset
data = {
    "Occupants": [2, 3, 4, 5, 6, 7, 8, 9],
    "Price": [120000, 150000, 180000, 220000, 250000, 280000, 320000, 350000]
}

df = pd.DataFrame(data)

# Input and output
X = df[["Occupants"]]
Y = df["Price"]

# Scale input
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Create SGD model
model = SGDRegressor(
    max_iter=10000,
    learning_rate="constant",
    eta0=0.01,
    tol=1e-3,
    random_state=42
)

# Train
model.fit(X_scaled, Y)

# Predict for 5 occupants
new_data = scaler.transform(pd.DataFrame({"Occupants": [5]}))
prediction = model.predict(new_data)

print("Predicted House Price:", prediction[0])
```

## Output:
```
Predicted House Price: 217144.56127701697
```


## Result:
Thus the program to implement the multivariate linear regression model for predicting the price of the house and number of occupants in the house with SGD regressor is written and verified using python programming.
