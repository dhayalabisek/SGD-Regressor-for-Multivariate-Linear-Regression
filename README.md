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

# Dataset
data = {
    "Area": [500, 700, 900, 1100, 1300, 1500, 1700, 1900],
    "Price": [100000, 140000, 180000, 220000, 260000, 300000, 340000, 380000],
    "Occupants": [2, 3, 3, 4, 5, 5, 6, 7]
}

df = pd.DataFrame(data)

# Input
X = df[["Area"]]

# Scale X
x_scaler = StandardScaler()
X_scaled = x_scaler.fit_transform(X)

# ---------- PRICE MODEL ----------
price_scaler = StandardScaler()
Y_price = price_scaler.fit_transform(df[["Price"]]).ravel()

price_model = SGDRegressor(
    max_iter=10000,
    learning_rate="constant",
    eta0=0.01,
    random_state=42
)

price_model.fit(X_scaled, Y_price)

# ---------- OCCUPANTS MODEL ----------
occupant_scaler = StandardScaler()
Y_occupants = occupant_scaler.fit_transform(
    df[["Occupants"]]
).ravel()

occupant_model = SGDRegressor(
    max_iter=10000,
    learning_rate="constant",
    eta0=0.01,
    random_state=42
)

occupant_model.fit(X_scaled, Y_occupants)

# New house area
new_area = pd.DataFrame({"Area": [1200]})
new_area_scaled = x_scaler.transform(new_area)

# Predict price
price_scaled = price_model.predict(new_area_scaled)
predicted_price = price_scaler.inverse_transform(
    price_scaled.reshape(-1, 1)
)[0][0]

# Predict occupants
occupants_scaled = occupant_model.predict(new_area_scaled)
predicted_occupants = occupant_scaler.inverse_transform(
    occupants_scaled.reshape(-1, 1)
)[0][0]

# Display results
print("Predicted House Price:", round(predicted_price, 2))
print("Number of Occupants:", max(1, round(predicted_occupants)))
```

## Output:
```
Predicted House Price: 240029.82
Number of Occupants: 4
```


## Result:
Thus the program to implement the multivariate linear regression model for predicting the price of the house and number of occupants in the house with SGD regressor is written and verified using python programming.
