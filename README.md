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

# Predict Price
price_model = SGDRegressor(max_iter=10000, tol=1e-3, random_state=42)
price_model.fit(X, df["Price"])

# Predict Occupants
occupant_model = SGDRegressor(max_iter=10000, tol=1e-3, random_state=42)
occupant_model.fit(X, df["Occupants"])

# Input area of new house
new_area = pd.DataFrame({"Area": [1200]})

# Predictions
predicted_price = price_model.predict(new_area)
predicted_occupants = occupant_model.predict(new_area)

print("Predicted House Price:", predicted_price[0])
print("Predicted Number of Occupants:", round(predicted_occupants[0]))
```

## Output:
```
Predicted House Price: 2159832426685048.2
Predicted Number of Occupants: -1045166153011204
```


## Result:
Thus the program to implement the multivariate linear regression model for predicting the price of the house and number of occupants in the house with SGD regressor is written and verified using python programming.
