import pandas as pd
import joblib

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression
from sklearn.metrics import (
    accuracy_score,
    mean_squared_error,
    r2_score
)

print("Loading dataset...")

df = pd.read_csv("diabetes.csv")

print(df.head())

# Features and target
X = df.drop("Outcome", axis=1)
y = df["Outcome"]

# Split
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42
)

# Polynomial transform
poly = PolynomialFeatures(
    degree=2
)

X_train_poly = poly.fit_transform(X_train)
X_test_poly = poly.transform(X_test)

# Model
model = LinearRegression()

model.fit(
    X_train_poly,
    y_train
)

pred = model.predict(
    X_test_poly
)

# Convert regression output to 0/1
pred_binary = (pred > 0.5).astype(int)

acc = accuracy_score(
    y_test,
    pred_binary
)

mse = mean_squared_error(
    y_test,
    pred
)

r2 = r2_score(
    y_test,
    pred
)

print("\nResults")
print("Accuracy:", acc)
print("MSE:", mse)
print("R2:", r2)

joblib.dump(
    model,
    "model.pkl"
)

joblib.dump(
    poly,
    "poly.pkl"
)

print("Saved successfully")