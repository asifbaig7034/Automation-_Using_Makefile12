import pandas as pd
import joblib
from sklearn.metrics import accuracy_score

# Load model and test data
model = joblib.load("models/model.pkl")
X_test = pd.read_csv("data/processed/X_test.csv")
y_test = pd.read_csv("data/processed/y_test.csv")

# Predict
predictions = model.predict(X_test)

accuracy = accuracy_score(y_test, predictions)

print(f"Model Accuracy: {accuracy}")