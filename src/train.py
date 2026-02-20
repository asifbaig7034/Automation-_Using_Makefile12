import pandas as pd
from sklearn.linear_model import LogisticRegression
import joblib

# Load processed data
X_train = pd.read_csv("data/processed/X_train.csv")
y_train = pd.read_csv("data/processed/y_train.csv")

# Train model
model = LogisticRegression()
model.fit(X_train, y_train.values.ravel())

# Save model
joblib.dump(model, "models/model.pkl")

print("Model trained successfully!")