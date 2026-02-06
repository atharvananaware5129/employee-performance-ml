import pandas as pd
import pickle
from sklearn.preprocessing import LabelEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

# Load training data
df = pd.read_csv("train.csv")

# Target column
TARGET = "KPIs_met_more_than_80"

# Separate features and target
X = df.drop(columns=[TARGET])
y = df[TARGET]

# Identify column types properly
categorical_cols = X.select_dtypes(include=["object"]).columns
numeric_cols = X.select_dtypes(exclude=["object"]).columns

# Handle missing values safely
for col in categorical_cols:
    X[col] = X[col].fillna(X[col].mode()[0])

for col in numeric_cols:
    X[col] = X[col].fillna(X[col].median())

# Encode categorical features
label_encoders = {}
for col in categorical_cols:
    le = LabelEncoder()
    X[col] = le.fit_transform(X[col])
    label_encoders[col] = le

# Train model
model = LogisticRegression(max_iter=1000)
model.fit(X, y)

# Training accuracy
preds = model.predict(X)
accuracy = accuracy_score(y, preds)
print(f"Training Accuracy: {accuracy:.2f}")

# Save model and encoders
with open("models/employee_model.pkl", "wb") as f:
    pickle.dump(model, f)

with open("models/label_encoders.pkl", "wb") as f:
    pickle.dump(label_encoders, f)

print("✅ Model and encoders saved successfully")


