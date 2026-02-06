from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
import joblib
import os

MODEL_PATH = "models/employee_performance_model.pkl"
ENCODER_PATH = "models/encoders.pkl"
SCALER_PATH = "models/scaler.pkl"

def train_model(X, y):
    model = RandomForestClassifier(
        n_estimators=200,
        max_depth=10,
        random_state=42
    )
    model.fit(X, y)
    return model

def save_model(model, encoders, scaler):
    os.makedirs("models", exist_ok=True)
    joblib.dump(model, MODEL_PATH)
    joblib.dump(encoders, ENCODER_PATH)
    joblib.dump(scaler, SCALER_PATH)

def load_model():
    model = joblib.load(MODEL_PATH)
    encoders = joblib.load(ENCODER_PATH)
    scaler = joblib.load(SCALER_PATH)
    return model, encoders, scaler

def evaluate_model(model, X_test, y_test):
    preds = model.predict(X_test)
    return {
        "accuracy": accuracy_score(y_test, preds),
        "precision": precision_score(y_test, preds, average="weighted"),
        "recall": recall_score(y_test, preds, average="weighted"),
        "f1_score": f1_score(y_test, preds, average="weighted")
    }

