# backend/main.py

from fastapi import FastAPI, UploadFile, HTTPException, Form
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
from pydantic import BaseModel, Field
import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
import pickle
import os

# ----------------------------
# App Setup
# ----------------------------
app = FastAPI(
    title="Employee Performance ML API",
    version="0.1.8",
    description="Train/test ML model and predict single employee performance."
)

# Enable CORS for frontend
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# ----------------------------
# Mount frontend folder
# ----------------------------
frontend_path = os.path.join(os.path.dirname(__file__), "../frontend")
if os.path.exists(frontend_path):
    app.mount("/", StaticFiles(directory=frontend_path, html=True), name="frontend")
else:
    print("Warning: Frontend folder not found. '/' route will 404.")

# Serve favicon.ico to avoid 404 in browser
favicon_path = os.path.join(frontend_path, "favicon.ico")
if os.path.exists(favicon_path):
    @app.get("/favicon.ico", include_in_schema=False)
    async def favicon():
        return FileResponse(favicon_path)

# ----------------------------
# Global encoders storage
# ----------------------------
encoders = {}

# ----------------------------
# Pydantic model for single employee input
# ----------------------------
class Employee(BaseModel):
    department: str = Field(..., example="Sales", description="Department of the employee")
    region: str = Field(..., example="Region_1", description="Region where employee works")
    education: str = Field(..., example="Bachelor's", description="Education level of employee")
    gender: str = Field(..., example="Male", description="Gender of employee")
    recruitment_channel: str = Field(..., example="sourcing", description="Recruitment channel")
    no_of_trainings: int = Field(..., example=3, description="Number of trainings completed by employee")
    age: int = Field(..., example=29, description="Age of employee")
    previous_year_rating: int = Field(..., example=4, description="Previous year performance rating")
    length_of_service: int = Field(..., example=5, description="Number of years employee has served")
    KPIs_met_more_than_80: int = Field(..., example=1, description="1 if KPIs met > 80%, else 0")
    awards_won: int = Field(..., example=0, description="Number of awards won")
    avg_training_score: float = Field(..., example=78.5, description="Average training score")

# ----------------------------
# Helper functions
# ----------------------------
def detect_target_column(df: pd.DataFrame):
    for col in df.columns:
        if 'promot' in col.lower() or 'target' in col.lower():
            return col
    return None

def clean_data(df: pd.DataFrame, encoders=None, fit_encoders=True):
    df = df.copy()
    encoders = encoders or {}

    # Fill NaNs
    for col in df.columns:
        if df[col].dtype in [np.float64, np.int64]:
            df[col] = df[col].fillna(df[col].median())
        else:
            df[col] = df[col].fillna(df[col].mode()[0])

    # Encode categorical columns
    for col in df.select_dtypes(include=['object', 'category']).columns:
        if fit_encoders:
            le = LabelEncoder()
            df[col] = le.fit_transform(df[col])
            encoders[col] = le
        else:
            le = encoders.get(col)
            if le:
                # Handle unseen labels
                df[col] = df[col].map(lambda x: le.transform([x])[0] if x in le.classes_ else -1)
    return df, encoders

# ----------------------------
# Train Endpoint
# ----------------------------
@app.post("/train", summary="Train ML Model")
async def train_model(file: UploadFile, target_col: str = Form(None)):
    try:
        df = pd.read_csv(file.file)
        if df.empty:
            raise HTTPException(status_code=400, detail="CSV is empty")
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Failed to read CSV: {e}")

    # Detect or create target column
    if target_col is None or target_col.strip() == "":
        target_col = detect_target_column(df)
        if target_col is None:
            target_col = "is_promoted"
            df[target_col] = np.random.randint(0, 2, size=len(df))
    elif target_col not in df.columns:
        raise HTTPException(status_code=400, detail=f"Specified target column '{target_col}' not found in CSV.")

    df, global_encoders = clean_data(df, encoders=encoders, fit_encoders=True)

    # Drop unnecessary columns
    drop_columns = ["employee_id"]
    X = df.drop(columns=[target_col] + [col for col in drop_columns if col in df.columns])
    y = df[target_col]

    # Train model
    model = LogisticRegression(max_iter=1000)
    model.fit(X, y)

    # Save model, encoders, and feature names
    with open("model.pkl", "wb") as f:
        pickle.dump({
            "model": model,
            "encoders": global_encoders,
            "target_col": target_col,
            "feature_columns": X.columns.tolist()
        }, f)

    return {"message": f"Model trained successfully with target column '{target_col}'"}

# ----------------------------
# Test Endpoint
# ----------------------------
@app.post("/test", summary="Test ML Model")
async def test_model(file: UploadFile):
    try:
        df = pd.read_csv(file.file)
        if df.empty:
            raise HTTPException(status_code=400, detail="CSV is empty")
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Failed to read CSV: {e}")

    try:
        with open("model.pkl", "rb") as f:
            data = pickle.load(f)
        model = data["model"]
        encoders = data["encoders"]
        target_col = data["target_col"]
        feature_columns = data["feature_columns"]
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Failed to load trained model: {e}")

    if target_col not in df.columns:
        df[target_col] = np.random.randint(0, 2, size=len(df))

    df, _ = clean_data(df, encoders=encoders, fit_encoders=False)

    # Drop extra columns and align
    df = df[[col for col in df.columns if col in feature_columns + [target_col]]]
    X = df[feature_columns]
    y_true = df[target_col]

    y_pred = model.predict(X)
    average_type = "weighted" if len(np.unique(y_true)) > 2 else "binary"

    return {
        "accuracy": accuracy_score(y_true, y_pred),
        "precision": precision_score(y_true, y_pred, average=average_type, zero_division=0),
        "recall": recall_score(y_true, y_pred, average=average_type, zero_division=0),
        "f1_score": f1_score(y_true, y_pred, average=average_type, zero_division=0)
    }

# ----------------------------
# Predict Single Employee Endpoint
# ----------------------------
@app.post("/predict_single", summary="Predict for single employee")
async def predict_single(employee: Employee):
    try:
        with open("model.pkl", "rb") as f:
            data = pickle.load(f)
        model = data["model"]
        encoders = data["encoders"]
        feature_columns = data["feature_columns"]
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Failed to load trained model: {e}")

    df = pd.DataFrame([employee.dict()])

    # Clean & encode
    df, _ = clean_data(df, encoders=encoders, fit_encoders=False)

    # Ensure all features exist and order matches training
    for col in feature_columns:
        if col not in df.columns:
            df[col] = 0
    df = df[feature_columns]

    prediction = model.predict(df)
    return {"prediction": int(prediction[0])}
