import pandas as pd
from sklearn.preprocessing import LabelEncoder, StandardScaler

def preprocess_data(df: pd.DataFrame, training=True, encoders=None, scaler=None):
    df = df.copy()

    # Drop ID column
    if "employee_id" in df.columns:
        df.drop("employee_id", axis=1, inplace=True)

    # Feature Engineering
    df["experience_ratio"] = df["total_experience"] / df["age"]
    df["stability_score"] = df["years_at_company"] / (df["total_experience"] + 1)
    df["growth_index"] = df["training_hours_last_year"] + df["certifications_count"]

    # Categorical columns
    categorical_cols = [
        "gender",
        "department",
        "job_role",
        "education_level"
    ]

    if training:
        encoders = {}
        for col in categorical_cols:
            le = LabelEncoder()
            df[col] = df[col].astype(str)
            df[col] = le.fit_transform(df[col])
            encoders[col] = le
    else:
        for col in categorical_cols:
            df[col] = df[col].astype(str)
            df[col] = encoders[col].transform(df[col])

    # Fill missing values
    df.fillna(df.mean(numeric_only=True), inplace=True)

    # Scale numerical data
    if training:
        scaler = StandardScaler()
        df[df.columns] = scaler.fit_transform(df[df.columns])
    else:
        df[df.columns] = scaler.transform(df[df.columns])

    return df, encoders, scaler
