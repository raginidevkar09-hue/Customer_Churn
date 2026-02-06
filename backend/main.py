from fastapi import FastAPI, UploadFile, File
from backend.schemas import ChurnInput
from backend.model_utils import predict_churn
from fastapi import UploadFile, File
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
import joblib
import os
import pandas as pd

app = FastAPI(
    title="Customer Churn Prediction API",
    description="Industry-grade ML API using FastAPI",
    version="1.0"
)

@app.get("/")
def health_check():
    return {"status": "API is running"}

@app.post("/predict")
def predict(data: ChurnInput):
    result = predict_churn(data.dict())
    return result

@app.post("/train")
def train_model(file: UploadFile = File(...)):
    df = pd.read_csv(file.file)

    # Drop unnecessary columns
    df = df.drop(columns=["RowNumber", "CustomerId", "Surname"])

    X = df.drop("Exited", axis=1)
    y = df["Exited"]

    # Encode categorical
    le_geo = LabelEncoder()
    le_gender = LabelEncoder()
    X["Geography"] = le_geo.fit_transform(X["Geography"])
    X["Gender"] = le_gender.fit_transform(X["Gender"])

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    X_train, X_test, y_train, y_test = train_test_split(
        X_scaled, y, test_size=0.2, random_state=42
    )

    model = RandomForestClassifier(n_estimators=200, random_state=42)
    model.fit(X_train, y_train)

    preds = model.predict(X_test)
    acc = accuracy_score(y_test, preds)

    os.makedirs("models", exist_ok=True)

    joblib.dump(model, "models/churn_model.pkl")
    joblib.dump(scaler, "models/scaler.pkl")
    joblib.dump(le_geo, "models/le_geography.pkl")
    joblib.dump(le_gender, "models/le_gender.pkl")

    return {
        "message": "Model trained successfully",
        "accuracy": round(acc, 4)
    }

@app.post("/test")
def test_model(file: UploadFile = File(...)):
    df = pd.read_csv(file.file)

    df = df.drop(columns=["RowNumber", "CustomerId", "Surname"])

    X = df.drop("Exited", axis=1)
    y = df["Exited"]

    model = joblib.load("models/churn_model.pkl")
    scaler = joblib.load("models/scaler.pkl")
    le_geo = joblib.load("models/le_geography.pkl")
    le_gender = joblib.load("models/le_gender.pkl")

    X["Geography"] = le_geo.transform(X["Geography"])
    X["Gender"] = le_gender.transform(X["Gender"])

    X_scaled = scaler.transform(X)

    preds = model.predict(X_scaled)
    acc = accuracy_score(y, preds)

    return {
        "test_accuracy": round(acc, 4),
        "total_records": len(df),
        "predicted_churn": int(preds.sum())
    }
