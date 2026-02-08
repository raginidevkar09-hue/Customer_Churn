from fastapi import FastAPI, UploadFile, File
import pandas as pd
import joblib
import os

from backend.schemas import ChurnInput
from backend.model_utils import predict_churn_logic

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

app = FastAPI()

MODEL_DIR = "backend/models"
MODEL_PATH = f"{MODEL_DIR}/churn_model.pkl"

os.makedirs(MODEL_DIR, exist_ok=True)

@app.post("/train")
def train_model(file: UploadFile = File(...)):
    df = pd.read_csv(file.file)

    FEATURES = [
        "Age", "Gender", "Balance", "Tenure",
        "CreditScore", "NumOfProducts", "IsActiveMember"
    ]

    X = df[FEATURES]
    y = df["Exited"]

    num_cols = ["Age", "Balance", "Tenure", "CreditScore", "NumOfProducts", "IsActiveMember"]
    cat_cols = ["Gender"]

    preprocessor = ColumnTransformer([
        ("num", "passthrough", num_cols),
        ("cat", OneHotEncoder(drop="first"), cat_cols)
    ])

    model_pipeline = Pipeline([
        ("preprocessor", preprocessor),
        ("model", LogisticRegression(max_iter=1000))
    ])

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    model_pipeline.fit(X_train, y_train)

    acc = accuracy_score(y_test, model_pipeline.predict(X_test))

    joblib.dump(model_pipeline, MODEL_PATH)

    return {
        "message": "Model trained successfully",
        "accuracy": round(acc, 3)
    }

@app.post("/test")
def test_model(file: UploadFile = File(...)):
    import pandas as pd
    import joblib
    from sklearn.metrics import accuracy_score

    # Load trained model
    model = joblib.load("backend/models/churn_model.pkl")

    # Read CSV
    df = pd.read_csv(file.file)

    # Split X and y
    X = df.drop("Exited", axis=1)
    y = df["Exited"]

    # Predict
    predictions = model.predict(X)

    accuracy = accuracy_score(y, predictions)

    return {
        "test_accuracy": round(accuracy, 4),
        "total_records": len(df),
        "predicted_churn": int(predictions.sum())
    }

@app.post("/predict")
def predict_churn(data: ChurnInput):
    return predict_churn_logic(data)
