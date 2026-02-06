import joblib
import pandas as pd

MODEL_PATH = "models/churn_model.pkl"
SCALER_PATH = "models/scaler.pkl"
GEO_PATH = "models/le_geography.pkl"
GENDER_PATH = "models/le_gender.pkl"

model = joblib.load(MODEL_PATH)
scaler = joblib.load(SCALER_PATH)
le_geo = joblib.load(GEO_PATH)
le_gender = joblib.load(GENDER_PATH)

FEATURE_ORDER = [
    "CreditScore", "Geography", "Gender", "Age", "Tenure",
    "Balance", "NumOfProducts", "HasCrCard",
    "IsActiveMember", "EstimatedSalary"
]

def preprocess_input(data: dict):
    df = pd.DataFrame([data])

    df["Geography"] = le_geo.transform(df["Geography"])
    df["Gender"] = le_gender.transform(df["Gender"])

    df = df[FEATURE_ORDER]
    scaled = scaler.transform(df)

    return scaled

def predict_churn(data: dict):
    processed = preprocess_input(data)
    prediction = model.predict(processed)[0]
    probability = model.predict_proba(processed)[0][1]

    return {
        "churn_prediction": int(prediction),
        "churn_probability": round(float(probability), 3)
    }
