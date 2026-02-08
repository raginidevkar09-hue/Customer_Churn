import joblib
import pandas as pd

MODEL_PATH = "backend/models/churn_model.pkl"

def load_model():
    return joblib.load(MODEL_PATH)

def predict_churn_logic(data):
    model = load_model()

    df = pd.DataFrame([{
        "Age": data.Age,
        "Gender": data.Gender,
        "Balance": data.Balance,
        "Tenure": data.Tenure,
        "CreditScore": data.CreditScore,
        "NumOfProducts": data.NumOfProducts,
        "IsActiveMember": data.IsActiveMember
    }])

    pred = model.predict(df)[0]
    prob = model.predict_proba(df)[0][1]

    return {
        "churn_prediction": int(pred),
        "churn_probability": round(float(prob), 3)
    }
