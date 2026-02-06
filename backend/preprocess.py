import pandas as pd
from sklearn.preprocessing import LabelEncoder, StandardScaler

def preprocess_data(df, is_training=True, scaler=None):
    df = df.copy()

    # Drop unnecessary columns
    drop_cols = ["CustomerID", "Churn Reason"]
    df.drop(columns=[c for c in drop_cols if c in df.columns], inplace=True)

    # Target
    if is_training:
        y = df["Churn"]
        df.drop("Churn", axis=1, inplace=True)
    else:
        y = None

    # Encode categorical columns
    for col in df.select_dtypes(include="object").columns:
        df[col] = LabelEncoder().fit_transform(df[col].astype(str))

    # Scaling
    if is_training:
        scaler = StandardScaler()
        X = scaler.fit_transform(df)
        return X, y, scaler
    else:
        X = scaler.transform(df)
        return X
