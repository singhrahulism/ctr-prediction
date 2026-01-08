import pandas as pd

def extract_hour_of_day(df: pd.DataFrame) -> pd.Series:
    return df["hour"].astype(str).str[-2:].astype(int)


def engineer_features(df: pd.DataFrame):
    """
    Feature engineering for both training and inference.
    If 'click' is present, it is returned as y.
    """
    df = df.copy()

    # Extract hour_of_day
    df["hour_of_day"] = extract_hour_of_day(df)

    # Drop unused columns
    drop_cols = ["id", "hour"]
    df.drop(columns=[c for c in drop_cols if c in df.columns], inplace=True)

    # Separate target if present
    if "click" in df.columns:
        y = df["click"]
        x = df.drop(columns=["click"])
    else:
        y = None
        x = df

    numeric_features = ["hour_of_day"]
    categorical_features = [c for c in x.columns if c not in numeric_features]

    return x, y, numeric_features, categorical_features
