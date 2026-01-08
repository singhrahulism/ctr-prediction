import pandas as pd
from src.feature_engineering import engineer_features
from src.preprocess import preprocess_features
from src.save_load import load_artifacts

def predict_click_probability(df, artifacts_path):
    artifacts = load_artifacts(artifacts_path)

    model = artifacts["model"]
    metadata = artifacts["metadata"]

    x, _, _, _ = engineer_features(df)

    x_proc = preprocess_features(
        x,
        metadata["categorical_features"],
        metadata["numeric_features"],
        metadata["hashing_dim"]
    )

    probs = model.predict_proba(x_proc)[:, 1]
    return probs
