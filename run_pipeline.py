import pandas as pd

from src.feature_engineering import engineer_features
from src.preprocess import preprocess_features
from src.train import train_and_evaluate
from src.save_load import save_artifacts

def main():
    DATA_PATH = "data/sampled/sample_train.csv"
    MODEL_PATH = "models/ctr_logreg.pkl"
    HASH_DIM = 2**18

    # Load data
    df = pd.read_csv(DATA_PATH)

    # Feature engineering
    X, y, numeric_features, categorical_features = engineer_features(df)

    # Preprocess
    X_proc = preprocess_features(
        X,
        categorical_features,
        numeric_features,
        n_features=HASH_DIM
    )

    # Train
    model = train_and_evaluate(X_proc, y)

    # Save artifacts
    metadata = {
        "categorical_features": categorical_features,
        "numeric_features": numeric_features,
        "hashing_dim": HASH_DIM
    }

    save_artifacts(
        model=model,
        metadata=metadata,
        path=MODEL_PATH
    )

    print("Model and metadata saved.")


if __name__ == "__main__":
    main()
