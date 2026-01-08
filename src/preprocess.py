from sklearn.feature_extraction import FeatureHasher
from scipy.sparse import hstack
import pandas as pd

def hash_categorical_features(x: pd.DataFrame, categorical_features, n_features=2**18):

    hasher = FeatureHasher(
        n_features=n_features,
        input_type="string"
    )

    # convert each row to list of "feature=value" strings
    cat_data = (
        x[categorical_features]
        .astype(str)
        .apply(lambda row: [f"{col}={row[col]}" for col in categorical_features], axis=1)
    )

    x_hashed = hasher.transform(cat_data)
    return x_hashed

def preprocess_features(x: pd.DataFrame, categorical_features: list[str], numeric_features, n_features=2**18):

    x_cat = hash_categorical_features(x, categorical_features, n_features)

    x_num = x[numeric_features].values

    x_final = hstack([x_cat, x_num])

    return x_final