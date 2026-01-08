from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score

def train_and_evaluate(x, y, test_size=0.2, random_state=42):

    x_train, x_val, y_train, y_val = train_test_split(x, y, test_size=test_size, random_state=random_state, stratify=y)

    model = LogisticRegression(
        solver="saga",
        max_iter=1000,
        class_weight="balanced",
        n_jobs=1
    )

    model.fit(x_train, y_train)

    val_probs = model.predict_proba(x_val)[:, 1]
    roc_auc = roc_auc_score(y_val, val_probs)

    print(f"Validation ROC-AUC: {roc_auc:.4f}")

    return model
