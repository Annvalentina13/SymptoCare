import os
import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline

from src.preprocessing import FullFeatureTransformer


BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_PATH = os.path.join(BASE_DIR, "data", "Healthcare.csv")
MODELS_DIR = os.path.join(BASE_DIR, "models")
REPORTS_DIR = os.path.join(BASE_DIR, "reports")
EDA_PLOTS_DIR = os.path.join(REPORTS_DIR, "eda_plots")
METRICS_PATH = os.path.join(REPORTS_DIR, "metrics.txt")
MODEL_PATH = os.path.join(MODELS_DIR, "disease_model.joblib")


def ensure_dirs():
    os.makedirs(MODELS_DIR, exist_ok=True)
    os.makedirs(REPORTS_DIR, exist_ok=True)
    os.makedirs(EDA_PLOTS_DIR, exist_ok=True)


def load_data():
    df = pd.read_csv(DATA_PATH)

    df = df[["Age", "Gender", "Symptoms", "Symptom_Count", "Disease"]]

    df = df.dropna(subset=["Symptoms", "Disease"])

    df = df.reset_index(drop=True)
    return df


def build_pipeline():
    feature_transformer = FullFeatureTransformer()

    clf = RandomForestClassifier(
        n_estimators=200,
        random_state=42,
        class_weight="balanced"
    )

    model = Pipeline(steps=[
        ("features", feature_transformer),
        ("clf", clf)
    ])

    return model


def train_and_evaluate():
    ensure_dirs()
    df = load_data()

    X = df[["Age", "Gender", "Symptoms", "Symptom_Count"]]
    y = df["Disease"]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, stratify=y, random_state=42
    )

    model = build_pipeline()

    print("Fitting model...")
    model.fit(X_train, y_train)

    print("Evaluating...")
    y_pred = model.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    cm = confusion_matrix(y_test, y_pred)
    report = classification_report(y_test, y_pred)

    print(f"Accuracy: {acc:.4f}")
    print(report)

    with open(METRICS_PATH, "w") as f:
        f.write(f"Accuracy: {acc:.4f}\n\n")
        f.write("Classification report:\n")
        f.write(report)
        f.write("\nConfusion matrix:\n")
        f.write(np.array2string(cm))

    joblib.dump(model, MODEL_PATH)
    print(f"Model saved to: {MODEL_PATH}")


if __name__ == "__main__":
    train_and_evaluate()
