import os
import csv
from datetime import datetime

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
HISTORY_PATH = os.path.join(BASE_DIR, "data", "prediction_history.csv")


def ensure_history_file():
    os.makedirs(os.path.dirname(HISTORY_PATH), exist_ok=True)
    if not os.path.exists(HISTORY_PATH):
        with open(HISTORY_PATH, "w", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            writer.writerow([
                "timestamp",
                "age",
                "gender",
                "symptoms",
                "symptom_count",
                "predicted_disease",
                "top1_prob",
                "top2_disease",
                "top2_prob",
                "top3_disease",
                "top3_prob",
            ])


def append_prediction(age, gender, symptoms, symptom_count,
                      classes, proba, top_idx):
    ensure_history_file()
    ts = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    row = [
        ts,
        age,
        gender,
        symptoms,
        symptom_count,
        classes[top_idx[0]],
        float(proba[top_idx[0]]),
        classes[top_idx[1]],
        float(proba[top_idx[1]]),
        classes[top_idx[2]],
        float(proba[top_idx[2]]),
    ]
    with open(HISTORY_PATH, "a", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(row)
