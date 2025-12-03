import os
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns


BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_PATH = os.path.join(BASE_DIR, "data", "Healthcare.csv")
EDA_DIR = os.path.join(BASE_DIR, "reports", "eda_plots")


def ensure_dirs():
    os.makedirs(EDA_DIR, exist_ok=True)


def load_data():
    df = pd.read_csv(DATA_PATH)
    df = df[["Age", "Gender", "Symptoms", "Symptom_Count", "Disease"]]
    df = df.dropna(subset=["Symptoms", "Disease"])
    return df


def plot_disease_counts(df):
    plt.figure(figsize=(10, 5))
    order = df["Disease"].value_counts().index
    sns.countplot(data=df, x="Disease", order=order, palette="Blues_r")
    plt.xticks(rotation=90)
    plt.xlabel("Disease")
    plt.ylabel("Number of patients")
    plt.title("Number of Patients per Disease")
    plt.tight_layout()
    path = os.path.join(EDA_DIR, "disease_counts.png")
    plt.savefig(path, dpi=120)
    plt.close()


def plot_age_distribution(df):
    plt.figure(figsize=(8, 4))
    sns.histplot(data=df, x="Age", bins=20, kde=True, color="#22b8a7")
    plt.xlabel("Age")
    plt.ylabel("Count")
    plt.title("Age Distribution of Patients")
    plt.tight_layout()
    path = os.path.join(EDA_DIR, "age_distribution.png")
    plt.savefig(path, dpi=120)
    plt.close()


def plot_top_symptoms(df, top_n=15):
    # explode symptoms
    s = (
        df["Symptoms"]
        .fillna("")
        .str.split(",")
        .explode()
        .str.strip()
        .str.lower()
    )
    counts = s.value_counts().head(top_n)

    plt.figure(figsize=(8, 5))
    sns.barplot(
        x=counts.values,
        y=counts.index,
        palette="viridis",
        orient="h",
    )
    plt.xlabel("Frequency")
    plt.ylabel("Symptom")
    plt.title(f"Top {top_n} Symptoms")
    plt.tight_layout()
    path = os.path.join(EDA_DIR, "top_symptoms.png")
    plt.savefig(path, dpi=120)
    plt.close()


def run_eda():
    ensure_dirs()
    df = load_data()
    plot_disease_counts(df)
    plot_age_distribution(df)
    plot_top_symptoms(df)
    print(f"Charts saved to: {EDA_DIR}")


if __name__ == "__main__":
    run_eda()
