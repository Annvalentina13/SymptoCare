from typing import List
import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin


class FullFeatureTransformer(BaseEstimator, TransformerMixin):
    """
    Transform the raw DataFrame with columns:
    Age, Gender, Symptom_Count, Symptoms
    into a single numeric feature matrix.
    """

    def __init__(self):
        self.symptom_vocab_: List[str] = []
        self.gender_to_idx_ = {}

    def fit(self, X: pd.DataFrame, y=None):
        # Build symptom vocabulary
        symptom_set = set()
        for text in X["Symptoms"].fillna(""):
            parts = [s.strip().lower() for s in str(text).split(",") if s.strip()]
            symptom_set.update(parts)
        self.symptom_vocab_ = sorted(symptom_set)

        # Build gender mapping
        genders = X["Gender"].fillna("Other").str.strip().str.title().unique()
        self.gender_to_idx_ = {g: i for i, g in enumerate(sorted(genders))}
        return self

    def transform(self, X: pd.DataFrame) -> np.ndarray:
        X = X.copy()

        # Numeric features
        age = X["Age"].fillna(X["Age"].median()).to_numpy(dtype=float).reshape(-1, 1)
        sc = X["Symptom_Count"].fillna(X["Symptom_Count"].median()).to_numpy(dtype=float).reshape(-1, 1)

        # Gender one-hot
        genders = X["Gender"].fillna("Other").str.strip().str.title()
        n_samples = len(X)
        n_genders = len(self.gender_to_idx_)
        gender_oh = np.zeros((n_samples, n_genders), dtype=int)
        for i, g in enumerate(genders):
            j = self.gender_to_idx_.get(g, None)
            if j is not None:
                gender_oh[i, j] = 1

        # Symptom multi-hot
        n_symptoms = len(self.symptom_vocab_)
        symptom_mh = np.zeros((n_samples, n_symptoms), dtype=int)
        vocab_index = {s: idx for idx, s in enumerate(self.symptom_vocab_)}
        for i, text in enumerate(X["Symptoms"].fillna("")):
            parts = [s.strip().lower() for s in str(text).split(",") if s.strip()]
            for s in parts:
                j = vocab_index.get(s)
                if j is not None:
                    symptom_mh[i, j] = 1

        # Concatenate: [age | symptom_count | gender_one_hot | symptom_multi_hot]
        features = np.hstack([age, sc, gender_oh, symptom_mh])
        return features
