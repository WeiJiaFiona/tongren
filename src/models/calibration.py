#!/usr/bin/env python3
import pickle

import numpy as np
from sklearn.linear_model import LogisticRegression


class PlattCalibrator:
    def __init__(self):
        self.model = LogisticRegression(solver="lbfgs")

    def fit(self, logits, labels):
        x = np.asarray(logits, dtype=float).reshape(-1, 1)
        y = np.asarray(labels, dtype=int)
        self.model.fit(x, y)
        return self

    def predict_proba(self, logits):
        x = np.asarray(logits, dtype=float).reshape(-1, 1)
        return self.model.predict_proba(x)[:, 1]

    def save(self, path):
        with open(path, "wb") as f:
            pickle.dump(self, f)

    @staticmethod
    def load(path):
        with open(path, "rb") as f:
            return pickle.load(f)
