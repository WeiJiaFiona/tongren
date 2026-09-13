#!/usr/bin/env python3
import math


class RiskAssessmentTool:
    def __init__(self, predictor=None, calibrator=None, threshold=0.5):
        self.predictor = predictor
        self.calibrator = calibrator
        self.threshold = threshold

    def predict(self, patient_state):
        if self.predictor is None:
            raise RuntimeError("Predictor is not configured.")
        raw_logit = self.predictor(patient_state)
        raw_probability = 1.0 / (1.0 + math.exp(-raw_logit))
        if self.calibrator is not None:
            calibrated_probability = float(self.calibrator.predict_proba([raw_logit])[0])
        else:
            calibrated_probability = raw_probability
        return {
            "raw_logit": raw_logit,
            "raw_probability": raw_probability,
            "calibrated_probability": calibrated_probability,
            "threshold": self.threshold,
            "predicted_label": int(calibrated_probability >= self.threshold),
        }
