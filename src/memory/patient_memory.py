#!/usr/bin/env python3
import json
import math
from pathlib import Path


LAB_FIELDS = [
    "高密度脂蛋白胆固醇",
    "低密度脂蛋白胆固醇",
    "非高密度脂蛋白胆固醇",
    "甘油三酯",
    "载脂蛋白B",
    "脂蛋白(a)",
    "糖化血红蛋白",
    "同型半胱氨酸",
    "纤维蛋白原",
    "小而密低密度脂蛋白胆固醇",
]


def read_jsonl(path):
    with Path(path).open("r", encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def scalar_features(state):
    values = {}
    for section in ("危险因素", "血液指标"):
        for key, value in state[section].items():
            if value is not None:
                values[f"{section}.{key}"] = value
    for key, wrapped in state["颈动脉超声结构化"].items():
        if isinstance(wrapped, dict) and wrapped.get("value") is not None:
            values[f"颈动脉超声结构化.{key}"] = wrapped["value"]
    return values


class PatientMemory:
    def __init__(self, states):
        self.states = list(states)
        self.by_id = {state["患者ID"]: state for state in self.states}
        self.feature_scales = self._fit_scales()

    @classmethod
    def from_jsonl(cls, path):
        return cls(read_jsonl(path))

    def to_json(self, path):
        payload = {"states": self.states, "feature_scales": self.feature_scales}
        with Path(path).open("w", encoding="utf-8") as f:
            json.dump(payload, f, ensure_ascii=False)

    def _fit_scales(self):
        numeric_values = {}
        for state in self.states:
            for key, value in scalar_features(state).items():
                if isinstance(value, (int, float)) and not isinstance(value, bool):
                    numeric_values.setdefault(key, []).append(float(value))
        scales = {}
        for key, values in numeric_values.items():
            span = max(values) - min(values)
            scales[key] = span if span > 0 else 1.0
        return scales

    def distance(self, query, candidate, exclude_lab=None):
        q = scalar_features(query)
        c = scalar_features(candidate)
        distances = []
        for key in sorted(set(q) & set(c)):
            if key.startswith("血液指标.") and key.split(".", 1)[1] == exclude_lab:
                continue
            qv, cv = q[key], c[key]
            if isinstance(qv, (int, float)) and isinstance(cv, (int, float)):
                scale = self.feature_scales.get(key, 1.0)
                distances.append(abs(float(qv) - float(cv)) / scale)
            else:
                distances.append(0.0 if qv == cv else 1.0)
        if len(distances) < 3:
            return None
        return sum(distances) / len(distances)

    def estimate_lab(self, query, lab_name, k=20, exclude_patient_id=None):
        candidates = []
        for state in self.states:
            if exclude_patient_id is not None and state["患者ID"] == exclude_patient_id:
                continue
            value = state["血液指标"].get(lab_name)
            if value is None:
                continue
            d = self.distance(query, state, exclude_lab=lab_name)
            if d is not None and math.isfinite(d):
                candidates.append((d, float(value)))
        candidates.sort(key=lambda x: x[0])
        top = candidates[:k]
        if len(top) < max(3, min(k, 5)):
            return None
        weights = [1.0 / (d + 1e-6) for d, _ in top]
        estimate = sum(w * value for w, (_, value) in zip(weights, top)) / sum(weights)
        return {"value": round(estimate, 6), "k": len(top)}
