#!/usr/bin/env python3
import json
import random
import sys
from pathlib import Path

ROOT = Path("/public_bme/home/jiawei2022/tongren_bme_transition")
sys.path.insert(0, str(ROOT / "code"))

from src.memory.patient_memory import LAB_FIELDS, PatientMemory, read_jsonl


INPUT = ROOT / "code/src/data/train_core/patient_state_train_core.jsonl"
OUTPUT = ROOT / "code/src/data/train_core/patient_state_train_core_oof_estimated.jsonl"
METRICS = ROOT / "outputs/metrics/oof_estimated_exposure_metrics.json"
K = 20
N_FOLDS = 5
SEED = 20260913


def write_jsonl(path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def main():
    states = read_jsonl(INPUT)
    rng = random.Random(SEED)
    shuffled = list(states)
    rng.shuffle(shuffled)
    folds = [shuffled[i::N_FOLDS] for i in range(N_FOLDS)]
    outputs = []
    errors = []
    attempts = 0
    covered = 0

    for fold_idx, heldout in enumerate(folds):
        heldout_ids = {s["患者ID"] for s in heldout}
        memory_rows = [s for s in states if s["患者ID"] not in heldout_ids]
        memory = PatientMemory(memory_rows)
        for state in heldout:
            updated = json.loads(json.dumps(state, ensure_ascii=False))
            for lab in LAB_FIELDS:
                true_value = state["血液指标"].get(lab)
                if true_value is None:
                    continue
                attempts += 1
                updated["血液指标"][lab] = None
                estimate = memory.estimate_lab(updated, lab, k=K, exclude_patient_id=state["患者ID"])
                if estimate is None:
                    continue
                covered += 1
                errors.append(abs(float(true_value) - estimate["value"]))
                updated["血液指标"][lab] = {
                    "value": estimate["value"],
                    "status": "estimated",
                    "method": "oof_retrieval_assisted_feature_estimation",
                    "k": estimate["k"],
                    "true_value_was_masked": true_value,
                }
            updated["oof_fold"] = fold_idx
            outputs.append(updated)

    write_jsonl(OUTPUT, sorted(outputs, key=lambda x: x["患者ID"]))
    metrics = {
        "k": K,
        "n_folds": N_FOLDS,
        "seed": SEED,
        "attempted_masked_lab_values": attempts,
        "estimated_lab_values": covered,
        "coverage": covered / attempts if attempts else 0.0,
        "mae": sum(errors) / len(errors) if errors else None,
        "median_ae": sorted(errors)[len(errors) // 2] if errors else None,
        "no_validation_or_test_used": True,
        "self_retrieval_forbidden": True,
    }
    METRICS.parent.mkdir(parents=True, exist_ok=True)
    with METRICS.open("w", encoding="utf-8") as f:
        json.dump(metrics, f, ensure_ascii=False, indent=2)
    print(json.dumps(metrics, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
