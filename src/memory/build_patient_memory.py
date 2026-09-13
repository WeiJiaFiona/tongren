#!/usr/bin/env python3
import json
import sys
from pathlib import Path

ROOT = Path("/public_bme/home/jiawei2022/tongren_bme_transition")
sys.path.insert(0, str(ROOT / "code"))

from src.memory.patient_memory import read_jsonl


TRAIN_CORE = ROOT / "code/src/data/train_core/patient_state_train_core.jsonl"
VAL_IDS = ROOT / "code/src/data/validation/validation_patient_ids.json"
TEST_IDS = ROOT / "code/src/data/test/test_patient_ids.json"
OUT_PARQUET = ROOT / "memory/train_core_patient_memory.parquet"
OUT_JSON = ROOT / "memory/train_core_patient_memory.json"


def load_ids(path):
    with path.open("r", encoding="utf-8") as f:
        return set(json.load(f))


def main():
    states = read_jsonl(TRAIN_CORE)
    train_ids = {s["患者ID"] for s in states}
    val_ids = load_ids(VAL_IDS)
    test_ids = load_ids(TEST_IDS)
    if train_ids & val_ids:
        raise RuntimeError("validation leakage in train_core memory")
    if train_ids & test_ids:
        raise RuntimeError("test leakage in train_core memory")

    rows = [
        {
            "患者ID": state["患者ID"],
            "label_removed": True,
            "state_json": json.dumps({k: v for k, v in state.items() if k != "标签"}, ensure_ascii=False),
        }
        for state in states
    ]

    OUT_PARQUET.parent.mkdir(parents=True, exist_ok=True)
    try:
        import pandas as pd

        pd.DataFrame(rows).to_parquet(OUT_PARQUET, index=False)
        output = str(OUT_PARQUET)
    except Exception:
        with OUT_JSON.open("w", encoding="utf-8") as f:
            json.dump(rows, f, ensure_ascii=False)
        output = str(OUT_JSON)

    report = {
        "output": output,
        "memory_patient_count": len(rows),
        "validation_leakage": 0,
        "test_leakage": 0,
        "label_removed": True,
    }
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
