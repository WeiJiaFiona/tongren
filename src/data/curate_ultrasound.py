#!/usr/bin/env python3
import json
import sys
from pathlib import Path

ROOT = Path("/public_bme/home/jiawei2022/tongren_bme_transition")
sys.path.insert(0, str(ROOT / "code"))

from src.tools.curation_tool import curate


INPUT = ROOT / "code/src/data/train/patient_features_train.json"
OUTPUT = ROOT / "dataprocess/3_curated_patient_state/curation_sample.jsonl"


def main():
    with INPUT.open("r", encoding="utf-8") as f:
        patients = json.load(f)
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    with OUTPUT.open("w", encoding="utf-8") as f:
        for patient in patients[:100]:
            item = {"患者ID": patient["患者ID"], **curate(patient)}
            f.write(json.dumps(item, ensure_ascii=False) + "\n")
    print(f"wrote={OUTPUT}")
    print("sample_count=100")


if __name__ == "__main__":
    main()
