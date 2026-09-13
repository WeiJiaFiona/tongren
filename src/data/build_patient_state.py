#!/usr/bin/env python3
import json
import sys
from pathlib import Path


ROOT = Path("/public_bme/home/jiawei2022/tongren_bme_transition")
sys.path.insert(0, str(ROOT / "code"))

from src.tools.curation_tool import curate


INPUTS = [
    ROOT / "code/src/data/train/patient_features_train.json",
    ROOT / "code/src/data/test/patient_features_test.json",
]
OUTPUT = ROOT / "dataprocess/3_curated_patient_state/patient_state.jsonl"


def missing_status(value):
    if value is None:
        return True
    if isinstance(value, dict):
        return value.get("status") == "missing"
    return False


def build_missing_mask(patient, curated):
    mask = {
        "危险因素": {k: patient["危险因素"][k] is None for k in patient["危险因素"]},
        "血液指标": {k: patient["血液指标"][k] is None for k in patient["血液指标"]},
        "颈动脉超声结构化": {
            k: missing_status(v) for k, v in curated["颈动脉超声结构化"].items()
        },
    }
    return mask


def patient_state(patient):
    curated = curate(patient)
    return {
        "患者ID": patient["患者ID"],
        "标签": patient["标签"],
        "危险因素": patient["危险因素"],
        "血液指标": patient["血液指标"],
        "颈动脉超声结构化": curated["颈动脉超声结构化"],
        "颈动脉病灶级记录": curated["颈动脉病灶级记录"],
        "颈动脉血管表格记录": curated["颈动脉血管表格记录"],
        "缺失掩码": build_missing_mask(patient, curated),
        "质量标记": curated["质量标记"],
        "证据溯源": {
            "原始超声描述存在": patient["颈动脉超声报告"].get("超声描述") is not None,
            "原始超声结论存在": patient["颈动脉超声报告"].get("超声结论") is not None,
        },
    }


def main():
    patients = []
    for path in INPUTS:
        with path.open("r", encoding="utf-8") as f:
            patients.extend(json.load(f))
    patients.sort(key=lambda x: x["患者ID"])

    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    with OUTPUT.open("w", encoding="utf-8") as f:
        for patient in patients:
            f.write(json.dumps(patient_state(patient), ensure_ascii=False) + "\n")
    print(f"output={OUTPUT}")
    print(f"patient_state_count={len(patients)}")


if __name__ == "__main__":
    main()
