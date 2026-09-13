#!/usr/bin/env python3
import json
from pathlib import Path


ROOT = Path("/public_bme/home/jiawei2022/tongren_bme_transition")
TRAIN_PATH = ROOT / "code/src/data/train/patient_features_train.json"
TEST_PATH = ROOT / "code/src/data/test/patient_features_test.json"
TRAIN_IDS_PATH = ROOT / "code/src/data/train/train_patient_ids.json"
TEST_IDS_PATH = ROOT / "code/src/data/test/test_patient_ids.json"
OUT_PATH = ROOT / "outputs/metrics/patient_features_validation_report.json"

RISK_FIELDS = [
    "年龄",
    "性别",
    "体质指数(BMI)",
    "收缩压",
    "舒张压",
    "吸烟状态",
    "饮酒",
    "高血压",
    "糖尿病",
    "脑卒中家族史",
    "血脂异常",
    "缺乏体育活动",
]

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

ULTRASOUND_FIELDS = ["超声描述", "超声结论"]


def load_json(path):
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def is_number_or_null(value):
    return value is None or isinstance(value, (int, float))


def validate_record(record, errors, prefix):
    if set(record.keys()) != {"患者ID", "标签", "危险因素", "血液指标", "颈动脉超声报告"}:
        errors.append(f"{prefix}: top-level keys mismatch")
    if not isinstance(record.get("患者ID"), int):
        errors.append(f"{prefix}: 患者ID is not int")
    label = record.get("标签", {}).get("半年内脑梗")
    if label not in (0, 1):
        errors.append(f"{prefix}: invalid label {label!r}")
    if list(record.get("危险因素", {}).keys()) != RISK_FIELDS:
        errors.append(f"{prefix}: risk fields/order mismatch")
    if list(record.get("血液指标", {}).keys()) != LAB_FIELDS:
        errors.append(f"{prefix}: lab fields/order mismatch")
    if list(record.get("颈动脉超声报告", {}).keys()) != ULTRASOUND_FIELDS:
        errors.append(f"{prefix}: ultrasound fields/order mismatch")

    risk = record.get("危险因素", {})
    for field in ["年龄", "体质指数(BMI)", "收缩压", "舒张压"]:
        if not is_number_or_null(risk.get(field)):
            errors.append(f"{prefix}: {field} is not numeric/null")
    for field, value in record.get("血液指标", {}).items():
        if not is_number_or_null(value):
            errors.append(f"{prefix}: lab {field} is not numeric/null")
    for field, value in record.get("颈动脉超声报告", {}).items():
        if value is not None and not isinstance(value, str):
            errors.append(f"{prefix}: ultrasound {field} is not string/null")


def summarize(records):
    n = len(records)
    positives = sum(r["标签"]["半年内脑梗"] == 1 for r in records)
    return {
        "n": n,
        "positive": positives,
        "positive_rate": positives / n if n else 0.0,
        "id_min": min(r["患者ID"] for r in records) if records else None,
        "id_max": max(r["患者ID"] for r in records) if records else None,
    }


def main():
    train = load_json(TRAIN_PATH)
    test = load_json(TEST_PATH)
    train_ids = load_json(TRAIN_IDS_PATH)
    test_ids = load_json(TEST_IDS_PATH)

    errors = []
    for name, records in [("train", train), ("test", test)]:
        ids = [r.get("患者ID") for r in records]
        if len(ids) != len(set(ids)):
            errors.append(f"{name}: duplicate patient IDs")
        for idx, record in enumerate(records[:50]):
            validate_record(record, errors, f"{name}[{idx}]")

    if set(train_ids) != {r["患者ID"] for r in train}:
        errors.append("train IDs file does not match train JSON")
    if set(test_ids) != {r["患者ID"] for r in test}:
        errors.append("test IDs file does not match test JSON")
    overlap = sorted(set(train_ids) & set(test_ids))
    if overlap:
        errors.append(f"train/test overlap: {overlap[:10]}")

    report = {
        "status": "pass" if not errors else "fail",
        "errors": errors,
        "train": summarize(train),
        "test": summarize(test),
        "overlap_count": len(overlap),
        "expected_schema": {
            "risk_fields": RISK_FIELDS,
            "lab_fields": LAB_FIELDS,
            "ultrasound_fields": ULTRASOUND_FIELDS,
        },
    }
    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    with OUT_PATH.open("w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=2)
    print(json.dumps(report, ensure_ascii=False, indent=2))
    if errors:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
