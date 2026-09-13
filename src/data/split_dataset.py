#!/usr/bin/env python3
import json
import random
from collections import Counter, defaultdict
from pathlib import Path


ROOT = Path("/public_bme/home/jiawei2022/tongren_bme_transition")
STATE_PATH = ROOT / "dataprocess/3_curated_patient_state/patient_state.jsonl"
OLD_TRAIN_IDS = ROOT / "code/src/data/train/train_patient_ids.json"
TEST_IDS = ROOT / "code/src/data/test/test_patient_ids.json"
TRAIN_CORE_DIR = ROOT / "code/src/data/train_core"
VAL_DIR = ROOT / "code/src/data/validation"
TEST_DIR = ROOT / "code/src/data/test"
REPORT = ROOT / "outputs/metrics/split_balance_72_8_20.json"

SEED = 20260913
VAL_FRACTION_OF_TOTAL = 0.08
TOTAL_N = 24670


def read_json(path):
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def read_jsonl(path):
    with path.open("r", encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def write_json(path, data):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)


def write_jsonl(path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def lab_missing_count(state):
    return sum(state["缺失掩码"]["血液指标"].values())


def ultrasound_complete(state):
    return not any(state["缺失掩码"]["颈动脉超声结构化"].values())


def key(state):
    return (
        state["标签"]["半年内脑梗"],
        lab_missing_count(state),
        "us_complete" if ultrasound_complete(state) else "us_incomplete",
    )


def summarize(rows):
    n = len(rows)
    pos = sum(row["标签"]["半年内脑梗"] for row in rows)
    return {
        "n": n,
        "positive": pos,
        "positive_rate": pos / n if n else 0.0,
        "n_missing_lab_distribution": dict(sorted(Counter(lab_missing_count(r) for r in rows).items())),
        "ultrasound_structured_complete_rate": sum(ultrasound_complete(r) for r in rows) / n if n else 0.0,
    }


def main():
    states = {row["患者ID"]: row for row in read_jsonl(STATE_PATH)}
    old_train_ids = read_json(OLD_TRAIN_IDS)
    test_ids = read_json(TEST_IDS)
    if set(old_train_ids) & set(test_ids):
        raise RuntimeError("Existing train/test overlap detected.")

    train_pool = [states[i] for i in old_train_ids]
    target_val = round(TOTAL_N * VAL_FRACTION_OF_TOTAL)
    rng = random.Random(SEED)
    strata = defaultdict(list)
    for row in train_pool:
        strata[key(row)].append(row)

    val_ids = set()
    fractional = []
    for stratum_key, rows in strata.items():
        rows = list(rows)
        rng.shuffle(rows)
        exact = len(rows) * target_val / len(train_pool)
        n_val = int(exact)
        fractional.append((exact - n_val, rows[n_val:]))
        for row in rows[:n_val]:
            val_ids.add(row["患者ID"])

    deficit = target_val - len(val_ids)
    if deficit > 0:
        candidates = []
        for _, rows in sorted(fractional, reverse=True, key=lambda x: x[0]):
            candidates.extend(rows)
        rng.shuffle(candidates)
        for row in candidates[:deficit]:
            val_ids.add(row["患者ID"])

    train_core = sorted([row for row in train_pool if row["患者ID"] not in val_ids], key=lambda r: r["患者ID"])
    validation = sorted([row for row in train_pool if row["患者ID"] in val_ids], key=lambda r: r["患者ID"])
    test = sorted([states[i] for i in test_ids], key=lambda r: r["患者ID"])

    if set(r["患者ID"] for r in train_core) & set(r["患者ID"] for r in validation):
        raise RuntimeError("train_core/validation overlap")
    if (set(r["患者ID"] for r in train_core) | set(r["患者ID"] for r in validation)) & set(test_ids):
        raise RuntimeError("test leakage")

    write_jsonl(TRAIN_CORE_DIR / "patient_state_train_core.jsonl", train_core)
    write_jsonl(VAL_DIR / "patient_state_validation.jsonl", validation)
    write_jsonl(TEST_DIR / "patient_state_test.jsonl", test)
    write_json(TRAIN_CORE_DIR / "train_core_patient_ids.json", [r["患者ID"] for r in train_core])
    write_json(VAL_DIR / "validation_patient_ids.json", [r["患者ID"] for r in validation])

    report = {
        "seed": SEED,
        "test_is_existing_frozen_split": True,
        "train_core": summarize(train_core),
        "validation": summarize(validation),
        "test": summarize(test),
    }
    write_json(REPORT, report)
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
