#!/usr/bin/env python3
import json
import statistics
import sys
from pathlib import Path

ROOT = Path("/public_bme/home/jiawei2022/tongren_bme_transition")
sys.path.insert(0, str(ROOT / "code"))

from src.data.serialize_patient import serialize_patient_state


INPUT = ROOT / "code/src/data/train_core/patient_state_train_core.jsonl"
MODEL_DIR = ROOT / "code/model/Baichuan-M1-14B-Instruct"
OUTPUT = ROOT / "outputs/metrics/token_length_stats.json"


def read_jsonl(path):
    with path.open("r", encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def percentile(values, q):
    values = sorted(values)
    idx = min(len(values) - 1, round((len(values) - 1) * q))
    return values[idx]


def main():
    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(MODEL_DIR, trust_remote_code=True, use_fast=False)
    tokenizer.padding_side = "right"
    states = read_jsonl(INPUT)
    lengths = []
    for state in states:
        messages = [
            {"role": "system", "content": "你是临床脑梗标签识别模型。"},
            {"role": "user", "content": serialize_patient_state(state)},
        ]
        ids = tokenizer.apply_chat_template(messages, tokenize=True, add_generation_prompt=True)
        lengths.append(len(ids))
    report = {
        "n": len(lengths),
        "mean": statistics.mean(lengths),
        "p50": percentile(lengths, 0.50),
        "p95": percentile(lengths, 0.95),
        "p99": percentile(lengths, 0.99),
        "max": max(lengths),
        "recommended_max_length": 512 if percentile(lengths, 0.99) <= 512 else 768 if percentile(lengths, 0.99) <= 768 else 1024,
    }
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    with OUTPUT.open("w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=2)
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
