#!/usr/bin/env python3
import json
import os
import sys
from pathlib import Path

import numpy as np
import torch
import yaml
from peft import PeftModel
from sklearn.metrics import (
    average_precision_score,
    brier_score_loss,
    f1_score,
    precision_recall_curve,
    roc_auc_score,
)
from torch.nn import functional as F
from torch.utils.data import DataLoader
from tqdm import tqdm
from transformers import AutoTokenizer


ROOT = Path(os.environ.get("TONGREN_PROJECT_ROOT", "/public_bme/home/jiawei2022/tongren_bme_transition"))
sys.path.insert(0, str(ROOT / "code"))

from src.models.baichuan_risk_classifier import BaichuanRiskClassifier, load_baichuan_causal_lm
from src.models.calibration import PlattCalibrator
from src.train.train_qlora import PatientStateDataset, build_collate_fn, input_device, labels_summary, read_jsonl


CONFIG_PATH = ROOT / "code/src/config/train_config.yaml"
CHECKPOINT_ROOT = ROOT / "outputs/checkpoints/qlora_risk_classifier"
DEFAULT_CHECKPOINT = CHECKPOINT_ROOT / "best_validation_auprc"
METRICS_DIR = ROOT / "outputs/metrics"


def read_config():
    with CONFIG_PATH.open("r", encoding="utf-8") as f:
        config = yaml.safe_load(f)
    for key, value in list(config.get("paths", {}).items()):
        path = Path(value)
        if not path.is_absolute() or path.exists():
            continue
        marker = "tongren_bme_transition/"
        text = str(path)
        if marker in text:
            config["paths"][key] = str(ROOT / text.split(marker, 1)[1])
    return config


def sigmoid(values):
    return 1.0 / (1.0 + np.exp(-np.asarray(values, dtype=np.float64)))


def metric_report(labels, probs):
    y = np.asarray(labels, dtype=np.int32)
    p = np.asarray(probs, dtype=np.float64)
    return {
        "auprc": float(average_precision_score(y, p)) if len(set(y.tolist())) > 1 else None,
        "roc_auc": float(roc_auc_score(y, p)) if len(set(y.tolist())) > 1 else None,
        "brier": float(brier_score_loss(y, p)),
    }


def select_threshold(labels, probs):
    y = np.asarray(labels, dtype=np.int32)
    p = np.asarray(probs, dtype=np.float64)
    precision, recall, thresholds = precision_recall_curve(y, p)
    candidates = []
    for idx, threshold in enumerate(thresholds):
        pred = (p >= threshold).astype(np.int32)
        candidates.append(
            {
                "threshold": float(threshold),
                "f1": float(f1_score(y, pred, zero_division=0)),
                "precision": float(precision[idx]),
                "recall": float(recall[idx]),
                "positive_prediction_rate": float(pred.mean()),
            }
        )
    if not candidates:
        return {
            "threshold": 0.5,
            "f1": None,
            "precision": None,
            "recall": None,
            "positive_prediction_rate": None,
        }
    return max(candidates, key=lambda x: (x["f1"], x["recall"], -x["threshold"]))


def load_selected_model(config, checkpoint_dir):
    adapter_dir = checkpoint_dir / "lora_adapter"
    head_path = checkpoint_dir / "classifier_head.pt"
    if not adapter_dir.exists():
        raise FileNotFoundError(f"Missing LoRA adapter dir: {adapter_dir}")
    if not head_path.exists():
        raise FileNotFoundError(f"Missing classifier head: {head_path}")

    print(f"selected_checkpoint={checkpoint_dir}")
    print("loading_base_model=started")
    base = load_baichuan_causal_lm(config["paths"]["model_dir"])
    base.config.use_cache = False
    print("loading_base_model=done")

    print("loading_lora_adapter=started")
    lm = PeftModel.from_pretrained(base, adapter_dir, is_trainable=False)
    model = BaichuanRiskClassifier(
        lm,
        hidden_size=int(config["model"]["hidden_size"]),
        dropout=float(config["model"]["classifier_dropout"]),
    )
    model.classifier.load_state_dict(torch.load(head_path, map_location="cpu"))
    device = input_device(model.causal_lm)
    model.classifier.to(device)
    model.eval()
    print("loading_lora_adapter=done")
    return model, device


@torch.no_grad()
def collect_validation_logits(model, loader, device, pos_weight):
    rows = []
    losses = []
    model.eval()
    for encoded, labels, patient_ids in tqdm(loader, desc="validation_logits"):
        encoded = {k: v.to(device) for k, v in encoded.items()}
        labels = labels.to(device)
        logits = model(**encoded)["logits"]
        loss = F.binary_cross_entropy_with_logits(logits, labels, pos_weight=pos_weight)
        losses.append(float(loss.detach().cpu().item()))
        probs = torch.sigmoid(logits.detach().float().cpu())
        for patient_id, label, logit, prob in zip(
            patient_ids,
            labels.detach().float().cpu().tolist(),
            logits.detach().float().cpu().tolist(),
            probs.tolist(),
        ):
            rows.append(
                {
                    "患者ID": patient_id,
                    "label": float(label),
                    "logit": float(logit),
                    "raw_probability": float(prob),
                }
            )
    return rows, float(np.mean(losses)) if losses else None


def write_json(path, data):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)


def write_jsonl(path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def main():
    config = read_config()
    checkpoint_dir = Path(os.environ.get("SELECTED_CHECKPOINT", str(DEFAULT_CHECKPOINT)))
    max_length = int(os.environ.get("TRAIN_MAX_LENGTH", "512"))
    eval_batch_size = int(os.environ.get("EVAL_BATCH_SIZE", str(config["training"]["micro_batch_size"])))

    val_path = Path(config["paths"]["validation"])
    val_rows_raw = read_jsonl(val_path)
    val_summary = labels_summary(val_rows_raw)
    positive = val_summary["positive"]
    negative = val_summary["n"] - positive
    pos_weight = torch.tensor(negative / positive if positive else 1.0, dtype=torch.float32)

    tokenizer = AutoTokenizer.from_pretrained(config["paths"]["model_dir"], trust_remote_code=True, use_fast=False)
    tokenizer.padding_side = "right"
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    loader = DataLoader(
        PatientStateDataset(val_path),
        batch_size=eval_batch_size,
        shuffle=False,
        num_workers=0,
        collate_fn=build_collate_fn(tokenizer, max_length),
    )
    model, device = load_selected_model(config, checkpoint_dir)
    pos_weight = pos_weight.to(device)

    rows, val_loss = collect_validation_logits(model, loader, device, pos_weight)
    labels = [row["label"] for row in rows]
    logits = [row["logit"] for row in rows]
    raw_probs = [row["raw_probability"] for row in rows]

    calibrator = PlattCalibrator().fit(logits, labels)
    calibrated_probs = calibrator.predict_proba(logits)
    for row, calibrated in zip(rows, calibrated_probs):
        row["calibrated_probability"] = float(calibrated)

    threshold = select_threshold(labels, calibrated_probs)
    raw_metrics = metric_report(labels, raw_probs)
    calibrated_metrics = metric_report(labels, calibrated_probs)

    calibrator_path = checkpoint_dir / "platt_calibrator.pkl"
    calibrator.save(calibrator_path)
    logits_path = METRICS_DIR / "validation_logits_selected_checkpoint.jsonl"
    summary_path = METRICS_DIR / "calibration_summary.json"
    frozen_config_path = METRICS_DIR / "frozen_agent_config.json"

    write_jsonl(logits_path, rows)
    summary = {
        "status": "pass",
        "selected_checkpoint": str(checkpoint_dir),
        "validation_path": str(val_path),
        "validation_loss": val_loss,
        "validation": val_summary,
        "max_length": max_length,
        "eval_batch_size": eval_batch_size,
        "raw_metrics": raw_metrics,
        "calibrated_metrics": calibrated_metrics,
        "selected_threshold": threshold,
        "calibrator_path": str(calibrator_path),
        "validation_logits_path": str(logits_path),
        "frozen_test_used": False,
    }
    write_json(summary_path, summary)
    frozen_config = {
        "selected_checkpoint": str(checkpoint_dir),
        "calibrator_path": str(calibrator_path),
        "threshold": threshold["threshold"],
        "max_length": max_length,
        "model_dir": config["paths"]["model_dir"],
        "validation_calibration_summary": str(summary_path),
        "frozen_test_used": False,
    }
    write_json(frozen_config_path, frozen_config)
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
