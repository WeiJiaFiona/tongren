#!/usr/bin/env python3
import csv
import json
import os
import sys
from pathlib import Path

import numpy as np
import torch
import yaml
from peft import PeftModel
from sklearn.calibration import calibration_curve
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    brier_score_loss,
    confusion_matrix,
    f1_score,
    fbeta_score,
    precision_score,
    recall_score,
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


def fit_platt_calibrator(logits, labels):
    x = np.asarray(logits, dtype=np.float64).reshape(-1, 1)
    y = np.asarray(labels, dtype=np.int32)
    try:
        model = LogisticRegression(penalty=None, solver="lbfgs")
        model.fit(x, y)
        method = {"penalty": None, "solver": "lbfgs"}
    except (TypeError, ValueError):
        model = LogisticRegression(C=1e6, solver="lbfgs")
        model.fit(x, y)
        method = {"C": 1e6, "solver": "lbfgs", "fallback_for_penalty_none": True}
    calibrator = PlattCalibrator()
    calibrator.model = model
    return calibrator, method


def metric_report(labels, probs):
    y = np.asarray(labels, dtype=np.int32)
    p = np.asarray(probs, dtype=np.float64)
    return {
        "auroc": float(roc_auc_score(y, p)) if len(set(y.tolist())) > 1 else None,
        "auprc": float(average_precision_score(y, p)) if len(set(y.tolist())) > 1 else None,
        "brier": float(brier_score_loss(y, p)),
    }


def calibration_slope_intercept(labels, probs):
    y = np.asarray(labels, dtype=np.int32)
    p = np.clip(np.asarray(probs, dtype=np.float64), 1e-6, 1 - 1e-6)
    logits = np.log(p / (1 - p)).reshape(-1, 1)
    model = LogisticRegression(penalty=None, solver="lbfgs")
    try:
        model.fit(logits, y)
    except (TypeError, ValueError):
        model = LogisticRegression(C=1e6, solver="lbfgs")
        model.fit(logits, y)
    return {
        "intercept": float(model.intercept_[0]),
        "slope": float(model.coef_[0][0]),
    }


def calibration_curve_report(labels, probs, n_bins=10):
    fraction_pos, mean_pred = calibration_curve(labels, probs, n_bins=n_bins, strategy="quantile")
    return [
        {"mean_predicted_probability": float(x), "fraction_of_positives": float(y)}
        for x, y in zip(mean_pred, fraction_pos)
    ]


def threshold_metrics(labels, probs, threshold):
    y = np.asarray(labels, dtype=np.int32)
    p = np.asarray(probs, dtype=np.float64)
    pred = (p >= float(threshold)).astype(np.int32)
    tn, fp, fn, tp = confusion_matrix(y, pred, labels=[0, 1]).ravel()
    sensitivity = recall_score(y, pred, zero_division=0)
    specificity = tn / (tn + fp) if (tn + fp) else 0.0
    precision = precision_score(y, pred, zero_division=0)
    npv = tn / (tn + fn) if (tn + fn) else 0.0
    return {
        "threshold": float(threshold),
        "tp": int(tp),
        "fp": int(fp),
        "tn": int(tn),
        "fn": int(fn),
        "sensitivity": float(sensitivity),
        "recall": float(sensitivity),
        "specificity": float(specificity),
        "precision": float(precision),
        "ppv": float(precision),
        "npv": float(npv),
        "f1": float(f1_score(y, pred, zero_division=0)),
        "f2": float(fbeta_score(y, pred, beta=2, zero_division=0)),
        "accuracy": float(accuracy_score(y, pred)),
        "positive_prediction_rate": float(pred.mean()) if len(pred) else 0.0,
        "youden_j": float(sensitivity + specificity - 1),
    }


def threshold_sweep(labels, probs):
    thresholds = sorted(set(float(x) for x in probs), reverse=True)
    return [threshold_metrics(labels, probs, threshold) for threshold in thresholds]


def choose_constrained(rows, min_specificity, label):
    candidates = [row for row in rows if row["specificity"] >= min_specificity]
    if not candidates:
        return {"label": label, "min_specificity": min_specificity, "available": False}
    selected = max(
        candidates,
        key=lambda row: (row["sensitivity"], row["f2"], row["specificity"], row["precision"]),
    )
    return {"label": label, "min_specificity": min_specificity, "available": True, **selected}


def choose_sensitivity_target(rows, target):
    selected = min(
        rows,
        key=lambda row: (abs(row["sensitivity"] - target), -row["specificity"], -row["precision"]),
    )
    return {"label": f"sensitivity_target_{target:.2f}", "target_sensitivity": target, **selected}


def write_json(path, data):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)


def write_jsonl(path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def write_csv(path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        return
    fieldnames = []
    for row in rows:
        for key in row.keys():
            if key not in fieldnames:
                fieldnames.append(key)
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


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

    calibrator, platt_fit = fit_platt_calibrator(logits, labels)
    calibrated_probs = calibrator.predict_proba(logits)
    for row, calibrated in zip(rows, calibrated_probs):
        row["calibrated_probability"] = float(calibrated)

    sweep_rows = threshold_sweep(labels, calibrated_probs)
    conservative = choose_constrained(sweep_rows, 0.85, "conservative_specificity_ge_0.85")
    primary = choose_constrained(sweep_rows, 0.80, "balanced_screening_primary_specificity_ge_0.80")
    high_sensitivity = choose_constrained(sweep_rows, 0.75, "high_sensitivity_specificity_ge_0.75")
    target_points = [choose_sensitivity_target(sweep_rows, target) for target in (0.65, 0.70, 0.75, 0.80)]
    candidate_rows = [conservative, primary, high_sensitivity] + target_points
    if not primary.get("available"):
        raise RuntimeError("No validation threshold satisfies specificity >= 0.80.")

    raw_metrics = metric_report(labels, raw_probs)
    calibrated_metrics = metric_report(labels, calibrated_probs)
    calibration_report = {
        "raw": {
            **raw_metrics,
            "calibration_intercept_slope": calibration_slope_intercept(labels, raw_probs),
            "calibration_curve": calibration_curve_report(labels, raw_probs),
        },
        "platt_calibrated": {
            **calibrated_metrics,
            "platt_a": float(calibrator.model.coef_[0][0]),
            "platt_b": float(calibrator.model.intercept_[0]),
            "platt_fit": platt_fit,
            "calibration_intercept_slope": calibration_slope_intercept(labels, calibrated_probs),
            "calibration_curve": calibration_curve_report(labels, calibrated_probs),
        },
    }

    calibrator_path = checkpoint_dir / "platt_calibrator_screening.pkl"
    calibrator.save(calibrator_path)
    logits_path = METRICS_DIR / "validation_logits_selected_checkpoint.jsonl"
    sweep_path = METRICS_DIR / "validation_calibrated_threshold_sweep.csv"
    candidate_path = METRICS_DIR / "validation_candidate_operating_points.csv"
    selected_path = METRICS_DIR / "selected_screening_threshold.json"
    summary_path = METRICS_DIR / "calibration_summary.json"
    frozen_config_path = METRICS_DIR / "frozen_agent_config.json"

    selected_threshold = {
        "selection_dataset": "validation",
        "probability_type": "platt_calibrated",
        "selection_rule": "maximize sensitivity subject to specificity >= 0.80",
        **primary,
    }

    write_jsonl(logits_path, rows)
    write_csv(sweep_path, sweep_rows)
    write_csv(candidate_path, candidate_rows)
    write_json(selected_path, selected_threshold)
    summary = {
        "status": "pass",
        "selected_checkpoint": str(checkpoint_dir),
        "validation_path": str(val_path),
        "validation_loss": val_loss,
        "validation": val_summary,
        "max_length": max_length,
        "eval_batch_size": eval_batch_size,
        "calibration": calibration_report,
        "selected_threshold": selected_threshold,
        "candidate_operating_points": candidate_rows,
        "calibrator_path": str(calibrator_path),
        "validation_logits_path": str(logits_path),
        "threshold_sweep_path": str(sweep_path),
        "candidate_operating_points_path": str(candidate_path),
        "selected_screening_threshold_path": str(selected_path),
        "frozen_test_used": False,
    }
    write_json(summary_path, summary)
    frozen_config = {
        "selected_checkpoint": str(checkpoint_dir),
        "calibrator_path": str(calibrator_path),
        "threshold": selected_threshold["threshold"],
        "threshold_selection": selected_threshold,
        "max_length": max_length,
        "model_dir": config["paths"]["model_dir"],
        "validation_calibration_summary": str(summary_path),
        "selected_screening_threshold_path": str(selected_path),
        "frozen_test_used": False,
    }
    write_json(frozen_config_path, frozen_config)
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
