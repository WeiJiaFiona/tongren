#!/usr/bin/env python3
import json
import os
import sys
from pathlib import Path

import numpy as np
import torch
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

from src.models.calibration import PlattCalibrator
from src.train.calibrate import load_selected_model, read_config, write_json, write_jsonl
from src.train.train_qlora import PatientStateDataset, build_collate_fn, labels_summary, read_jsonl


METRICS_DIR = ROOT / "outputs/metrics"
FROZEN_CONFIG_PATH = METRICS_DIR / "frozen_agent_config.json"
OLD_THRESHOLD = 0.147638
OLD_REFERENCE = {
    "threshold": 0.147638,
    "sensitivity": 0.4937,
    "specificity": 0.9428,
    "precision": 0.3714,
    "f1": 0.4239,
    "tp": 156,
    "fn": 160,
    "fp": 264,
    "tn": 4354,
}


def resolve_transition_path(value):
    path = Path(value)
    if path.exists():
        return path
    marker = "tongren_bme_transition/"
    text = str(path)
    if marker in text:
        return ROOT / text.split(marker, 1)[1]
    return path


def calibration_slope_intercept(labels, probs):
    y = np.asarray(labels, dtype=np.int32)
    p = np.clip(np.asarray(probs, dtype=np.float64), 1e-6, 1 - 1e-6)
    logits = np.log(p / (1 - p)).reshape(-1, 1)
    try:
        model = LogisticRegression(penalty=None, solver="lbfgs")
        model.fit(logits, y)
    except (TypeError, ValueError):
        model = LogisticRegression(C=1e6, solver="lbfgs")
        model.fit(logits, y)
    return {"intercept": float(model.intercept_[0]), "slope": float(model.coef_[0][0])}


def calibration_curve_report(labels, probs, n_bins=10):
    fraction_pos, mean_pred = calibration_curve(labels, probs, n_bins=n_bins, strategy="quantile")
    return [
        {"mean_predicted_probability": float(x), "fraction_of_positives": float(y)}
        for x, y in zip(mean_pred, fraction_pos)
    ]


def metric_report(labels, probs, threshold):
    y = np.asarray(labels, dtype=np.int32)
    p = np.asarray(probs, dtype=np.float64)
    pred = (p >= threshold).astype(np.int32)
    tn, fp, fn, tp = confusion_matrix(y, pred, labels=[0, 1]).ravel()
    sensitivity = recall_score(y, pred, zero_division=0)
    specificity = tn / (tn + fp) if (tn + fp) else 0.0
    precision = precision_score(y, pred, zero_division=0)
    npv = tn / (tn + fn) if (tn + fn) else 0.0
    return {
        "n": int(len(y)),
        "positive": int(y.sum()),
        "positive_rate": float(y.mean()) if len(y) else 0.0,
        "threshold": float(threshold),
        "auroc": float(roc_auc_score(y, p)) if len(set(y.tolist())) > 1 else None,
        "auprc": float(average_precision_score(y, p)) if len(set(y.tolist())) > 1 else None,
        "brier": float(brier_score_loss(y, p)),
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
        "tp": int(tp),
        "fp": int(fp),
        "tn": int(tn),
        "fn": int(fn),
        "confusion_matrix": {"tn": int(tn), "fp": int(fp), "fn": int(fn), "tp": int(tp)},
    }


def bootstrap_ci(labels, probs, threshold, iterations=2000, seed=42):
    y = np.asarray(labels, dtype=np.int32)
    p = np.asarray(probs, dtype=np.float64)
    rng = np.random.default_rng(seed)
    values = {"auroc": [], "auprc": [], "sensitivity": [], "specificity": []}
    n = len(y)
    for _ in range(iterations):
        idx = rng.integers(0, n, size=n)
        by = y[idx]
        bp = p[idx]
        if len(set(by.tolist())) < 2:
            continue
        report = metric_report(by, bp, threshold)
        for key in values:
            values[key].append(report[key])
    ci = {}
    for key, vals in values.items():
        if vals:
            low, high = np.percentile(vals, [2.5, 97.5])
            ci[key] = {"lower": float(low), "upper": float(high), "n_bootstrap_valid": len(vals)}
        else:
            ci[key] = {"lower": None, "upper": None, "n_bootstrap_valid": 0}
    return ci


@torch.no_grad()
def collect_test_logits(model, loader, device, pos_weight, calibrator):
    rows = []
    losses = []
    model.eval()
    for encoded, labels, patient_ids in tqdm(loader, desc="frozen_test_logits"):
        encoded = {k: v.to(device) for k, v in encoded.items()}
        labels = labels.to(device)
        logits = model(**encoded)["logits"]
        loss = F.binary_cross_entropy_with_logits(logits, labels, pos_weight=pos_weight)
        losses.append(float(loss.detach().cpu().item()))
        logit_values = logits.detach().float().cpu().tolist()
        raw_probs = torch.sigmoid(logits.detach().float().cpu()).tolist()
        calibrated_probs = calibrator.predict_proba(logit_values)
        for patient_id, label, logit, raw_prob, calibrated_prob in zip(
            patient_ids,
            labels.detach().float().cpu().tolist(),
            logit_values,
            raw_probs,
            calibrated_probs,
        ):
            rows.append(
                {
                    "患者ID": patient_id,
                    "label": float(label),
                    "logit": float(logit),
                    "raw_probability": float(raw_prob),
                    "calibrated_probability": float(calibrated_prob),
                }
            )
    return rows, float(np.mean(losses)) if losses else None


def compare_old_new(old_report, new_report):
    reduced_fn = old_report["fn"] - new_report["fn"]
    additional_fp = new_report["fp"] - old_report["fp"]
    return {
        "old_threshold_reference": OLD_REFERENCE,
        "old_threshold_recomputed": old_report,
        "new_threshold": new_report,
        "delta_sensitivity": float(new_report["sensitivity"] - old_report["sensitivity"]),
        "delta_specificity": float(new_report["specificity"] - old_report["specificity"]),
        "reduced_fn": int(reduced_fn),
        "additional_fp": int(additional_fp),
        "additional_fp_per_reduced_fn": float(additional_fp / reduced_fn) if reduced_fn > 0 else None,
    }


def main():
    if not FROZEN_CONFIG_PATH.exists():
        raise FileNotFoundError(
            f"Missing frozen config: {FROZEN_CONFIG_PATH}. Run calibration before frozen test evaluation."
        )

    config = read_config()
    with FROZEN_CONFIG_PATH.open("r", encoding="utf-8") as f:
        frozen_config = json.load(f)

    checkpoint_dir = resolve_transition_path(frozen_config["selected_checkpoint"])
    calibrator_path = resolve_transition_path(frozen_config["calibrator_path"])
    threshold = float(frozen_config["threshold"])
    max_length = int(frozen_config.get("max_length", os.environ.get("TRAIN_MAX_LENGTH", "512")))
    eval_batch_size = int(os.environ.get("EVAL_BATCH_SIZE", str(config["training"]["micro_batch_size"])))
    bootstrap_iterations = int(os.environ.get("BOOTSTRAP_ITERATIONS", "2000"))
    bootstrap_seed = int(os.environ.get("BOOTSTRAP_SEED", "42"))

    test_path = Path(config["paths"]["test"])
    test_rows_raw = read_jsonl(test_path)
    test_summary = labels_summary(test_rows_raw)
    positive = test_summary["positive"]
    negative = test_summary["n"] - positive
    pos_weight = torch.tensor(negative / positive if positive else 1.0, dtype=torch.float32)

    tokenizer = AutoTokenizer.from_pretrained(config["paths"]["model_dir"], trust_remote_code=True, use_fast=False)
    tokenizer.padding_side = "right"
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    loader = DataLoader(
        PatientStateDataset(test_path),
        batch_size=eval_batch_size,
        shuffle=False,
        num_workers=0,
        collate_fn=build_collate_fn(tokenizer, max_length),
    )
    model, device = load_selected_model(config, checkpoint_dir)
    pos_weight = pos_weight.to(device)
    calibrator = PlattCalibrator.load(calibrator_path)

    rows, test_loss = collect_test_logits(model, loader, device, pos_weight, calibrator)
    labels = [row["label"] for row in rows]
    raw_probs = [row["raw_probability"] for row in rows]
    calibrated_probs = [row["calibrated_probability"] for row in rows]
    for row in rows:
        row["predicted_label"] = int(row["calibrated_probability"] >= threshold)

    raw_metrics = metric_report(labels, raw_probs, 0.5)
    calibrated_metrics = metric_report(labels, calibrated_probs, threshold)
    old_metrics = metric_report(labels, calibrated_probs, OLD_THRESHOLD)
    comparison = compare_old_new(old_metrics, calibrated_metrics)
    bootstrap = bootstrap_ci(
        labels,
        calibrated_probs,
        threshold,
        iterations=bootstrap_iterations,
        seed=bootstrap_seed,
    )
    calibration_report = {
        "calibration_intercept_slope": calibration_slope_intercept(labels, calibrated_probs),
        "calibration_curve": calibration_curve_report(labels, calibrated_probs),
    }

    predictions_path = METRICS_DIR / "frozen_test_screening_predictions.jsonl"
    summary_path = METRICS_DIR / "frozen_test_screening_metrics.json"
    legacy_predictions_path = METRICS_DIR / "frozen_test_predictions.jsonl"
    legacy_summary_path = METRICS_DIR / "frozen_test_evaluation_summary.json"

    summary = {
        "status": "pass",
        "frozen_test_used": True,
        "selected_checkpoint": str(checkpoint_dir),
        "calibrator_path": str(calibrator_path),
        "test_path": str(test_path),
        "test_loss": test_loss,
        "test": test_summary,
        "max_length": max_length,
        "eval_batch_size": eval_batch_size,
        "threshold": threshold,
        "raw_metrics_threshold_0_5": raw_metrics,
        "calibrated_metrics_frozen_threshold": calibrated_metrics,
        "calibration": calibration_report,
        "bootstrap": {"iterations": bootstrap_iterations, "seed": bootstrap_seed, "ci_95": bootstrap},
        "old_vs_new_threshold_summary": comparison,
        "predictions_path": str(predictions_path),
        "frozen_config_path": str(FROZEN_CONFIG_PATH),
    }
    write_jsonl(predictions_path, rows)
    write_jsonl(legacy_predictions_path, rows)
    write_json(summary_path, summary)
    write_json(legacy_summary_path, summary)
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
