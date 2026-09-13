#!/usr/bin/env python3
import json
import math
import os
import random
import shutil
import sys
from pathlib import Path

import numpy as np
import torch
import yaml
from sklearn.metrics import average_precision_score, roc_auc_score
from torch.nn import functional as F
from torch.optim import AdamW
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm
from transformers import AutoTokenizer, get_linear_schedule_with_warmup


ROOT = Path("/public_bme/home/jiawei2022/tongren_bme_transition")
sys.path.insert(0, str(ROOT / "code"))

from src.data.serialize_patient import serialize_patient_state
from src.models.baichuan_risk_classifier import BaichuanRiskClassifier, load_baichuan_causal_lm
from src.models.qlora import attach_lora


CONFIG_PATH = ROOT / "code/src/config/train_config.yaml"
RUN_DIR = ROOT / "outputs/checkpoints/qlora_risk_classifier"
METRICS_DIR = ROOT / "outputs/metrics"
BEST_DIR = RUN_DIR / "best_validation_auprc"
LAST_DIR = RUN_DIR / "last"


class PatientStateDataset(Dataset):
    def __init__(self, path):
        with Path(path).open("r", encoding="utf-8") as f:
            self.rows = [json.loads(line) for line in f if line.strip()]

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, idx):
        row = self.rows[idx]
        return {
            "patient_id": row["患者ID"],
            "label": float(row["标签"]["半年内脑梗"]),
            "text": serialize_patient_state(row),
        }


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def read_config():
    with CONFIG_PATH.open("r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def read_jsonl(path):
    with Path(path).open("r", encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def input_device(model):
    if hasattr(model, "hf_device_map") and model.hf_device_map:
        for device in model.hf_device_map.values():
            if isinstance(device, str) and device.startswith("cuda"):
                return torch.device(device)
            if isinstance(device, int):
                return torch.device(f"cuda:{device}")
    if torch.cuda.is_available():
        return torch.device("cuda:0")
    return torch.device("cpu")


def build_collate_fn(tokenizer, max_length):
    def collate(batch):
        messages = [
            [
                {"role": "system", "content": "你是临床脑梗标签识别模型。"},
                {"role": "user", "content": item["text"]},
            ]
            for item in batch
        ]
        texts = [
            tokenizer.apply_chat_template(m, tokenize=False, add_generation_prompt=True)
            for m in messages
        ]
        encoded = tokenizer(
            texts,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=max_length,
        )
        labels = torch.tensor([item["label"] for item in batch], dtype=torch.float32)
        patient_ids = [item["patient_id"] for item in batch]
        return encoded, labels, patient_ids

    return collate


def trainable_parameter_report(model):
    trainable = 0
    total = 0
    names = []
    for name, param in model.named_parameters():
        n = param.numel()
        total += n
        if param.requires_grad:
            trainable += n
            if len(names) < 30:
                names.append(name)
    return {"trainable": trainable, "total": total, "trainable_names_head": names}


def split_parameter_groups(model, lora_lr, classifier_lr, weight_decay):
    lora_params = []
    classifier_params = []
    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue
        if name.startswith("classifier."):
            classifier_params.append(param)
        else:
            lora_params.append(param)
    return [
        {"params": lora_params, "lr": lora_lr, "weight_decay": weight_decay},
        {"params": classifier_params, "lr": classifier_lr, "weight_decay": weight_decay},
    ]


def labels_summary(rows):
    labels = [row["标签"]["半年内脑梗"] for row in rows]
    n = len(labels)
    pos = sum(labels)
    return {"n": n, "positive": pos, "positive_rate": pos / n if n else 0.0}


def save_checkpoint(model, output_dir, config, metrics):
    if output_dir.exists():
        shutil.rmtree(output_dir)
    adapter_dir = output_dir / "lora_adapter"
    output_dir.mkdir(parents=True, exist_ok=True)
    model.causal_lm.save_pretrained(adapter_dir)
    torch.save(model.classifier.state_dict(), output_dir / "classifier_head.pt")
    with (output_dir / "train_config_used.yaml").open("w", encoding="utf-8") as f:
        yaml.safe_dump(config, f, allow_unicode=True, sort_keys=False)
    with (output_dir / "metrics.json").open("w", encoding="utf-8") as f:
        json.dump(metrics, f, ensure_ascii=False, indent=2)


@torch.no_grad()
def evaluate(model, loader, device, pos_weight):
    model.eval()
    all_logits = []
    all_labels = []
    losses = []
    for encoded, labels, _ in tqdm(loader, desc="validation", leave=False):
        encoded = {k: v.to(device) for k, v in encoded.items()}
        labels = labels.to(device)
        logits = model(**encoded)["logits"]
        loss = F.binary_cross_entropy_with_logits(logits, labels, pos_weight=pos_weight)
        losses.append(float(loss.detach().cpu().item()))
        all_logits.extend(logits.detach().float().cpu().tolist())
        all_labels.extend(labels.detach().float().cpu().tolist())

    probs = torch.sigmoid(torch.tensor(all_logits, dtype=torch.float32)).numpy()
    y = np.array(all_labels, dtype=np.float32)
    report = {
        "loss": float(np.mean(losses)) if losses else None,
        "auprc": float(average_precision_score(y, probs)) if len(set(y.tolist())) > 1 else None,
        "roc_auc": float(roc_auc_score(y, probs)) if len(set(y.tolist())) > 1 else None,
        "n": int(len(y)),
        "positive": int(y.sum()),
        "positive_rate": float(y.mean()) if len(y) else 0.0,
    }
    model.train()
    return report


def main():
    config = read_config()
    train_cfg = config["training"]
    paths = config["paths"]
    model_cfg = config["model"]

    seed = int(os.environ.get("TRAIN_SEED", "20260913"))
    max_length = int(os.environ.get("TRAIN_MAX_LENGTH", "512"))
    max_steps_env = os.environ.get("TRAIN_MAX_STEPS")
    max_steps = int(max_steps_env) if max_steps_env else None
    eval_steps = int(os.environ.get("EVAL_STEPS", "200"))
    save_last = os.environ.get("SAVE_LAST", "1") != "0"

    set_seed(seed)
    RUN_DIR.mkdir(parents=True, exist_ok=True)
    METRICS_DIR.mkdir(parents=True, exist_ok=True)

    train_path = Path(paths["train_core"])
    val_path = Path(paths["validation"])
    train_rows = read_jsonl(train_path)
    val_rows = read_jsonl(val_path)
    train_summary = labels_summary(train_rows)
    val_summary = labels_summary(val_rows)
    print("train_summary=" + json.dumps(train_summary, ensure_ascii=False))
    print("validation_summary=" + json.dumps(val_summary, ensure_ascii=False))

    train_dataset = PatientStateDataset(train_path)
    val_dataset = PatientStateDataset(val_path)

    tokenizer = AutoTokenizer.from_pretrained(paths["model_dir"], trust_remote_code=True, use_fast=False)
    tokenizer.padding_side = "right"
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    collate_fn = build_collate_fn(tokenizer, max_length)
    train_loader = DataLoader(
        train_dataset,
        batch_size=int(train_cfg["micro_batch_size"]),
        shuffle=True,
        num_workers=int(os.environ.get("NUM_WORKERS", "0")),
        collate_fn=collate_fn,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=int(os.environ.get("EVAL_BATCH_SIZE", str(train_cfg["micro_batch_size"]))),
        shuffle=False,
        num_workers=0,
        collate_fn=collate_fn,
    )

    print("loading_base_model=started")
    causal_lm = load_baichuan_causal_lm(paths["model_dir"])
    causal_lm.config.use_cache = False
    print("loading_base_model=done")

    print("attaching_lora=started")
    causal_lm = attach_lora(
        causal_lm,
        prepare_kbit=bool(os.environ.get("GRADIENT_CHECKPOINTING", "1") != "0"),
    )
    causal_lm.print_trainable_parameters()
    print("attaching_lora=done")

    model = BaichuanRiskClassifier(
        causal_lm,
        hidden_size=int(model_cfg["hidden_size"]),
        dropout=float(model_cfg["classifier_dropout"]),
    )
    device = input_device(model.causal_lm)
    model.classifier.to(device)
    model.train()
    if hasattr(model.causal_lm, "enable_input_require_grads"):
        model.causal_lm.enable_input_require_grads()

    positive = train_summary["positive"]
    negative = train_summary["n"] - positive
    pos_weight_value = negative / positive if positive else 1.0
    pos_weight = torch.tensor(pos_weight_value, dtype=torch.float32, device=device)

    grad_accum = int(train_cfg["gradient_accumulation_steps"])
    epochs = int(train_cfg["epochs"])
    steps_per_epoch = math.ceil(len(train_loader) / grad_accum)
    total_steps = steps_per_epoch * epochs
    if max_steps is not None:
        total_steps = min(total_steps, max_steps)
    warmup_steps = int(total_steps * float(train_cfg["warmup_ratio"]))

    optimizer = AdamW(
        split_parameter_groups(
            model,
            lora_lr=float(train_cfg["lora_lr"]),
            classifier_lr=float(train_cfg["classifier_lr"]),
            weight_decay=float(train_cfg["weight_decay"]),
        )
    )
    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=warmup_steps,
        num_training_steps=total_steps,
    )

    run_meta = {
        "seed": seed,
        "max_length": max_length,
        "max_steps": max_steps,
        "eval_steps": eval_steps,
        "gradient_accumulation_steps": grad_accum,
        "effective_batch_size": int(train_cfg["micro_batch_size"]) * grad_accum,
        "total_optimizer_steps_planned": total_steps,
        "warmup_steps": warmup_steps,
        "pos_weight": pos_weight_value,
        "train": train_summary,
        "validation": val_summary,
        "trainable_parameters": trainable_parameter_report(model),
        "frozen_test_used": False,
    }
    print("run_meta=" + json.dumps(run_meta, ensure_ascii=False))

    best_auprc = -1.0
    best_metrics = None
    global_step = 0
    micro_step = 0
    running_loss = []
    stop_training = False

    optimizer.zero_grad(set_to_none=True)
    for epoch in range(epochs):
        progress = tqdm(train_loader, desc=f"epoch_{epoch + 1}")
        for encoded, labels, _ in progress:
            micro_step += 1
            encoded = {k: v.to(device) for k, v in encoded.items()}
            labels = labels.to(device)
            logits = model(**encoded)["logits"]
            loss = F.binary_cross_entropy_with_logits(logits, labels, pos_weight=pos_weight)
            (loss / grad_accum).backward()
            running_loss.append(float(loss.detach().cpu().item()))

            if micro_step % grad_accum != 0:
                continue

            torch.nn.utils.clip_grad_norm_(model.parameters(), float(train_cfg["gradient_clip"]))
            optimizer.step()
            scheduler.step()
            optimizer.zero_grad(set_to_none=True)
            global_step += 1

            mean_loss = float(np.mean(running_loss[-grad_accum:]))
            progress.set_postfix({"step": global_step, "loss": f"{mean_loss:.4f}"})

            should_eval = global_step == 1 or global_step % eval_steps == 0 or global_step == total_steps
            if should_eval:
                val_metrics = evaluate(model, val_loader, device, pos_weight)
                val_metrics.update({"global_step": global_step, "epoch": epoch + 1, "train_loss_recent": mean_loss})
                print("validation_metrics=" + json.dumps(val_metrics, ensure_ascii=False))
                current_auprc = val_metrics["auprc"] if val_metrics["auprc"] is not None else -1.0
                if current_auprc > best_auprc:
                    best_auprc = current_auprc
                    best_metrics = val_metrics
                    save_checkpoint(model, BEST_DIR, config, {"run_meta": run_meta, "validation": best_metrics})
                    print(f"best_checkpoint_saved={BEST_DIR}")

            if global_step >= total_steps:
                stop_training = True
                break
        if stop_training:
            break

    final_metrics = evaluate(model, val_loader, device, pos_weight)
    final_metrics.update({"global_step": global_step, "train_loss_recent": float(np.mean(running_loss[-grad_accum:]))})
    if save_last:
        save_checkpoint(model, LAST_DIR, config, {"run_meta": run_meta, "validation": final_metrics})

    summary = {
        "status": "pass",
        "run_meta": run_meta,
        "best_validation": best_metrics,
        "final_validation": final_metrics,
        "best_checkpoint": str(BEST_DIR),
        "last_checkpoint": str(LAST_DIR) if save_last else None,
    }
    with (METRICS_DIR / "train_qlora_summary.json").open("w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
