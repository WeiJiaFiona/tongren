#!/usr/bin/env python3
import json
import os
import shutil
import sys
from pathlib import Path

import torch
from torch.optim import AdamW


ROOT = Path("/public_bme/home/jiawei2022/tongren_bme_transition")
sys.path.insert(0, str(ROOT / "code"))

from src.data.serialize_patient import serialize_patient_state
from src.models.baichuan_risk_classifier import BaichuanRiskClassifier, load_baichuan_causal_lm
from src.models.qlora import attach_lora


MODEL_DIR = ROOT / "code/model/Baichuan-M1-14B-Instruct"
DATA_PATH = ROOT / "code/src/data/train_core/patient_state_train_core.jsonl"
OUT_DIR = ROOT / "outputs/checkpoints/smoke_baichuan"
METRICS_PATH = ROOT / "outputs/metrics/baichuan_smoke_test.json"

MAX_LENGTH = int(os.environ.get("SMOKE_MAX_LENGTH", "512"))
BATCH_SIZE = int(os.environ.get("SMOKE_BATCH_SIZE", "2"))
SKIP_RELOAD = os.environ.get("SMOKE_SKIP_RELOAD", "0") == "1"


def read_jsonl(path):
    with path.open("r", encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def pick_smoke_batch(states, batch_size):
    positives = [s for s in states if s["标签"]["半年内脑梗"] == 1]
    negatives = [s for s in states if s["标签"]["半年内脑梗"] == 0]
    if not positives or not negatives:
        raise RuntimeError("Need at least one positive and one negative sample for smoke test.")
    batch = [negatives[0], positives[0]]
    for state in states:
        if len(batch) >= batch_size:
            break
        if state["患者ID"] not in {b["患者ID"] for b in batch}:
            batch.append(state)
    return batch[:batch_size]


def encode_batch(tokenizer, states):
    texts = []
    for state in states:
        messages = [
            {"role": "system", "content": "你是临床脑梗标签识别模型。"},
            {"role": "user", "content": serialize_patient_state(state)},
        ]
        texts.append(tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True))
    encoded = tokenizer(
        texts,
        return_tensors="pt",
        padding=True,
        truncation=True,
        max_length=MAX_LENGTH,
    )
    labels = torch.tensor([s["标签"]["半年内脑梗"] for s in states], dtype=torch.float32)
    return encoded, labels


def input_device(model):
    if hasattr(model, "hf_device_map") and model.hf_device_map:
        for device in model.hf_device_map.values():
            if isinstance(device, str) and device.startswith("cuda"):
                return torch.device(device)
    if torch.cuda.is_available():
        return torch.device("cuda:0")
    return torch.device("cpu")


def trainable_parameter_report(model):
    trainable = 0
    total = 0
    trainable_names = []
    for name, param in model.named_parameters():
        n = param.numel()
        total += n
        if param.requires_grad:
            trainable += n
            if len(trainable_names) < 30:
                trainable_names.append(name)
    return {"trainable": trainable, "total": total, "trainable_names_head": trainable_names}


def load_tokenizer():
    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(MODEL_DIR, trust_remote_code=True, use_fast=False)
    tokenizer.padding_side = "right"
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    return tokenizer


def forward_logits(classifier, encoded, labels=None):
    device = input_device(classifier.causal_lm)
    encoded = {k: v.to(device) for k, v in encoded.items()}
    if labels is not None:
        labels = labels.to(device)
    return classifier(**encoded, labels=labels)


def main():
    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()

    states = read_jsonl(DATA_PATH)
    batch_states = pick_smoke_batch(states, BATCH_SIZE)
    tokenizer = load_tokenizer()
    encoded, labels = encode_batch(tokenizer, batch_states)

    print("loading_base_model=started")
    causal_lm = load_baichuan_causal_lm(MODEL_DIR)
    causal_lm.config.use_cache = False
    print("loading_base_model=done")

    print("attaching_lora=started")
    causal_lm = attach_lora(causal_lm)
    causal_lm.print_trainable_parameters()
    print("attaching_lora=done")

    classifier = BaichuanRiskClassifier(causal_lm, hidden_size=5120, dropout=0.1)
    classifier.train()
    if hasattr(classifier.causal_lm, "enable_input_require_grads"):
        classifier.causal_lm.enable_input_require_grads()
    trainable_report = trainable_parameter_report(classifier)

    optimizer = AdamW([p for p in classifier.parameters() if p.requires_grad], lr=1e-4)

    out = forward_logits(classifier, encoded, labels=labels)
    if out["logits"].shape[0] != len(batch_states):
        raise RuntimeError(f"Unexpected logits shape: {tuple(out['logits'].shape)}")
    loss = out["loss"]
    loss.backward()
    optimizer.step()
    optimizer.zero_grad(set_to_none=True)

    classifier.eval()
    with torch.no_grad():
        logits_after_step = forward_logits(classifier, encoded)["logits"].detach().float().cpu()
        probs_after_step = torch.sigmoid(logits_after_step)

    if OUT_DIR.exists():
        shutil.rmtree(OUT_DIR)
    adapter_dir = OUT_DIR / "lora_adapter"
    classifier_head_path = OUT_DIR / "classifier_head.pt"
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    classifier.causal_lm.save_pretrained(adapter_dir)
    torch.save(classifier.classifier.state_dict(), classifier_head_path)

    reload_max_abs_diff = None
    if not SKIP_RELOAD:
        from peft import PeftModel

        del classifier
        del causal_lm
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        print("reload_check=started")
        base = load_baichuan_causal_lm(MODEL_DIR)
        reloaded_lm = PeftModel.from_pretrained(base, adapter_dir, is_trainable=True)
        reloaded = BaichuanRiskClassifier(reloaded_lm, hidden_size=5120, dropout=0.1)
        reloaded.classifier.load_state_dict(torch.load(classifier_head_path, map_location="cpu"))
        reloaded.eval()
        with torch.no_grad():
            reloaded_logits = forward_logits(reloaded, encoded)["logits"].detach().float().cpu()
        reload_max_abs_diff = float((logits_after_step - reloaded_logits).abs().max().item())
        print("reload_check=done")

    peak_memory_gb = None
    if torch.cuda.is_available():
        peak_memory_gb = torch.cuda.max_memory_allocated() / 1024**3

    metrics = {
        "status": "pass",
        "model_dir": str(MODEL_DIR),
        "data_path": str(DATA_PATH),
        "batch_size": len(batch_states),
        "patient_ids": [s["患者ID"] for s in batch_states],
        "labels": labels.tolist(),
        "max_length": MAX_LENGTH,
        "input_shape": list(encoded["input_ids"].shape),
        "loss": float(loss.detach().float().cpu().item()),
        "probabilities_after_step": [float(x) for x in probs_after_step.tolist()],
        "adapter_dir": str(adapter_dir),
        "classifier_head_path": str(classifier_head_path),
        "reload_max_abs_logit_diff": reload_max_abs_diff,
        "reload_checked": not SKIP_RELOAD,
        "gpu_peak_memory_gb": peak_memory_gb,
        "trainable_parameters": trainable_report,
        "checks": {
            "tokenizer_local_load": True,
            "four_bit_base_model_load": True,
            "qlora_inserted": True,
            "decoder_last_hidden_state_forward": True,
            "classifier_forward": True,
            "bce_backward": True,
            "optimizer_step": True,
            "adapter_and_head_saved": True,
            "reload_probability_consistency": SKIP_RELOAD or (reload_max_abs_diff is not None and reload_max_abs_diff < 1e-4),
        },
    }
    if peak_memory_gb is not None:
        metrics["checks"]["gpu_peak_memory_lt_48gb"] = peak_memory_gb < 48
    if not all(metrics["checks"].values()):
        metrics["status"] = "fail"

    METRICS_PATH.parent.mkdir(parents=True, exist_ok=True)
    with METRICS_PATH.open("w", encoding="utf-8") as f:
        json.dump(metrics, f, ensure_ascii=False, indent=2)
    print(json.dumps(metrics, ensure_ascii=False, indent=2))
    if metrics["status"] != "pass":
        raise SystemExit(1)


if __name__ == "__main__":
    main()
