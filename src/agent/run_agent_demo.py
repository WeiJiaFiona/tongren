#!/usr/bin/env python3
import json
import os
import sys
from pathlib import Path

import torch
from transformers import AutoTokenizer


ROOT = Path(os.environ.get("TONGREN_PROJECT_ROOT", "/public_bme/home/jiawei2022/tongren_bme_transition"))
sys.path.insert(0, str(ROOT / "code"))

from src.agent.controller import ClinicalAgentController
from src.data.serialize_patient import serialize_patient_state
from src.memory.patient_memory import PatientMemory
from src.models.calibration import PlattCalibrator
from src.tools.risk_assessment_tool import RiskAssessmentTool
from src.train.calibrate import load_selected_model, read_config, write_json
from src.train.evaluate import FROZEN_CONFIG_PATH, resolve_transition_path
from src.train.train_qlora import input_device


REPORT_DIR = ROOT / "outputs/reports"


def read_json(path):
    with Path(path).open("r", encoding="utf-8") as f:
        return json.load(f)


def pick_patient(patients):
    patient_id = os.environ.get("DEMO_PATIENT_ID")
    if patient_id:
        target = int(patient_id)
        for patient in patients:
            if int(patient["患者ID"]) == target:
                return patient
        raise ValueError(f"DEMO_PATIENT_ID={target} not found in test patients.")
    index = int(os.environ.get("DEMO_PATIENT_INDEX", "0"))
    if index < 0 or index >= len(patients):
        raise IndexError(f"DEMO_PATIENT_INDEX={index} out of range for n={len(patients)}.")
    return patients[index]


class BaichuanCheckpointPredictor:
    def __init__(self, model, tokenizer, device, max_length):
        self.model = model
        self.tokenizer = tokenizer
        self.device = device
        self.max_length = max_length

    @torch.no_grad()
    def __call__(self, patient_state):
        messages = [
            {"role": "system", "content": "你是临床脑梗标签识别模型。"},
            {"role": "user", "content": serialize_patient_state(patient_state)},
        ]
        text = self.tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        encoded = self.tokenizer(
            [text],
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=self.max_length,
        )
        encoded = {k: v.to(self.device) for k, v in encoded.items()}
        self.model.eval()
        logit = self.model(**encoded)["logits"][0]
        return float(logit.detach().float().cpu().item())


def load_predictor(config, frozen_config):
    checkpoint_dir = resolve_transition_path(frozen_config["selected_checkpoint"])
    calibrator_path = resolve_transition_path(frozen_config["calibrator_path"])
    max_length = int(frozen_config.get("max_length", os.environ.get("TRAIN_MAX_LENGTH", "512")))

    tokenizer = AutoTokenizer.from_pretrained(config["paths"]["model_dir"], trust_remote_code=True, use_fast=False)
    tokenizer.padding_side = "right"
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model, device = load_selected_model(config, checkpoint_dir)
    calibrator = PlattCalibrator.load(calibrator_path)
    predictor = BaichuanCheckpointPredictor(model, tokenizer, device, max_length)
    return predictor, calibrator, max_length


def maybe_load_memory(config):
    memory_path = os.environ.get("AGENT_MEMORY_PATH")
    if memory_path:
        path = Path(memory_path)
    else:
        path = Path(config["paths"]["train_core"])
    if not path.exists():
        print(f"memory_unavailable={path}")
        return None
    return PatientMemory.from_jsonl(path)


def main():
    if not FROZEN_CONFIG_PATH.exists():
        raise FileNotFoundError(f"Missing frozen config: {FROZEN_CONFIG_PATH}. Run calibration first.")

    config = read_config()
    frozen_config = read_json(FROZEN_CONFIG_PATH)
    test_patients_path = ROOT / "code/src/data/test/patient_features_test.json"
    patients = read_json(test_patients_path)
    raw_patient = pick_patient(patients)

    predictor, calibrator, max_length = load_predictor(config, frozen_config)
    threshold = float(frozen_config["threshold"])
    delta = float(os.environ.get("AGENT_DELTA", "0.02"))
    k = int(os.environ.get("AGENT_K", "20"))
    train_memory = maybe_load_memory(config)

    risk_tool = RiskAssessmentTool(predictor=predictor, calibrator=calibrator, threshold=threshold)
    agent = ClinicalAgentController(
        risk_tool=risk_tool,
        train_memory=train_memory,
        threshold=threshold,
        delta=delta,
        k=k,
    )
    report = agent.run(raw_patient)
    report["配置"] = {
        "selected_checkpoint": frozen_config["selected_checkpoint"],
        "calibrator_path": frozen_config["calibrator_path"],
        "threshold": threshold,
        "delta": delta,
        "k": k,
        "max_length": max_length,
        "patient_source": str(test_patients_path),
    }

    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    report_path = REPORT_DIR / f"agent_demo_patient_{raw_patient['患者ID']}.json"
    write_json(report_path, report)
    print(json.dumps({"status": "pass", "report_path": str(report_path), "report": report}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
