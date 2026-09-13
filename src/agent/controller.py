#!/usr/bin/env python3
import sys
from pathlib import Path

ROOT = Path("/public_bme/home/jiawei2022/tongren_bme_transition")
sys.path.insert(0, str(ROOT / "code"))

from src.agent.report_builder import build_report
from src.agent.state import AgentState
from src.tools.curation_tool import curate
from src.tools.retrieval_estimation_tool import estimate_missing_labs


class ClinicalAgentController:
    def __init__(self, risk_tool, train_memory=None, threshold=0.5, delta=0.02, k=20):
        self.risk_tool = risk_tool
        self.train_memory = train_memory
        self.threshold = threshold
        self.delta = delta
        self.k = k

    def build_patient_state(self, raw_patient):
        curated = curate(raw_patient)
        return {
            "患者ID": raw_patient["患者ID"],
            "标签": raw_patient.get("标签", {}),
            "危险因素": raw_patient["危险因素"],
            "血液指标": raw_patient["血液指标"],
            "颈动脉超声结构化": curated["颈动脉超声结构化"],
            "颈动脉病灶级记录": curated["颈动脉病灶级记录"],
            "质量标记": curated["质量标记"],
        }

    def has_retrievable_lab_missing(self, patient_state):
        return any(value is None for value in patient_state["血液指标"].values())

    def run(self, raw_patient):
        state = AgentState(raw_patient=raw_patient)
        state.route.append("OBSERVE")
        state.patient_state = self.build_patient_state(raw_patient)
        state.route.append("CURATE")
        state.first_prediction = self.risk_tool.predict(state.patient_state)
        state.route.append("PREDICT")

        p1 = state.first_prediction["calibrated_probability"]
        if (
            self.train_memory is not None
            and abs(p1 - self.threshold) < self.delta
            and self.has_retrievable_lab_missing(state.patient_state)
        ):
            state.patient_state = estimate_missing_labs(state.patient_state, self.train_memory, k=self.k)
            state.estimated_features = state.patient_state.get("估计信息", [])
            state.route.append("RETRIEVE/ESTIMATE")
            state.final_prediction = self.risk_tool.predict(state.patient_state)
            state.route.append("RE-PREDICT")
        else:
            state.final_prediction = state.first_prediction
        state.route.append("FINAL")
        report = build_report(state.patient_state, state.final_prediction)
        report["route"] = state.route
        return report
