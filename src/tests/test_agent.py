#!/usr/bin/env python3
from src.agent.controller import ClinicalAgentController


class DummyRiskTool:
    def predict(self, patient_state):
        return {"raw_probability": 0.1, "calibrated_probability": 0.1, "threshold": 0.5, "predicted_label": 0}


def test_agent_route_without_retrieval():
    raw = {"患者ID": 1, "标签": {}, "危险因素": {}, "血液指标": {}, "颈动脉超声报告": {"超声描述": None, "超声结论": None}}
    report = ClinicalAgentController(DummyRiskTool()).run(raw)
    assert report["route"] == ["OBSERVE", "CURATE", "PREDICT", "FINAL"]
