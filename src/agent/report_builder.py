#!/usr/bin/env python3
def build_report(patient_state, prediction, attribution=None):
    return {
        "患者ID": patient_state["患者ID"],
        "任务": "半年内脑梗标签预测",
        "预测": {
            "原始概率": prediction["raw_probability"],
            "校准概率": prediction["calibrated_probability"],
            "决策阈值": prediction["threshold"],
            "预测标签": prediction["predicted_label"],
        },
        "主要临床证据": attribution or [],
        "缺失信息": [],
        "估计信息": patient_state.get("估计信息", []),
    }
