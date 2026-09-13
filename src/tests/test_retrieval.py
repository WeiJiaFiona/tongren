#!/usr/bin/env python3
from src.memory.patient_memory import PatientMemory
from src.tools.retrieval_estimation_tool import estimate_missing_labs


def test_retrieval_no_self_required():
    def row(pid, age, sex, bmi, lab):
        return {
            "患者ID": pid,
            "危险因素": {"年龄": age, "性别": sex, "体质指数(BMI)": bmi},
            "血液指标": {"高密度脂蛋白胆固醇": lab, "低密度脂蛋白胆固醇": 2.0},
            "颈动脉超声结构化": {"CPS报告值": {"value": 1, "status": "observed"}},
        }

    rows = [
        row(1, 60, "男", 24.0, 1.0),
        row(2, 61, "男", 24.5, 1.1),
        row(3, 62, "男", 25.0, 1.2),
        row(4, 63, "男", 25.5, 1.3),
    ]
    query = row(1, 60, "男", 24.0, None)
    out = estimate_missing_labs(query, PatientMemory(rows), k=3)
    assert out["血液指标"]["高密度脂蛋白胆固醇"]["status"] == "estimated"
