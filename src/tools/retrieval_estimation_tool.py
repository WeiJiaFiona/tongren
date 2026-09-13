#!/usr/bin/env python3
import copy

from src.memory.patient_memory import LAB_FIELDS, PatientMemory


def estimate_missing_labs(patient_state, train_memory, k=20, exclude_self=True):
    memory = train_memory if isinstance(train_memory, PatientMemory) else PatientMemory(train_memory)
    updated = copy.deepcopy(patient_state)
    estimated = []
    exclude_id = patient_state["患者ID"] if exclude_self else None
    for lab in LAB_FIELDS:
        if updated["血液指标"].get(lab) is not None:
            continue
        result = memory.estimate_lab(patient_state, lab, k=k, exclude_patient_id=exclude_id)
        if result is None:
            continue
        updated["血液指标"][lab] = {
            "value": result["value"],
            "status": "estimated",
            "method": "retrieval_assisted_feature_estimation",
            "k": result["k"],
        }
        estimated.append(lab)
    updated.setdefault("估计信息", []).extend(estimated)
    return updated
