#!/usr/bin/env python3
import copy
import random


def feature_dropout(patient_state, p=0.05, seed=None):
    rng = random.Random(seed)
    state = copy.deepcopy(patient_state)
    for section in ("危险因素", "血液指标"):
        for key, value in list(state[section].items()):
            if value is not None and rng.random() < p:
                state[section][key] = None
    for key, value in list(state["颈动脉超声结构化"].items()):
        if isinstance(value, dict) and value.get("status") != "missing" and rng.random() < p:
            state["颈动脉超声结构化"][key] = {"value": None, "status": "missing"}
    return state


def group_dropout(patient_state, lab_p=0.10, carotid_p=0.05, seed=None):
    rng = random.Random(seed)
    state = copy.deepcopy(patient_state)
    if rng.random() < lab_p:
        for key in state["血液指标"]:
            state["血液指标"][key] = None
    if rng.random() < carotid_p:
        for key in state["颈动脉超声结构化"]:
            state["颈动脉超声结构化"][key] = {"value": None, "status": "missing"}
    return state
