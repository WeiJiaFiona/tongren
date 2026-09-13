#!/usr/bin/env python3
import copy


def occlude_feature(patient_state, section, key):
    state = copy.deepcopy(patient_state)
    if section in ("危险因素", "血液指标"):
        state[section][key] = None
    elif section == "颈动脉超声结构化":
        state[section][key] = {"value": None, "status": "missing"}
    return state


def feature_occlusion_attribution(patient_state, predict_fn, candidates):
    full = predict_fn(patient_state)
    rows = []
    for section, key in candidates:
        occluded = occlude_feature(patient_state, section, key)
        prob = predict_fn(occluded)
        rows.append({"section": section, "指标": key, "p_full": full, "p_without": prob, "delta_p": full - prob})
    return sorted(rows, key=lambda x: abs(x["delta_p"]), reverse=True)
