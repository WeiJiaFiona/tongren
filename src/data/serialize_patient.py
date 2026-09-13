#!/usr/bin/env python3
MISSING_TEXT = "缺失"


RISK_ORDER = [
    "年龄",
    "性别",
    "体质指数(BMI)",
    "收缩压",
    "舒张压",
    "吸烟状态",
    "饮酒",
    "高血压",
    "糖尿病",
    "脑卒中家族史",
    "血脂异常",
    "缺乏体育活动",
]

LAB_ORDER = [
    "高密度脂蛋白胆固醇",
    "低密度脂蛋白胆固醇",
    "非高密度脂蛋白胆固醇",
    "甘油三酯",
    "载脂蛋白B",
    "脂蛋白(a)",
    "糖化血红蛋白",
    "同型半胱氨酸",
    "纤维蛋白原",
    "小而密低密度脂蛋白胆固醇",
]


def render_value(value):
    if isinstance(value, dict):
        status = value.get("status")
        val = value.get("value")
        if val is None or status == "missing":
            return MISSING_TEXT
        if status == "estimated":
            return f"{val}（检索估计）"
        if status == "derived":
            return f"{val}（计算值）"
        return str(val)
    if value is None:
        return MISSING_TEXT
    return str(value)


def serialize_patient_state(patient_state):
    lines = ["【任务】", "根据以下患者基线临床信息完成半年内脑梗标签预测。", "", "【危险因素】"]
    for key in RISK_ORDER:
        lines.append(f"{key}：{render_value(patient_state['危险因素'].get(key))}")
    lines.extend(["", "【血液指标】"])
    for key in LAB_ORDER:
        lines.append(f"{key}：{render_value(patient_state['血液指标'].get(key))}")
    lines.extend(["", "【颈动脉超声结构化指标】"])
    for key, value in patient_state["颈动脉超声结构化"].items():
        lines.append(f"{key}：{render_value(value)}")
    return "\n".join(lines)
