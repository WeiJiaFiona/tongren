#!/usr/bin/env python3
import re


VESSEL_ALIASES = {
    "颈总A": "CCA",
    "颈内A": "ICA",
    "颈外A": "ECA",
    "椎A": "VA",
}


def field(value=None, status="missing", source=None, evidence_text=None):
    data = {"value": value, "status": status}
    if source is not None:
        data["source"] = source
    if evidence_text is not None:
        data["evidence_text"] = evidence_text
    return data


def text_of(raw_patient):
    report = raw_patient.get("颈动脉超声报告", {})
    parts = []
    for key in ("超声描述", "超声结论"):
        value = report.get(key)
        if value:
            parts.append(str(value))
    return "\n".join(parts)


def parse_float(text):
    try:
        return float(text)
    except Exception:
        return None


def status_from_bool(value, evidence, source="超声报告"):
    if value is None:
        return field()
    return field(bool(value), "observed" if value else "explicitly_absent", source, evidence)


def extract_cimt(report_text):
    match = re.search(r"CIMT[：:]\s*『?(\d+(?:\.\d+)?)", report_text)
    if match:
        value = parse_float(match.group(1))
        return field(value, "observed", "超声描述", match.group(0))
    return field()


def extract_cps(report_text):
    match = re.search(r"CPS[：:]\s*『?(\d+(?:\.\d+)?)", report_text)
    if match:
        value = parse_float(match.group(1))
        return field(int(value) if value is not None and value.is_integer() else value, "observed", "超声描述", match.group(0))
    return field()


def imt_state(value):
    if value is None:
        return field()
    if value < 1.0:
        return field("正常", "derived", "CIMT/CCA-IMT")
    if value < 1.5:
        return field("IMT增厚", "derived", "CIMT/CCA-IMT")
    return field("达到斑块厚度判据之一", "derived", "CIMT/CCA-IMT")


def extract_table_rows(report_text):
    rows = []
    pattern = re.compile(
        r"(?P<side>[左右])(?P<vessel>颈总A|颈内A|颈外A|椎A)\s+"
        r"(?P<body>[^\n]+)"
    )
    for match in pattern.finditer(report_text):
        body = match.group("body")
        nums = [parse_float(x) for x in re.findall(r"『?(\d+(?:\.\d+)?)』?", body)]
        plaque = "可见" if "可见" in body else "未见" if "未见" in body else None
        row = {
            "侧别": match.group("side"),
            "血管类型": VESSEL_ALIASES.get(match.group("vessel"), match.group("vessel")),
            "IMT": nums[-3] if len(nums) >= 3 else None,
            "RI": nums[-2] if len(nums) >= 2 else None,
            "PSV": nums[-1] if len(nums) >= 1 else None,
            "斑块可见": plaque,
            "原文行": match.group(0),
        }
        rows.append(row)
    return rows


def max_cca_imt(vessel_rows):
    values = [row["IMT"] for row in vessel_rows if row["血管类型"] == "CCA" and row["IMT"] is not None]
    return max(values) if values else None


def extract_plaque_sizes(report_text):
    sizes = []
    for match in re.finditer(r"大小约?『?(\d+(?:\.\d+)?)』?\s*[×xX*]\s*『?(\d+(?:\.\d+)?)』?\s*mm", report_text):
        a = parse_float(match.group(1))
        b = parse_float(match.group(2))
        if a is not None and b is not None:
            sizes.append({"长度": max(a, b), "厚度": min(a, b), "证据": match.group(0)})
    return sizes


def extract_max_percent(report_text):
    matches = list(re.finditer(r"(?:狭窄率)?[＜<约\s]*(\d{1,3})(?:\s*[-~]\s*\d{1,3})?\s*%", report_text))
    values = []
    for match in matches:
        value = parse_float(match.group(1))
        if value is not None:
            values.append((value, match.group(0)))
    if not values:
        if "闭塞" in report_text:
            return field(100, "observed", "超声报告", "闭塞")
        return field()
    value, evidence = max(values, key=lambda item: item[0])
    return field(value, "observed", "超声报告", evidence)


def stenosis_grade(stenosis_value):
    value = stenosis_value.get("value")
    if value is None:
        return field()
    if value >= 100:
        grade = "闭塞"
    elif value >= 70:
        grade = "重度"
    elif value >= 50:
        grade = "中度"
    else:
        grade = "轻度"
    return field(grade, "derived", "最大颈动脉狭窄率(%)")


def max_ica_psv(vessel_rows, report_text):
    local = re.search(r"狭窄处最高流速约?(\d+(?:\.\d+)?)\s*cm/s", report_text)
    if local:
        return field(parse_float(local.group(1)), "observed", "超声描述", local.group(0))
    values = [row["PSV"] for row in vessel_rows if row["血管类型"] == "ICA" and row["PSV"] is not None]
    if values:
        return field(max(values), "observed", "超声表格")
    return field()


def ica_cca_ratio(vessel_rows):
    ratios = []
    by_side = {"左": {}, "右": {}}
    for row in vessel_rows:
        by_side.setdefault(row["侧别"], {})[row["血管类型"]] = row
    for side, rows in by_side.items():
        ica = rows.get("ICA", {}).get("PSV")
        cca = rows.get("CCA", {}).get("PSV")
        if ica is not None and cca not in (None, 0):
            ratios.append(ica / cca)
    if not ratios:
        return field()
    return field(round(max(ratios), 4), "derived", "ICA PSV / CCA PSV")


def ceus_grade(report_text):
    if not any(term in report_text for term in ("SonoVue", "超声造影", "CEUS", "微泡")):
        return field()
    if re.search(r"未见.*(?:增强|微泡显影)|未见明显.*(?:增强|微泡)", report_text):
        return field(1, "observed", "超声描述", "CEUS无增强")
    if any(term in report_text for term in ("大量", "广泛", "弥漫")):
        return field(3, "observed", "超声描述", "大量/广泛/弥漫增强")
    if any(term in report_text for term in ("少许", "少量", "点状", "基底部", "肩部")):
        return field(2, "observed", "超声描述", "少许/少量/基底部/肩部微泡增强")
    return field()


def curate(raw_patient):
    report_text = text_of(raw_patient)
    quality_flags = []
    if not report_text:
        quality_flags.append("missing_ultrasound_report")

    vessel_rows = extract_table_rows(report_text)
    cimt = extract_cimt(report_text)
    if cimt["value"] is None:
        cca = max_cca_imt(vessel_rows)
        cimt = field(cca, "derived", "左右CCA IMT最大值") if cca is not None else field()

    cps = extract_cps(report_text)
    plaque_present = None
    if re.search(r"斑块形成|斑块.*可见|粥样斑块", report_text):
        plaque_present = True
    elif "斑块" in report_text and "未见" in report_text:
        plaque_present = False

    sizes = extract_plaque_sizes(report_text)
    max_len = max((s["长度"] for s in sizes), default=None)
    max_thick = max((s["厚度"] for s in sizes), default=None)
    stenosis = extract_max_percent(report_text)
    ica_psv = max_ica_psv(vessel_rows, report_text)

    multifocal = bool(re.search(r"数个|多个|多发", report_text))
    plaque_count_state = "多发" if multifocal else "单发" if plaque_present else None
    bilateral = bool(re.search(r"双侧.*斑块|左右.*斑块", report_text)) if plaque_present else None

    echo_terms = [term for term in ("低回声", "等回声", "强回声", "不均质回声", "混合回声") if term in report_text]
    surface = "不规则" if "不规则" in report_text else "规则" if "规则" in report_text else None

    structured = {
        "最大颈总动脉IMT(mm)": cimt,
        "IMT临床状态": imt_state(cimt["value"]),
        "CPS报告值": cps,
        "颈动脉斑块存在": status_from_bool(plaque_present, "斑块形成/可见/未见"),
        "斑块数量状态": field(plaque_count_state, "derived" if plaque_count_state else "missing", "报告文本"),
        "双侧颈动脉斑块": status_from_bool(bilateral, "双侧斑块描述"),
        "最大颈动脉斑块长度(mm)": field(max_len, "observed", "超声描述") if max_len is not None else field(),
        "最大颈动脉斑块厚度(mm)": field(max_thick, "observed", "超声描述") if max_thick is not None else field(),
        "最大颈动脉狭窄率(%)": stenosis,
        "狭窄分级": stenosis_grade(stenosis),
        "最大颈内动脉PSV(cm/s)": ica_psv,
        "最大ICA/CCA PSV比值": ica_cca_ratio(vessel_rows),
        "斑块回声类型": field("、".join(echo_terms), "observed", "超声描述") if echo_terms else field(),
        "斑块回声不均匀": status_from_bool("不均匀" in report_text or "不均质" in report_text, "不均匀/不均质"),
        "斑块表面状态": field(surface, "observed", "超声描述") if surface else field(),
        "纤维帽缺损": status_from_bool(
            True if "纤维帽" in report_text and "缺损" in report_text and "未见缺损" not in report_text else False if "纤维帽未见缺损" in report_text or "纤维帽『未见缺损』" in report_text else None,
            "纤维帽缺损/未见缺损",
        ),
        "溃疡": status_from_bool(
            True if "溃疡" in report_text and "未见溃疡" not in report_text else False if "未见溃疡" in report_text else None,
            "溃疡/未见溃疡",
        ),
        "CEUS分级": ceus_grade(report_text),
    }

    lesion_records = [
        {"长度": item["长度"], "厚度": item["厚度"], "原文证据span": item["证据"]}
        for item in sizes
    ]

    if not vessel_rows:
        quality_flags.append("no_vessel_table_rows_parsed")
    if plaque_present and not sizes:
        quality_flags.append("plaque_present_without_size")

    return {
        "颈动脉超声结构化": structured,
        "颈动脉病灶级记录": lesion_records,
        "颈动脉血管表格记录": vessel_rows,
        "质量标记": quality_flags,
    }
