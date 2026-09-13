#!/usr/bin/env python3
from src.tools.curation_tool import curate


def test_curation_basic_fields():
    patient = {
        "患者ID": 1,
        "颈动脉超声报告": {
            "超声描述": "CIMT：『1.2』 CPS：『3』\n右颈总A  『正常 正常 正常』 『1.1』 『0.84』 『99』 『可见』\n右颈内A  『正常 正常 正常』 『0.8』 『0.70』 『258』 『可见』\n右侧颈内动脉起始部狭窄率约55%，大小约10.0×4.0mm，少许点状微泡增强，溃疡面。",
            "超声结论": "右侧颈动脉粥样斑块形成",
        },
    }
    out = curate(patient)["颈动脉超声结构化"]
    assert out["最大颈总动脉IMT(mm)"]["value"] == 1.2
    assert out["CPS报告值"]["value"] == 3
    assert out["颈动脉斑块存在"]["value"] is True
    assert out["狭窄分级"]["value"] == "中度"
    assert out["CEUS分级"]["value"] == 2
    assert out["溃疡"]["value"] is True
