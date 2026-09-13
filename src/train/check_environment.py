#!/usr/bin/env python3
import importlib.util
import json
from pathlib import Path


ROOT = Path("/public_bme/home/jiawei2022/tongren_bme_transition")
OUT = ROOT / "outputs/metrics/environment_check_tongren_m1.json"
MODEL_DIR = ROOT / "code/model/Baichuan-M1-14B-Instruct"


def module_version(name):
    spec = importlib.util.find_spec(name)
    if spec is None:
        return {"available": False, "version": None}
    module = __import__(name)
    return {"available": True, "version": getattr(module, "__version__", "unknown")}


def main():
    report = {"model_dir": str(MODEL_DIR), "modules": {}}
    for name in ["torch", "transformers", "peft", "bitsandbytes", "flash_attn", "accelerate", "safetensors"]:
        report["modules"][name] = module_version(name)

    if report["modules"]["torch"]["available"]:
        import torch

        report["cuda_available"] = torch.cuda.is_available()
        report["cuda_device_count"] = torch.cuda.device_count()
        report["bf16_supported"] = torch.cuda.is_available() and torch.cuda.is_bf16_supported()
        report["devices"] = [
            torch.cuda.get_device_name(i) for i in range(torch.cuda.device_count())
        ] if torch.cuda.is_available() else []
    else:
        report["cuda_available"] = False
        report["cuda_device_count"] = 0
        report["bf16_supported"] = False
        report["devices"] = []

    required = ["torch", "transformers", "peft", "bitsandbytes", "flash_attn"]
    report["status"] = "pass" if all(report["modules"][m]["available"] for m in required) and report["cuda_available"] else "fail"
    OUT.parent.mkdir(parents=True, exist_ok=True)
    with OUT.open("w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=2)
    print(json.dumps(report, ensure_ascii=False, indent=2))
    if report["status"] != "pass":
        raise SystemExit(1)


if __name__ == "__main__":
    main()
