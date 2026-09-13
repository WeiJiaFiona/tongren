#!/usr/bin/env python3
def test_model_dependencies_importable():
    import importlib.util

    for name in ["torch", "transformers", "peft", "bitsandbytes", "flash_attn"]:
        assert importlib.util.find_spec(name) is not None, f"{name} is required for Baichuan smoke test"
