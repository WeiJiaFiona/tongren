#!/usr/bin/env python3
def build_lora_config(r=16, alpha=32, dropout=0.05):
    from peft import LoraConfig

    return LoraConfig(
        r=r,
        lora_alpha=alpha,
        lora_dropout=dropout,
        bias="none",
        task_type="CAUSAL_LM",
        target_modules=["W_pack", "o_proj"],
    )


def attach_lora(model, config=None, prepare_kbit=True):
    from peft import get_peft_model, prepare_model_for_kbit_training

    if prepare_kbit:
        model = prepare_model_for_kbit_training(model, use_gradient_checkpointing=True)
    return get_peft_model(model, config or build_lora_config())
