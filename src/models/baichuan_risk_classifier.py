#!/usr/bin/env python3
import torch
from torch import nn


class BaichuanRiskClassifier(nn.Module):
    def __init__(self, causal_lm, hidden_size=5120, dropout=0.1):
        super().__init__()
        self.causal_lm = causal_lm
        self.classifier = nn.Sequential(nn.Dropout(dropout), nn.Linear(hidden_size, 1))
        self.classifier.float()

    def decoder(self):
        base = self.causal_lm.get_base_model() if hasattr(self.causal_lm, "get_base_model") else self.causal_lm
        return base.model

    def forward(self, input_ids, attention_mask, labels=None):
        outputs = self.decoder()(input_ids=input_ids, attention_mask=attention_mask, use_cache=False, return_dict=True)
        hidden = outputs.last_hidden_state
        last_index = attention_mask.sum(dim=1) - 1
        batch_idx = torch.arange(hidden.shape[0], device=hidden.device)
        pooled = hidden[batch_idx, last_index].float()
        logits = self.classifier(pooled).squeeze(-1)
        result = {"logits": logits}
        if labels is not None:
            result["loss"] = nn.functional.binary_cross_entropy_with_logits(logits, labels.float())
        return result


def load_baichuan_causal_lm(model_dir, max_gpu_memory="45GiB", max_cpu_memory="240GiB"):
    import torch
    from transformers import AutoModelForCausalLM, BitsAndBytesConfig

    device_map = {"": 0} if torch.cuda.is_available() else None
    max_memory = {0: max_gpu_memory, "cpu": max_cpu_memory} if torch.cuda.is_available() else None
    quant_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16,
        bnb_4bit_use_double_quant=True,
    )
    return AutoModelForCausalLM.from_pretrained(
        model_dir,
        trust_remote_code=True,
        quantization_config=quant_config,
        torch_dtype=torch.bfloat16,
        use_cache=False,
        device_map=device_map,
        max_memory=max_memory,
        low_cpu_mem_usage=True,
        local_files_only=True,
        attn_implementation="flash_attention_2",
    )
