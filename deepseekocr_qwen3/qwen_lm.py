from typing import Tuple

from transformers import AutoModelForCausalLM, AutoTokenizer
import torch

from .config import QwenConfig


def load_qwen3_lm(cfg: QwenConfig) -> Tuple[AutoModelForCausalLM, AutoTokenizer]:
    tokenizer = AutoTokenizer.from_pretrained(
        cfg.model_name,
        trust_remote_code=True,
    )
    model = AutoModelForCausalLM.from_pretrained(
        cfg.model_name,
        torch_dtype=torch.bfloat16,
        device_map="auto",
        trust_remote_code=True,
    )
    return model, tokenizer
