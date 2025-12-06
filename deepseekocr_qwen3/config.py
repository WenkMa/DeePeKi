from dataclasses import dataclass

@dataclass
class VisionConfig:
    image_size: int = 1024
    hidden_size: int = 1024        # DeepSeek-OCR encoder output dim
    max_vision_tokens: int = 256

@dataclass
class QwenConfig:
    model_name: str = "Qwen/Qwen3-0.6B-Base"

@dataclass
class TrainConfig:
    train_jsonl: str = "data/train.jsonl"
    output_dir: str = "outputs/deepseekocr_qwen3_sft"
    batch_size: int = 2
    lr: float = 2e-5
    num_epochs: int = 1
    max_samples: int | None = None
    gradient_accumulation_steps: int = 8
    bf16: bool = True
