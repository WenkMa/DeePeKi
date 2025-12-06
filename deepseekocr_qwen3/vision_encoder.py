from dataclasses import dataclass
from typing import Union

import torch
import torch.nn as nn
from PIL import Image

try:
    from deepseek_ocr_encoder import DeepSeekOCREncoder
except ImportError as exc:
    raise ImportError(
        "deepseek-ocr-encoder is not installed. Please run "
        "'pip install deepseek-ocr-encoder' first."
    ) from exc


@dataclass
class DeepSeekVisionCfg:
    model_name: str = "deepseek-ai/DeepSeek-OCR"
    image_size: int = 1024
    hidden_size: int = 1024          # encoder output dim
    max_vision_tokens: int = 256


class DeepSeekVisionEncoder(nn.Module):
    """Thin wrapper around deepseek-ocr-encoder.

    It turns a PIL image or an image path into vision tokens of shape
    [T, hidden_size].
    """

    def __init__(
        self,
        cfg: DeepSeekVisionCfg,
        device: Union[str, torch.device] = "cuda",
        dtype: torch.dtype = torch.bfloat16,
    ) -> None:
        super().__init__()
        self.cfg = cfg
        self.device = torch.device(device)
        self.dtype = dtype

        # This will download weights from Hugging Face on first use.
        self.encoder = DeepSeekOCREncoder.from_pretrained(
            cfg.model_name,
            device=str(self.device),
            dtype=self.dtype,
            freeze=True,
        )

    @torch.no_grad()
    def forward(self, image: Union[Image.Image, str]) -> torch.Tensor:
        """Encode one image into vision tokens.

        Returns:
            vision_tokens: Tensor of shape [T, hidden_size]
        """
        # deepseek-ocr-encoder accepts either path or PIL.Image.
        out = self.encoder(image)
        # out is [1, T, D]
        if out.dim() != 3:
            raise ValueError(f"Unexpected encoder output shape: {tuple(out.shape)}")
        tokens = out[0]

        # Optionally truncate to max_vision_tokens for safety.
        if tokens.size(0) > self.cfg.max_vision_tokens:
            tokens = tokens[: self.cfg.max_vision_tokens]

        return tokens
