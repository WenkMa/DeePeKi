from typing import Dict, Any, List

import torch
import torch.nn as nn
from transformers import PreTrainedModel, PreTrainedTokenizer

from .vision_encoder import DeepSeekVisionEncoder, DeepSeekVisionCfg


class DeepSeekOCRQwen3(nn.Module):
    """Simple VLM: DeepSeek-OCR encoder + projector + Qwen3-0.6B.

    Text uses a special token `<image>` as a placeholder. At training time
    we replace that single token with a sequence of vision tokens.
    """

    def __init__(
        self,
        qwen_model: PreTrainedModel,
        tokenizer: PreTrainedTokenizer,
        device: str = "cuda",
    ) -> None:
        super().__init__()
        self.llm = qwen_model
        self.tokenizer = tokenizer
        self.device = torch.device(device)

        hidden_size = self.llm.config.hidden_size

        # Vision encoder and projector
        self.vision_cfg = DeepSeekVisionCfg()
        self.vision_encoder = DeepSeekVisionEncoder(
            self.vision_cfg,
            device=self.device,
            dtype=torch.bfloat16,
        )
        self.vision_to_llm = nn.Linear(
            self.vision_cfg.hidden_size,
            hidden_size,
            bias=False,
        ).to(self.device, dtype=torch.bfloat16)

        # Add special token for image placeholder
        if "<image>" not in self.tokenizer.get_vocab():
            self.tokenizer.add_special_tokens(
                {"additional_special_tokens": ["<image>"]}
            )
            self.llm.resize_token_embeddings(len(self.tokenizer))
        self.image_token_id = self.tokenizer.convert_tokens_to_ids("<image>")

    @torch.no_grad()
    def encode_image(self, image) -> torch.Tensor:
        """Encode a PIL image or path to vision embeddings.

        Returns:
            Tensor of shape [T_v, hidden_size]
        """
        vision_tokens = self.vision_encoder(image)          # [T_v, 1024]
        vision_embeds = self.vision_to_llm(vision_tokens)   # [T_v, hidden]
        return vision_embeds

    def _expand_with_image_tokens(
        self,
        input_ids: torch.Tensor,
        labels: torch.Tensor,
        attention_mask: torch.Tensor,
        vision_embeds: torch.Tensor,
    ):
        """Replace `<image>` token with vision embeddings for one sample."""
        # [L] -> [L, hidden]
        input_embeds = self.llm.get_input_embeddings()(
            input_ids.unsqueeze(0)
        )[0]

        image_pos = (input_ids == self.image_token_id).nonzero(as_tuple=False)
        if len(image_pos) == 0:
            raise ValueError("Sample has no <image> token; every sample must have one.")
        image_pos = int(image_pos[0].item())

        before_embeds = input_embeds[:image_pos]
        after_embeds = input_embeds[image_pos + 1 :]

        before_labels = labels[:image_pos]
        after_labels = labels[image_pos + 1 :]

        before_attn = attention_mask[:image_pos]
        after_attn = attention_mask[image_pos + 1 :]

        T_v = vision_embeds.size(0)
        vision_labels = torch.full(
            (T_v,),
            -100,
            dtype=labels.dtype,
            device=labels.device,
        )
        vision_attn = torch.ones(
            T_v,
            dtype=attention_mask.dtype,
            device=attention_mask.device,
        )

        new_embeds = torch.cat(
            [before_embeds, vision_embeds, after_embeds],
            dim=0,
        )
        new_labels = torch.cat(
            [before_labels, vision_labels, after_labels],
            dim=0,
        )
        new_attn = torch.cat(
            [before_attn, vision_attn, after_attn],
            dim=0,
        )

        return new_embeds, new_labels, new_attn

    def forward(self, batch: Dict[str, Any]):
        """Forward pass for a batch.

        batch:
            - input_ids: [B, L]
            - labels: [B, L]
            - attention_mask: [B, L]
            - images: list[PIL.Image.Image] of length B
        """
        input_ids = batch["input_ids"].to(self.device)
        labels = batch["labels"].to(self.device)
        attention_mask = batch["attention_mask"].to(self.device)
        images: List[Any] = batch["images"]

        B, _ = input_ids.shape
        embeds_list = []
        labels_list = []
        attn_list = []

        for i in range(B):
            vision_embeds = self.encode_image(images[i])
            e_i, l_i, a_i = self._expand_with_image_tokens(
                input_ids[i],
                labels[i],
                attention_mask[i],
                vision_embeds,
            )
            embeds_list.append(e_i)
            labels_list.append(l_i)
            attn_list.append(a_i)

        max_len = max(e.size(0) for e in embeds_list)

        def pad_2d(x: torch.Tensor, value: float) -> torch.Tensor:
            pad_len = max_len - x.size(0)
            if pad_len <= 0:
                return x
            pad = x.new_full((pad_len, x.size(1)), value)
            return torch.cat([x, pad], dim=0)

        def pad_1d(x: torch.Tensor, value: float) -> torch.Tensor:
            pad_len = max_len - x.size(0)
            if pad_len <= 0:
                return x
            pad = x.new_full((pad_len,), value)
            return torch.cat([x, pad], dim=0)

        inputs_embeds = torch.stack(
            [pad_2d(e, 0.0) for e in embeds_list],
            dim=0,
        )
        labels_pad = torch.stack(
            [pad_1d(l, -100) for l in labels_list],
            dim=0,
        )
        attn_pad = torch.stack(
            [pad_1d(a, 0) for a in attn_list],
            dim=0,
        )

        outputs = self.llm(
            inputs_embeds=inputs_embeds,
            attention_mask=attn_pad,
            labels=labels_pad,
        )
        return outputs
