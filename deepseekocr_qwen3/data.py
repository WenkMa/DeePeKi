import json
from pathlib import Path
from typing import Any, Dict, List

from PIL import Image
import torch
from torch.utils.data import Dataset
from transformers import PreTrainedTokenizer


class OCRJsonlDataset(Dataset):
    """Simple JSONL dataset.

    Each line is a JSON object:
        {
          "image": "path/to/image.png",
          "text": "User: <image>\nAssistant: ..."
        }
    """

    def __init__(
        self,
        jsonl_path: str,
        tokenizer: PreTrainedTokenizer,
        max_samples: int | None = None,
        max_length: int = 2048,
    ) -> None:
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.items: List[Dict[str, Any]] = []

        p = Path(jsonl_path)
        if not p.is_file():
            raise FileNotFoundError(f"JSONL file not found: {jsonl_path}")

        with p.open("r", encoding="utf-8") as f:
            for i, line in enumerate(f):
                if max_samples is not None and i >= max_samples:
                    break
                line = line.strip()
                if not line:
                    continue
                obj = json.loads(line)
                self.items.append(obj)

    def __len__(self) -> int:
        return len(self.items)

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        obj = self.items[idx]
        img_path = obj["image"]
        text = obj["text"]

        image = Image.open(img_path).convert("RGB")

        encoded = self.tokenizer(
            text,
            return_tensors="pt",
            padding=False,
            truncation=True,
            max_length=self.max_length,
        )
        input_ids = encoded["input_ids"][0]
        attention_mask = encoded["attention_mask"][0]
        labels = input_ids.clone()

        return {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "labels": labels,
            "image": image,
        }


def collate_ocr(batch: List[Dict[str, Any]]) -> Dict[str, Any]:
    images = [item["image"] for item in batch]
    max_len = max(len(item["input_ids"]) for item in batch)

    def pad_1d(x: torch.Tensor, pad_id: int) -> torch.Tensor:
        out = x.new_full((max_len,), pad_id)
        out[: x.size(0)] = x
        return out

    input_ids = torch.stack(
        [pad_1d(item["input_ids"], pad_id=0) for item in batch],
        dim=0,
    )
    labels = torch.stack(
        [pad_1d(item["labels"], pad_id=-100) for item in batch],
        dim=0,
    )
    attention_mask = torch.stack(
        [pad_1d(item["attention_mask"], pad_id=0) for item in batch],
        dim=0,
    )

    return {
        "input_ids": input_ids,
        "labels": labels,
        "attention_mask": attention_mask,
        "images": images,
    }
