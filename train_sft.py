import os

import torch
from torch.utils.data import DataLoader
from torch.optim import AdamW
from torch.nn.utils import clip_grad_norm_

from accelerate import Accelerator

from deepseekocr_qwen3.config import QwenConfig, TrainConfig
from deepseekocr_qwen3.qwen_lm import load_qwen3_lm
from deepseekocr_qwen3.vlm_model import DeepSeekOCRQwen3
from deepseekocr_qwen3.data import OCRJsonlDataset, collate_ocr


def main() -> None:
    qwen_cfg = QwenConfig()
    train_cfg = TrainConfig()

    accelerator = Accelerator(
        mixed_precision="bf16" if train_cfg.bf16 else "no"
    )

    # 1. Load Qwen3
    llm, tokenizer = load_qwen3_lm(qwen_cfg)

    # 2. Build VLM
    vlm = DeepSeekOCRQwen3(llm, tokenizer, device=accelerator.device)

    # 3. Freeze DeepSeek-OCR encoder, train projector + Qwen
    for p in vlm.vision_encoder.parameters():
        p.requires_grad = False

    params = [p for p in vlm.parameters() if p.requires_grad]
    optimizer = AdamW(params, lr=train_cfg.lr)

    # 4. Dataset
    dataset = OCRJsonlDataset(
        train_cfg.train_jsonl,
        tokenizer=tokenizer,
        max_samples=train_cfg.max_samples,
    )
    dataloader = DataLoader(
        dataset,
        batch_size=train_cfg.batch_size,
        shuffle=True,
        collate_fn=collate_ocr,
    )

    vlm, optimizer, dataloader = accelerator.prepare(
        vlm, optimizer, dataloader
    )

    vlm.train()
    global_step = 0

    for epoch in range(train_cfg.num_epochs):
        for step, batch in enumerate(dataloader):
            outputs = vlm(batch)
            loss = outputs.loss
            loss = loss / train_cfg.gradient_accumulation_steps

            accelerator.backward(loss)

            if (step + 1) % train_cfg.gradient_accumulation_steps == 0:
                clip_grad_norm_(vlm.parameters(), 1.0)
                optimizer.step()
                optimizer.zero_grad()
                global_step += 1

                if accelerator.is_main_process and global_step % 10 == 0:
                    print(
                        f"epoch {epoch} step {global_step} "
                        f"loss {loss.item():.4f}"
                    )

        if accelerator.is_main_process:
            os.makedirs(train_cfg.output_dir, exist_ok=True)
            unwrapped = accelerator.unwrap_model(vlm)
            ckpt_path = os.path.join(
                train_cfg.output_dir, f"vlm_epoch{epoch}.pt"
            )
            torch.save(unwrapped.state_dict(), ckpt_path)
            print(f"Saved checkpoint to {ckpt_path}")


if __name__ == "__main__":
    main()
