# DeepSeek-OCR + Qwen3-0.6B Minimal VLM

这个工程把 DeepSeek-OCR 的视觉编码器（通过 `deepseek-ocr-encoder` 包）
接到 Qwen3-0.6B-Base 上，实现一个简化版的 OCR 多模态模型，并提供一个
监督微调脚本。

## 目录结构

- `deepseekocr_qwen3/` 包：
  - `config.py`：训练与模型配置
  - `vision_encoder.py`：DeepSeek-OCR 视觉编码包装
  - `qwen_lm.py`：加载 Qwen3-0.6B
  - `vlm_model.py`：多模态模型拼接逻辑
  - `data.py`：JSONL 数据集定义与 collate_fn
- `train_sft.py`：单机 SFT 训练脚本
- `requirements.txt`：依赖列表

## 安装步骤

1. 创建虚拟环境（可选）

```bash
python -m venv .venv
source .venv/bin/activate  # Windows 用 .venv\Scripts\activate
```

2. 安装依赖

```bash
pip install -r requirements.txt
```

这一步会自动安装：
- PyTorch
- Transformers（与 DeepSeek-OCR encoder 兼容的版本）
- `deepseek-ocr-encoder`（会在首次使用时从 Hugging Face 下载模型权重）
- Accelerate / PEFT / Datasets / Pillow 等

## 准备训练数据

在项目根目录创建 `data/train.jsonl`，每行一个样本，例如：

```jsonl
{"image": "data/images/doc1.png", "text": "用户：请识别这张图片中的全部文字。\n<image>\n助手：这是图片中的文字 ..."}
{"image": "data/images/doc2.jpg", "text": "User: Please OCR this document.\n<image>\nAssistant: The text is ..."}
```

要求：
- `image` 字段是图片的相对或绝对路径。
- `text` 字段中 **必须至少出现一次 `<image>`**，用于占位视觉 token。

## 运行训练（监督微调）

确认 `TrainConfig` 中的 `train_jsonl` 路径正确（默认 `data/train.jsonl`），然后执行：

```bash
python train_sft.py
```

或使用 accelerate 启动多卡：

```bash
accelerate launch train_sft.py
```

训练脚本会：
- 加载 `Qwen/Qwen3-0.6B-Base`
- 构建 `DeepSeekOCRQwen3` 多模态模型
- 冻结 DeepSeek-OCR 视觉编码器，只训练投影层和 Qwen 参数
- 每个 epoch 在 `outputs/` 下保存一个权重文件

## 推理思路（简要）

目前工程只提供训练脚本，你可以参考 `vlm_model.py` 中的逻辑，
写一个简单的推理脚本：

1. 加载 tokenizer 和 Qwen3-0.6B
2. 构建 `DeepSeekOCRQwen3` 并加载你训练好的权重
3. 构造一段文本，包含 `<image>` 占位符
4. 把图片和文本一起丢给模型，使用 `generate` 生成答案

后续你可以在此基础上扩展：
- 多轮对话模板
- LoRA 微调
- 文档问答 / 信息抽取等任务
