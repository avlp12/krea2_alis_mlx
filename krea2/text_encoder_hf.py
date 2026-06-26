"""HF-transformers Qwen3-VL-4B conditioner — ORACLE / interim path for Krea-2.

Faithful port of krea-2-official/encoder.py. Used to (a) drive an end-to-end MLX
image before the encoder is ported, and (b) serve as the numerical oracle when
validating the pure-MLX encoder port (P2b). NOT the final hot path.
"""

from __future__ import annotations

import mlx.core as mx
import numpy as np
import torch

SELECT_LAYERS = (2, 5, 8, 11, 14, 17, 20, 23, 26, 29, 32, 35)
PREFIX = (
    "<|im_start|>system\nDescribe the image by detailing the color, shape, size, "
    "texture, quantity, text, spatial relationships of the objects and background:"
    "<|im_end|>\n<|im_start|>user\n"
)
SUFFIX = "<|im_end|>\n<|im_start|>assistant\n"
PREFIX_START_IDX = 34
SUFFIX_START_IDX = 5


class Qwen3VLConditionerHF:
    def __init__(self, repo: str, max_length: int = 512, device: str = "cpu",
                 dtype: torch.dtype = torch.bfloat16):
        from transformers import AutoTokenizer, Qwen3VLForConditionalGeneration

        self.device = device
        self.max_length = max_length
        self.tokenizer = AutoTokenizer.from_pretrained(f"{repo}/tokenizer")
        self.qwen = (
            Qwen3VLForConditionalGeneration.from_pretrained(f"{repo}/text_encoder", torch_dtype=dtype)
            .eval()
            .requires_grad_(False)
            .to(device)
        )

    @torch.no_grad()
    def __call__(self, prompts: list[str]) -> tuple[mx.array, mx.array]:
        prefix_idx = PREFIX_START_IDX
        text = [PREFIX + p for p in prompts]
        suffix = [SUFFIX] * len(text)

        suffix_inputs = self.tokenizer(text=suffix, return_tensors="pt").to(self.device)
        suffix_ids = suffix_inputs["input_ids"]
        suffix_mask = suffix_inputs["attention_mask"].bool()

        inputs = self.tokenizer(
            text,
            truncation=True,
            padding="max_length",
            max_length=self.max_length + prefix_idx - SUFFIX_START_IDX,
            return_tensors="pt",
        ).to(self.device)
        input_ids = torch.cat([inputs["input_ids"], suffix_ids], dim=1)
        mask = torch.cat([inputs["attention_mask"].bool(), suffix_mask], dim=1)

        out = self.qwen(input_ids=input_ids, attention_mask=mask, output_hidden_states=True)
        hiddens = torch.stack([out.hidden_states[i] for i in SELECT_LAYERS], dim=2)  # (B,L,12,2560)
        hiddens = hiddens[:, prefix_idx:]
        mask = mask[:, prefix_idx:]

        ctx = mx.array(hiddens.float().cpu().numpy().astype(np.float32))
        msk = mx.array(mask.cpu().numpy().astype(np.float32))
        return ctx, msk
