"""P2b validation: pure-MLX Qwen3 encoder vs HF-transformers oracle (float32)."""

import numpy as np
import mlx.core as mx
import torch

REPO = "weights/Krea-2-Turbo"
PROMPT = "a fox in the snow"


def cos(a, b):
    a, b = a.ravel().astype(np.float64), b.ravel().astype(np.float64)
    return float((a @ b) / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-12))


# HF oracle (float32)
from krea2.text_encoder_hf import Qwen3VLConditionerHF

hf = Qwen3VLConditionerHF(REPO, device="cpu", dtype=torch.float32)
ctx_hf, mask_hf = hf([PROMPT])
ctx_hf = np.array(ctx_hf)  # (1, seq, 12, 2560)
del hf

# MLX
from krea2.text_encoder import Qwen3VLConditioner

ml = Qwen3VLConditioner(REPO, dtype=mx.float32)
print(f"[mlx-enc] loaded {ml.nloaded} language_model tensors")
ctx_ml, mask_ml = ml([PROMPT])
ctx_ml = np.array(ctx_ml)

print(f"[shapes] hf={ctx_hf.shape} mlx={ctx_ml.shape}")
assert ctx_hf.shape == ctx_ml.shape, "shape mismatch"
print(f"[mask] match={bool(np.array_equal(np.array(mask_ml), np.array(mask_hf)))}")

# overall + per selected-layer cosine (only over valid tokens)
valid = int(np.array(mask_ml).sum())
SEL = (2, 5, 8, 11, 14, 17, 20, 23, 26, 29, 32, 35)
print(f"[valid tokens] {valid}")
print(f"[OVERALL] cos={cos(ctx_ml, ctx_hf):.6f}")
for li in range(ctx_hf.shape[2]):
    a = ctx_ml[:, :valid, li, :]
    b = ctx_hf[:, :valid, li, :]
    print(f"  layer idx {SEL[li]:2d}: cos={cos(a, b):.6f}  max|diff|={np.abs(a-b).max():.4f}")

c = cos(ctx_ml[:, :valid], ctx_hf[:, :valid])
print(f"[ENCODER valid-only cos] {c:.6f}")
print("[P2b encoder] PASS" if c > 0.999 else "[P2b encoder] CHECK")
