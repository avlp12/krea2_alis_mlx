"""Compare MXFP4 / MXFP8 vs affine 4/8-bit on the real Krea-2 transformer.

Same honest metric as validate_quant.py: per-step velocity cosine vs bf16 on a fixed
trajectory (3 prompts x 8 steps) + size + latency. Decides whether any MXFP build is
worth shipping (given quant gives no speedup here — purely a download-size question).
"""

import time

import mlx.core as mx
import numpy as np
from mlx import nn
from mlx.utils import tree_flatten, tree_map

from mflux.models.common.weights.loading.weight_definition import ComponentDefinition
from mflux.models.common.weights.loading.weight_loader import WeightLoader
from krea2.quant_recipes import quantize_bulk
from krea2.sampling import build_positions, patchify, timesteps
from krea2.text_encoder import Qwen3VLConditioner
from krea2.transformer import Krea2Config, SingleStreamDiT
from mflux.models.qwen.model.qwen_vae.qwen_vae import QwenVAE
from mflux.models.qwen.weights.qwen_weight_mapping import QwenWeightMapping

REPO = "weights/Krea-2-Turbo"
CKPT = f"{REPO}/turbo.safetensors"
W = H = 1024
STEPS = 8
PROMPTS = ["a fox in the snow", "a neon city street at night, rain", "a close-up portrait of an old fisherman"]


def cos(a, b):
    a, b = a.ravel().astype(np.float64), b.ravel().astype(np.float64)
    return float((a @ b) / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-12))


def size_gb(m):
    return sum(v.nbytes for _, v in tree_flatten(m.parameters())) / 1e9


def load_bf16():
    m = SingleStreamDiT(Krea2Config())
    m.load_weights(CKPT, strict=True)
    m.update(tree_map(lambda a: a.astype(mx.bfloat16), m.parameters()))
    mx.eval(m.parameters())
    return m


def build_traj(prompt, enc):
    ctx, mask = enc([prompt]); ctx = ctx.astype(mx.bfloat16)
    lat = (H // 8)
    noise = mx.array(np.random.RandomState(0).randn(1, 16, lat, lat).astype(np.float32)).astype(mx.bfloat16)
    h_ = w_ = lat // 2
    img = patchify(noise, 2)
    pos = build_positions(1, ctx.shape[1], h_, w_)
    full_mask = mx.concatenate([mask, mx.ones((1, h_ * w_))], axis=1)
    ts = timesteps(h_ * w_, STEPS, (256 // 16) ** 2, (1280 // 16) ** 2)
    states, vbf = [], []
    return ctx, pos, full_mask, ts, img


def main():
    enc = Qwen3VLConditioner(REPO, dtype=mx.bfloat16)
    bf = load_bf16()
    trajs = {}
    for p in PROMPTS:
        ctx, pos, mask, ts, img = build_traj(p, enc)
        states, vbf = [], []
        for tc, tp in zip(ts[:-1], ts[1:]):
            t = mx.full((1,), tc, dtype=mx.bfloat16)
            v = bf(img, ctx, t, pos, mask); mx.eval(v)
            states.append((img, t)); vbf.append(np.array(v.astype(mx.float32)))
            img = img + (tp - tc) * v
        trajs[p] = (ctx, pos, mask, states, vbf)
    print(f"[bf16-ref] size={size_gb(bf):.1f}GB")
    del bf

    configs = [
        ("affine-8bit", dict(group_size=64, bits=8, class_predicate=quantize_bulk)),
        ("affine-4bit", dict(group_size=64, bits=4, class_predicate=quantize_bulk)),
        ("mxfp8", dict(group_size=32, bits=8, mode="mxfp8", class_predicate=quantize_bulk)),
        ("mxfp4", dict(group_size=32, bits=4, mode="mxfp4", class_predicate=quantize_bulk)),
    ]

    def fwd_ms(m, traj):
        ctx, pos, mask, states, _ = traj
        img, t = states[0]
        mx.eval(m(img, ctx, t, pos, mask))
        t0 = time.time()
        for _ in range(3):
            mx.eval(m(img, ctx, t, pos, mask))
        return (time.time() - t0) / 3 * 1000

    for name, kw in configs:
        m = load_bf16()
        nn.quantize(m, **kw); mx.eval(m.parameters())
        lat = fwd_ms(m, trajs[PROMPTS[0]])
        coss = []
        for p in PROMPTS:
            ctx, pos, mask, states, vbf = trajs[p]
            for (img, t), vb in zip(states, vbf):
                coss.append(cos(np.array(m(img, ctx, t, pos, mask).astype(mx.float32)), vb))
        coss = np.array(coss)
        bpw = size_gb(m) * 8 / 12.82  # transformer params ~12.82B
        print(f"[{name:11s}] size={size_gb(m):4.1f}GB (~{bpw:.2f}bpw)  fwd={lat:5.0f}ms  "
              f"vel_cos mean={coss.mean():.5f} min={coss.min():.5f}")
        del m


if __name__ == "__main__":
    main()
