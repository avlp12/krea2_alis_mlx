"""P6 honest quant validation (per 3-lens review):

Per-step VELOCITY error vs full-precision bf16 on a FIXED trajectory (isolates quant
error from benign ODE divergence) + per-step latency + size. Quantizes from f32, not
bf16-rounded. Reference = MLX full-precision velocity (already cos=1.0 vs PT).
"""

import time

import mlx.core as mx
import numpy as np
from mlx import nn
from mlx.utils import tree_flatten

from mflux.models.common.weights.loading.weight_definition import ComponentDefinition
from mflux.models.common.weights.loading.weight_loader import WeightLoader
from krea2.quant_recipes import mixed_4_8, quantize_bulk
from krea2.sampling import build_positions, patchify, timesteps
from krea2.text_encoder import Qwen3VLConditioner
from krea2.transformer import Krea2Config, SingleStreamDiT
from mflux.models.qwen.model.qwen_vae.qwen_vae import QwenVAE
from mflux.models.qwen.weights.qwen_weight_mapping import QwenWeightMapping

REPO = "weights/Krea-2-Turbo"
CKPT = f"{REPO}/turbo.safetensors"
W = H = 1024
STEPS = 8
# 12-prompt diverse set (animal/snow, urban/neon, portrait, landscape, food, sci-fi, painterly,
# bright animal, dramatic figure, interior, macro, dense scene) — stresses quant across content types.
# First 3 are the original set, kept for continuity / determinism cross-check.
PROMPTS = [
    "a fox in the snow",
    "a neon city street at night, rain",
    "a close-up portrait of an old fisherman",
    "a serene mountain lake at sunrise, mist over the water",
    "a bowl of ramen with steam rising, top-down food photography",
    "a futuristic spaceship interior, intricate machinery and glowing panels",
    "a watercolor painting of cherry blossoms, soft pastel tones",
    "a golden retriever puppy running through a sunlit meadow",
    "an astronaut standing on a red martian desert, dramatic lighting",
    "a vintage bookstore interior, warm light, shelves of old books",
    "a macro shot of a dewdrop on a green leaf, sharp detail",
    "a bustling Tokyo crosswalk at dusk, crowds and neon signage",
]


class _VaeDef:
    @staticmethod
    def get_components():
        return [ComponentDefinition(name="vae", hf_subdir="vae", loading_mode="single",
                                    mapping_getter=QwenWeightMapping.get_vae_mapping)]
    @staticmethod
    def get_download_patterns():
        return ["vae/*.safetensors", "vae/*.json"]


def cos(a, b):
    a, b = a.ravel().astype(np.float64), b.ravel().astype(np.float64)
    return float((a @ b) / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-12))


def size_gb(m):
    return sum(v.nbytes for _, v in tree_flatten(m.parameters())) / 1e9


def load_transformer(bf16=True):
    m = SingleStreamDiT(Krea2Config())
    m.load_weights(CKPT, strict=True)  # f32 source (no pre-round)
    if bf16:
        from mlx.utils import tree_map
        m.update(tree_map(lambda a: a.astype(mx.bfloat16), m.parameters()))
    mx.eval(m.parameters())
    return m


def build_traj(prompt, enc):
    ctx, mask = enc([prompt])
    ctx = ctx.astype(mx.bfloat16)
    lat_h = lat_w = (H // 8)
    noise = mx.array(np.random.RandomState(0).randn(1, 16, lat_h, lat_w).astype(np.float32)).astype(mx.bfloat16)
    h_ = w_ = lat_h // 2
    img = patchify(noise, 2)
    pos = build_positions(1, ctx.shape[1], h_, w_)
    full_mask = mx.concatenate([mask, mx.ones((1, h_ * w_))], axis=1)
    ts = timesteps(h_ * w_, STEPS, (256 // 16) ** 2, (1280 // 16) ** 2)
    return ctx, pos, full_mask, ts, img


def main():
    enc = Qwen3VLConditioner(REPO, dtype=mx.bfloat16)

    # bf16 reference: trajectory states + velocities (per prompt)
    bf = load_transformer(bf16=True)
    print(f"[bf16] size={size_gb(bf):.1f}GB")
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
    # bf16 reference latency (same fixed state), for a self-contained traceable log
    _ctx, _pos, _mask, _states, _ = trajs[PROMPTS[0]]
    _img, _t = _states[0]
    mx.eval(bf(_img, _ctx, _t, _pos, _mask))
    _t0 = time.time()
    for _ in range(3):
        mx.eval(bf(_img, _ctx, _t, _pos, _mask))
    print(f"[bf16-ref  ] size={size_gb(bf):4.1f}GB  fwd={(time.time()-_t0)/3*1000:5.0f}ms")
    del bf

    configs = [
        ("8bit-bulk", lambda m: nn.quantize(m, group_size=64, bits=8, class_predicate=quantize_bulk)),
        ("4bit-bulk", lambda m: nn.quantize(m, group_size=64, bits=4, class_predicate=quantize_bulk)),
        ("mixed-4/8", lambda m: nn.quantize(m, group_size=64, bits=4, class_predicate=mixed_4_8)),
    ]

    def fwd_latency(m, traj):
        ctx, pos, mask, states, _ = traj
        img, t = states[0]
        mx.eval(m(img, ctx, t, pos, mask))  # warmup
        t0 = time.time()
        for _ in range(3):
            mx.eval(m(img, ctx, t, pos, mask))
        return (time.time() - t0) / 3

    for name, qfn in configs:
        m = load_transformer(bf16=True)
        qfn(m); mx.eval(m.parameters())
        nq = sum(1 for _, mod in m.named_modules() if hasattr(mod, "bits"))
        lat = fwd_latency(m, trajs[PROMPTS[0]])
        # per-step velocity cos vs bf16, averaged over prompts & steps
        all_cos = []
        for p in PROMPTS:
            ctx, pos, mask, states, vbf = trajs[p]
            for (img, t), vb in zip(states, vbf):
                vq = np.array(m(img, ctx, t, pos, mask).astype(mx.float32))
                all_cos.append(cos(vq, vb))
        all_cos = np.array(all_cos)
        print(f"[{name:10s}] size={size_gb(m):4.1f}GB  qlinears={nq:3d}  fwd={lat*1000:5.0f}ms  "
              f"vel_cos vs bf16: mean={all_cos.mean():.5f} min={all_cos.min():.5f}  (n={len(all_cos)})")
        del m


if __name__ == "__main__":
    main()
