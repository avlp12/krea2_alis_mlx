"""Structural + smoke test for the Krea2 MLX transformer.

1. param-name/shape match vs the turbo.safetensors header (no full ckpt needed),
2. random-init forward pass to validate shapes/logic.
"""

import json
import sys

import mlx.core as mx
import numpy as np
from mlx.utils import tree_flatten, tree_map

from krea2.transformer import Krea2Config, SingleStreamDiT

HDR = "/tmp/hdr.json"  # turbo.safetensors header dumped earlier


def build():
    cfg = Krea2Config()
    m = SingleStreamDiT(cfg)
    mx.eval(m.parameters())
    return cfg, m


def check_keys(m):
    model_params = {k: v for k, v in tree_flatten(m.parameters())}
    model_keys = set(model_params)
    try:
        hdr = json.load(open(HDR))
    except FileNotFoundError:
        print("[keys] header dump missing; skipping key comparison")
        return
    ckpt = {k: v for k, v in hdr.items() if k != "__metadata__"}
    ckpt_keys = set(ckpt)

    missing = ckpt_keys - model_keys  # in ckpt, not in model
    extra = model_keys - ckpt_keys  # in model, not in ckpt
    print(f"[keys] model={len(model_keys)} ckpt={len(ckpt_keys)} "
          f"missing={len(missing)} extra={len(extra)}")
    for k in sorted(missing)[:20]:
        print("   MISSING (ckpt only):", k, ckpt[k]["shape"])
    for k in sorted(extra)[:20]:
        print("   EXTRA  (model only):", k, list(model_params[k].shape))

    # shape match on the intersection
    bad = 0
    for k in sorted(model_keys & ckpt_keys):
        ms = list(model_params[k].shape)
        cs = list(ckpt[k]["shape"])
        if ms != cs:
            bad += 1
            if bad <= 20:
                print(f"   SHAPE MISMATCH {k}: model={ms} ckpt={cs}")
    print(f"[keys] shape mismatches: {bad}")
    ok = not missing and not extra and bad == 0
    print("[keys] EXACT MATCH" if ok else "[keys] MISMATCH")
    return ok


def smoke_forward(cfg, m):
    B, seq, h_, w_ = 1, 8, 4, 4
    Limg = h_ * w_
    img = mx.random.normal((B, Limg, cfg.channels * cfg.patch**2)).astype(mx.bfloat16)
    context = mx.random.normal((B, seq, cfg.txtlayers, cfg.txtdim)).astype(mx.bfloat16)
    t = mx.array([1.0])

    txtpos = np.zeros((seq, 3), np.float32)
    imgids = np.zeros((h_, w_, 3), np.float32)
    imgids[..., 1] = np.arange(h_)[:, None]
    imgids[..., 2] = np.arange(w_)[None, :]
    pos = mx.array(np.concatenate([txtpos, imgids.reshape(-1, 3)], 0))
    mask = mx.ones((B, seq + Limg))

    # cast model to bf16 like reference inference
    m.update(tree_map(lambda a: a.astype(mx.bfloat16) if a.dtype != mx.bfloat16 else a, m.parameters()))
    out = m(img, context, t, pos, mask)
    mx.eval(out)
    expected = (B, Limg, cfg.channels * cfg.patch**2)
    print(f"[fwd] out={out.shape} expected={expected} dtype={out.dtype} "
          f"mean={float(mx.mean(out.astype(mx.float32))):.4f}")
    assert tuple(out.shape) == expected, "output shape mismatch"
    print("[fwd] OK")


if __name__ == "__main__":
    cfg, m = build()
    nparams = sum(v.size for _, v in tree_flatten(m.parameters()))
    print(f"[build] params={nparams/1e9:.2f}B")
    keys_ok = check_keys(m)
    smoke_forward(cfg, m)
    sys.exit(0 if keys_ok else 1)
