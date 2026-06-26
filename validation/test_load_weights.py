"""Load real turbo.safetensors into the MLX transformer (strict) and run a forward."""

import time

import mlx.core as mx
import numpy as np
from mlx.utils import tree_flatten, tree_map

from krea2.transformer import Krea2Config, SingleStreamDiT

CKPT = "weights/Krea-2-Turbo/turbo.safetensors"

cfg = Krea2Config()
m = SingleStreamDiT(cfg)

# strict load — errors if any param missing / shape mismatch (validates zero-remap claim)
t0 = time.time()
m.load_weights(CKPT, strict=True)
mx.eval(m.parameters())
print(f"[load] strict load OK in {time.time()-t0:.1f}s")

# dtype histogram before cast (ckpt is mixed f32/bf16)
from collections import Counter
dt = Counter(str(v.dtype) for _, v in tree_flatten(m.parameters()))
print(f"[load] ckpt dtypes: {dict(dt)}")

# cast whole model to bf16 like reference inference (.to(bf16)); RMSNorm upcasts internally
m.update(tree_map(lambda a: a.astype(mx.bfloat16), m.parameters()))
mx.eval(m.parameters())

# forward with random inputs (sanity: finite output, correct shape)
B, seq, h_, w_ = 1, 16, 8, 8
Limg = h_ * w_
img = mx.random.normal((B, Limg, cfg.channels * cfg.patch**2)).astype(mx.bfloat16)
context = mx.random.normal((B, seq, cfg.txtlayers, cfg.txtdim)).astype(mx.bfloat16)
t = mx.array([0.9])
txtpos = np.zeros((seq, 3), np.float32)
imgids = np.zeros((h_, w_, 3), np.float32)
imgids[..., 1] = np.arange(h_)[:, None]
imgids[..., 2] = np.arange(w_)[None, :]
pos = mx.array(np.concatenate([txtpos, imgids.reshape(-1, 3)], 0))
mask = mx.ones((B, seq + Limg))

t0 = time.time()
out = m(img, context, t, pos, mask)
mx.eval(out)
o = out.astype(mx.float32)
print(f"[fwd] {time.time()-t0:.2f}s  out={out.shape} dtype={out.dtype}")
print(f"[fwd] finite={bool(mx.all(mx.isfinite(o)).item())} "
      f"mean={float(mx.mean(o)):.4f} std={float(mx.std(o)):.4f} "
      f"min={float(mx.min(o)):.3f} max={float(mx.max(o)):.3f}")
