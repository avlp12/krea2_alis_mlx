"""P5: single-step velocity parity — MLX SingleStreamDiT vs PT reference (float32).

Same inputs (real encoder context + fixed noise) into both transformers; compare
velocity. Isolates the transformer port. cos>0.9999 expected for a faithful port.
PT runs first (then freed); MLX second; compared in numpy.
"""

import contextlib
import gc
import sys

import numpy as np

REPO = "weights/Krea-2-Turbo"
CKPT = f"{REPO}/turbo.safetensors"
sys.path.insert(0, "krea-2-official")

PROMPT = "a fox in the snow"
H = W = 256  # small res -> 256 img tokens, keeps PT forward cheap
PATCH, COMP, CH = 2, 8, 16


def cos(a, b):
    a, b = a.ravel().astype(np.float64), b.ravel().astype(np.float64)
    return float((a @ b) / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-12))


def build_inputs():
    """Returns numpy inputs shared by both frameworks + torch context."""
    import torch
    from krea2.text_encoder_hf import Qwen3VLConditionerHF

    enc = Qwen3VLConditionerHF(REPO, device="cpu", dtype=torch.float32)
    # call the underlying torch path directly to get a torch context
    ctx_mx, mask_mx = enc([PROMPT])
    ctx = np.array(ctx_mx)  # (1, seq, 12, 2560) float32
    mask = np.array(mask_mx)  # (1, seq)
    del enc
    gc.collect()

    rng = np.random.RandomState(0)
    lat = rng.randn(1, CH, H // COMP, W // COMP).astype(np.float32)
    # patchify (b,c,Hl,Wl)->(b, hw, c*p*p)
    b, c, Hl, Wl = lat.shape
    h_, w_ = Hl // PATCH, Wl // PATCH
    img = lat.reshape(b, c, h_, PATCH, w_, PATCH).transpose(0, 2, 4, 1, 3, 5).reshape(b, h_ * w_, c * PATCH * PATCH)

    seq = ctx.shape[1]
    txtpos = np.zeros((seq, 3), np.float32)
    imgids = np.zeros((h_, w_, 3), np.float32)
    imgids[..., 1] = np.arange(h_)[:, None]
    imgids[..., 2] = np.arange(w_)[None, :]
    pos = np.concatenate([txtpos, imgids.reshape(-1, 3)], axis=0)  # (L,3)
    full_mask = np.concatenate([mask, np.ones((1, h_ * w_), np.float32)], axis=1)  # (1,L)
    t = np.array([1.0], np.float32)
    return dict(ctx=ctx, mask=mask, img=img, pos=pos, full_mask=full_mask, t=t, h_=h_, w_=w_, seq=seq)


def run_pt(inp):
    import torch

    torch.compile = lambda *a, **k: (a[0] if a and callable(a[0]) else (lambda f: f))  # disable compile
    from safetensors.torch import load_file

    import mmdit  # noqa
    mmdit.sdpa_kernel = lambda *a, **k: contextlib.nullcontext()  # drop CUDNN-only backend
    from mmdit import SingleMMDiTConfig, SingleStreamDiT

    cfg = SingleMMDiTConfig(features=6144, tdim=256, txtdim=2560, heads=48, kvheads=12,
                            multiplier=4, layers=28, patch=2, channels=16,
                            txtheads=20, txtkvheads=20, txtlayers=12)
    dev = "cpu"  # MPS lacks float64 (reference rope); CPU = faithful reference
    with torch.device("meta"):
        m = SingleStreamDiT(cfg)
    m.load_state_dict(load_file(CKPT), strict=True, assign=True)
    m = m.to(device=dev, dtype=torch.float32).eval().requires_grad_(False)

    img = torch.tensor(inp["img"], device=dev)
    ctx = torch.tensor(inp["ctx"], device=dev)
    t = torch.tensor(inp["t"], device=dev)
    pos = torch.tensor(inp["pos"], device=dev)[None]  # PT wants (b,L,3)
    mask = torch.tensor(inp["full_mask"].astype(bool), device=dev)
    with torch.no_grad():
        out = m(img=img, context=ctx, t=t, pos=pos, mask=mask)
    v = out.float().cpu().numpy()
    print(f"[pt] device={dev} out={v.shape}")
    del m
    gc.collect()
    return v


def run_mlx(inp):
    import mlx.core as mx
    from krea2.transformer import Krea2Config, SingleStreamDiT

    m = SingleStreamDiT(Krea2Config())
    m.load_weights(CKPT, strict=True)
    mx.eval(m.parameters())  # keep float32 (no bf16 cast) for clean correctness check
    out = m(
        mx.array(inp["img"]), mx.array(inp["ctx"]), mx.array(inp["t"]),
        mx.array(inp["pos"]), mx.array(inp["full_mask"]),
    )
    mx.eval(out)
    v = np.array(out.astype(mx.float32))
    print(f"[mlx] out={v.shape}")
    return v


if __name__ == "__main__":
    inp = build_inputs()
    print(f"[inputs] seq={inp['seq']} img_tokens={inp['h_']*inp['w_']} valid_txt={int(inp['mask'].sum())}")
    v_pt = run_pt(inp)
    v_mlx = run_mlx(inp)
    c = cos(v_mlx, v_pt)
    rel = float(np.linalg.norm(v_mlx - v_pt) / (np.linalg.norm(v_pt) + 1e-9))
    mad = float(np.abs(v_mlx - v_pt).max())
    print(f"[VELOCITY cmp] cos={c:.6f}  rel_l2={rel:.5f}  max|diff|={mad:.5f}")
    print("[P5 transformer] PASS" if c > 0.9999 else "[P5 transformer] CHECK — cos below 0.9999")
