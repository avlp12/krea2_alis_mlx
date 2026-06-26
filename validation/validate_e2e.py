"""P5 final: full-native end-to-end — PT reference pipeline vs MLX pipeline.

Identical injected noise + same prompt; float32; 512² 8-step. Each pipeline uses its
own components (MLX-encoder for MLX, HF-encoder for PT — both validated cos=1.0).
Catches integration/sampler/schedule bugs per-component tests miss.
"""

import contextlib
import gc
import sys

import numpy as np

REPO = "weights/Krea-2-Turbo"
CKPT = f"{REPO}/turbo.safetensors"
sys.path.insert(0, "krea-2-official")
PROMPT = "a fox in the snow"
W = H = 512
PATCH, COMP, CH = 2, 8, 16
STEPS = 8


def cos(a, b):
    a, b = a.ravel().astype(np.float64), b.ravel().astype(np.float64)
    return float((a @ b) / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-12))


def timesteps(seq_len, steps, x1, x2, y1=0.5, y2=1.15):
    import math
    ts = np.linspace(1, 0, steps + 1)
    slope = (y2 - y1) / (x2 - x1)
    mu = slope * seq_len + (y1 - slope * x1)
    with np.errstate(divide="ignore"):
        ts = math.exp(mu) / (math.exp(mu) + (1.0 / ts - 1.0))
    return ts.tolist()


# shared injected noise + ts
rng = np.random.RandomState(0)
noise = rng.randn(1, CH, H // COMP, W // COMP).astype(np.float32)
h_, w_ = (H // COMP) // PATCH, (W // COMP) // PATCH
align = COMP * PATCH
ts = timesteps(h_ * w_, STEPS, (256 // align) ** 2, (1280 // align) ** 2)


def save(img_chw, path):
    from PIL import Image
    a = (np.transpose(img_chw[0], (1, 2, 0)) * 255).round().clip(0, 255).astype(np.uint8)
    Image.fromarray(a).save(path)


def run_pt():
    import torch
    torch.compile = lambda *a, **k: (a[0] if a and callable(a[0]) else (lambda f: f))
    from safetensors.torch import load_file
    import mmdit
    mmdit.sdpa_kernel = lambda *a, **k: contextlib.nullcontext()
    from mmdit import SingleMMDiTConfig, SingleStreamDiT
    from diffusers import AutoencoderKLQwenImage
    from krea2.text_encoder_hf import Qwen3VLConditionerHF

    cfg = SingleMMDiTConfig(features=6144, tdim=256, txtdim=2560, heads=48, kvheads=12,
                            multiplier=4, layers=28, patch=2, channels=16,
                            txtheads=20, txtkvheads=20, txtlayers=12)
    with torch.device("meta"):
        m = SingleStreamDiT(cfg)
    m.load_state_dict(load_file(CKPT), strict=True, assign=True)
    m = m.float().eval().requires_grad_(False)

    enc = Qwen3VLConditionerHF(REPO, device="cpu", dtype=torch.float32)
    ctx_mx, mask_mx = enc([PROMPT])
    ctx = torch.tensor(np.array(ctx_mx)); mask_np = np.array(mask_mx)
    del enc; gc.collect()
    txtlen = ctx.shape[1]

    img = torch.tensor(noise).reshape(1, CH, h_, PATCH, w_, PATCH).permute(0, 2, 4, 1, 3, 5).reshape(1, h_ * w_, CH * PATCH * PATCH)
    txtpos = np.zeros((txtlen, 3), np.float32)
    imgids = np.zeros((h_, w_, 3), np.float32)
    imgids[..., 1] = np.arange(h_)[:, None]; imgids[..., 2] = np.arange(w_)[None, :]
    pos = torch.tensor(np.concatenate([txtpos, imgids.reshape(-1, 3)], 0))[None]
    full_mask = torch.tensor(np.concatenate([mask_np, np.ones((1, h_ * w_), np.float32)], 1).astype(bool))

    with torch.no_grad():
        for tc, tp in zip(ts[:-1], ts[1:]):
            t = torch.full((1,), tc)
            v = m(img=img, context=ctx, t=t, pos=pos, mask=full_mask)
            img = img + (tp - tc) * v
    del m; gc.collect()

    latent = img.reshape(1, h_, w_, CH, PATCH, PATCH).permute(0, 3, 1, 4, 2, 5).reshape(1, CH, h_ * PATCH, w_ * PATCH)
    ae = AutoencoderKLQwenImage.from_pretrained(REPO, subfolder="vae", torch_dtype=torch.float32).eval()
    std = torch.tensor(ae.config.latents_std).view(1, -1, 1, 1, 1)
    mean = torch.tensor(ae.config.latents_mean).view(1, -1, 1, 1, 1)
    with torch.no_grad():
        dec = ae.decode((latent[:, :, None] * std + mean)).sample  # (1,3,1,H,W)
    dec = dec.clamp(-1, 1) * 0.5 + 0.5
    return dec[:, :, 0].numpy()


def run_mlx():
    import mlx.core as mx
    from mflux.models.common.weights.loading.weight_definition import ComponentDefinition
    from mflux.models.common.weights.loading.weight_loader import WeightLoader
    from krea2.sampling import sample
    from krea2.text_encoder import Qwen3VLConditioner
    from krea2.transformer import Krea2Config, SingleStreamDiT
    from mflux.models.qwen.model.qwen_vae.qwen_vae import QwenVAE
    from mflux.models.qwen.weights.qwen_weight_mapping import QwenWeightMapping

    class _VaeDef:
        @staticmethod
        def get_components():
            return [ComponentDefinition(name="vae", hf_subdir="vae", loading_mode="single",
                                        mapping_getter=QwenWeightMapping.get_vae_mapping)]
        @staticmethod
        def get_download_patterns():
            return ["vae/*.safetensors", "vae/*.json"]

    m = SingleStreamDiT(Krea2Config())
    m.load_weights(CKPT, strict=True)  # keep float32
    mx.eval(m.parameters())
    vae = QwenVAE()
    vae.update(WeightLoader.load(weight_definition=_VaeDef, model_path=REPO).components["vae"])
    enc = Qwen3VLConditioner(REPO, dtype=mx.float32)
    dec = sample(m, vae, enc, [PROMPT], width=W, height=H, steps=STEPS, guidance=0.0,
                 init_noise=noise, dtype=mx.float32)
    return np.array(dec.astype(mx.float32))


if __name__ == "__main__":
    print(f"[e2e] {W}x{H} {STEPS} steps, ts={[round(t,3) for t in ts]}")
    img_pt = run_pt()
    save(img_pt, "out/e2e_pt_512.png")
    img_mlx = run_mlx()
    save(img_mlx, "out/e2e_mlx_512.png")
    c = cos(img_mlx, img_pt)
    mad = float(np.abs(img_mlx - img_pt).max())
    print(f"[E2E pixel cmp] cos={c:.6f}  max|diff|={mad:.4f}  "
          f"mlx[{img_mlx.min():.3f},{img_mlx.max():.3f}] pt[{img_pt.min():.3f},{img_pt.max():.3f}]")
    print("[P5 e2e] PASS" if c > 0.99 else "[P5 e2e] CHECK")
