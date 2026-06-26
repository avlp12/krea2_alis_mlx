"""P3 validation: MLX QwenVAE.decode (Krea-2 weights) vs PT diffusers AutoencoderKLQwenImage.

Same normalized latent into both; compare decoded pixels. (Latent is the VAE's true
input, not a shared intermediate — no blind-test trap.)
"""

import numpy as np
import mlx.core as mx
import torch
from mlx.utils import tree_flatten

from mflux.models.common.weights.loading.weight_definition import ComponentDefinition
from mflux.models.common.weights.loading.weight_loader import WeightLoader
from mflux.models.qwen.model.qwen_vae.qwen_vae import QwenVAE
from mflux.models.qwen.weights.qwen_weight_mapping import QwenWeightMapping

REPO = "weights/Krea-2-Turbo"


class _VaeOnlyDef:
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


# ---- MLX side ----
loaded = WeightLoader.load(weight_definition=_VaeOnlyDef, model_path=REPO)
vae_weights = {k: v for k, v in tree_flatten(loaded.components["vae"])}
vae = QwenVAE()
model_keys = {k for k, _ in tree_flatten(vae.parameters())}
# diagnostics: how many mapped, do they hit decoder params, did values actually change?
before = {k: float(v.astype(mx.float32).sum().item()) for k, v in tree_flatten(vae.parameters()) if "decoder.conv_in" in k}
vae.update(loaded.components["vae"])
mx.eval(vae.parameters())
after = {k: float(v.astype(mx.float32).sum().item()) for k, v in tree_flatten(vae.parameters()) if "decoder.conv_in" in k}
mapped = set(vae_weights)
print(f"[diag] mapped weights={len(mapped)}  model params={len(model_keys)}  "
      f"covered={len(mapped & model_keys)}  unmapped_model={len(model_keys - mapped)}")
changed = sum(1 for k in before if abs(before[k] - after.get(k, 0)) > 1e-6)
print(f"[diag] decoder.conv_in params changed by update: {changed}/{len(before)} (must be >0)")

np.random.seed(0)
lat_np = np.random.randn(1, 16, 32, 32).astype(np.float32)
img_mx = np.array(vae.decode(mx.array(lat_np)).astype(mx.float32))  # (1,3,1,256,256)

# ---- PT reference ----
from diffusers import AutoencoderKLQwenImage

ae = AutoencoderKLQwenImage.from_pretrained(REPO, subfolder="vae", torch_dtype=torch.float32).eval()
std = torch.tensor(ae.config.latents_std).view(1, -1, 1, 1, 1)
mean = torch.tensor(ae.config.latents_mean).view(1, -1, 1, 1, 1)
x = torch.from_numpy(lat_np).reshape(1, 16, 1, 32, 32) * std + mean
with torch.no_grad():
    img_pt = ae.decode(x).sample.numpy()  # (1,3,1,256,256)

print(f"[shapes] mlx={img_mx.shape} pt={img_pt.shape}")
c = cos(img_mx, img_pt)
mad = float(np.abs(img_mx - img_pt).max())
print(f"[VAE cmp] pixel cos={c:.6f}  max|diff|={mad:.5f}  "
      f"mlx[min,max]=[{img_mx.min():.3f},{img_mx.max():.3f}] pt[min,max]=[{img_pt.min():.3f},{img_pt.max():.3f}]")
print("[VAE] PASS" if c > 0.999 else "[VAE] FAIL — investigate")
