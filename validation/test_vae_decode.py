"""P3: load Krea-2 VAE weights into mflux QwenVAE (reuse) and decode-smoke a latent."""

import time

import mlx.core as mx

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


t0 = time.time()
loaded = WeightLoader.load(weight_definition=_VaeOnlyDef, model_path=REPO)
vae = QwenVAE()
vae.update(loaded.components["vae"])  # strict by default
mx.eval(vae.parameters())
print(f"[vae] loaded + applied in {time.time()-t0:.1f}s")

# decode a random latent: (B, 16, h, w) -> image (B, 3, 1, h*8, w*8)
h = w = 32  # -> 256x256 image
lat = mx.random.normal((1, 16, h, w)).astype(mx.float32)
t0 = time.time()
img = vae.decode(lat)
mx.eval(img)
print(f"[vae] decode {time.time()-t0:.2f}s  out={img.shape} dtype={img.dtype}")
o = img.astype(mx.float32)
print(f"[vae] finite={bool(mx.all(mx.isfinite(o)).item())} "
      f"min={float(mx.min(o)):.3f} max={float(mx.max(o)):.3f} mean={float(mx.mean(o)):.3f}")
