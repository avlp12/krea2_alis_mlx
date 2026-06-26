"""Build + verify the 8-bit quantized transformer release artifact.

Saves release/transformer_8bit.safetensors (quantize_bulk @8-bit g64) and verifies
it RELOADS and generates a coherent image (don't ship unverified).
"""

import os
import time

import mlx.core as mx
import numpy as np
from mlx import nn
from mlx.utils import tree_flatten, tree_map

from mflux.models.common.weights.loading.weight_definition import ComponentDefinition
from mflux.models.common.weights.loading.weight_loader import WeightLoader
from krea2.quant_recipes import quantize_bulk
from krea2.sampling import sample, to_pil
from krea2.text_encoder import Qwen3VLConditioner
from krea2.transformer import Krea2Config, SingleStreamDiT
from mflux.models.qwen.model.qwen_vae.qwen_vae import QwenVAE
from mflux.models.qwen.weights.qwen_weight_mapping import QwenWeightMapping

REPO = "weights/Krea-2-Turbo"
CKPT = f"{REPO}/turbo.safetensors"
OUT = "release"
ART = f"{OUT}/transformer_8bit.safetensors"
os.makedirs(OUT, exist_ok=True)


class _VaeDef:
    @staticmethod
    def get_components():
        return [ComponentDefinition(name="vae", hf_subdir="vae", loading_mode="single",
                                    mapping_getter=QwenWeightMapping.get_vae_mapping)]
    @staticmethod
    def get_download_patterns():
        return ["vae/*.safetensors", "vae/*.json"]


def build():
    m = SingleStreamDiT(Krea2Config())
    m.load_weights(CKPT, strict=True)
    m.update(tree_map(lambda a: a.astype(mx.bfloat16), m.parameters()))
    nn.quantize(m, group_size=64, bits=8, class_predicate=quantize_bulk)
    mx.eval(m.parameters())
    flat = dict(tree_flatten(m.parameters()))
    mx.save_safetensors(ART, flat, metadata={
        "bits": "8", "group_size": "64", "recipe": "quantize_bulk",
        "note": "Krea-2-Turbo transformer, 8-bit MLX (28-block attn+mlp); other paths bf16.",
    })
    gb = os.path.getsize(ART) / 1e9
    print(f"[build] saved {ART}  ({gb:.1f}GB, {len(flat)} tensors)")
    return gb


def load_quantized():
    m = SingleStreamDiT(Krea2Config())
    nn.quantize(m, group_size=64, bits=8, class_predicate=quantize_bulk)  # recreate structure
    m.load_weights(ART, strict=True)
    mx.eval(m.parameters())
    return m


def verify():
    enc = Qwen3VLConditioner(REPO, dtype=mx.bfloat16)
    vae = QwenVAE()
    vae.update(WeightLoader.load(weight_definition=_VaeDef, model_path=REPO).components["vae"])
    mx.eval(vae.parameters())
    m = load_quantized()
    nq = sum(1 for _, mod in m.named_modules() if hasattr(mod, "bits"))
    t0 = time.time()
    dec = sample(m, vae, enc, ["a fox in the snow"], width=1024, height=1024,
                 steps=8, guidance=0.0, init_noise=np.random.RandomState(0).randn(1, 16, 128, 128).astype(np.float32))
    mx.eval(dec)
    to_pil(dec)[0].save(f"{OUT}/verify_8bit_fox.png")
    o = np.array(dec.astype(mx.float32))
    print(f"[verify] reloaded qlinears={nq} (expect 224)  gen {time.time()-t0:.0f}s  "
          f"finite={np.isfinite(o).all()} range=[{o.min():.3f},{o.max():.3f}] -> {OUT}/verify_8bit_fox.png")


if __name__ == "__main__":
    build()
    verify()
