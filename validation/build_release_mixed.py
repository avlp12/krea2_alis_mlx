"""Build + verify the mixed-4/8 release artifact (down_proj+endpoints @8, rest @4)."""

import os

import mlx.core as mx
import numpy as np
from mlx import nn
from mlx.utils import tree_flatten, tree_map

from mflux.models.common.weights.loading.weight_definition import ComponentDefinition
from mflux.models.common.weights.loading.weight_loader import WeightLoader
from krea2.quant_recipes import mixed_4_8
from krea2.sampling import sample, to_pil
from krea2.text_encoder import Qwen3VLConditioner
from krea2.transformer import Krea2Config, SingleStreamDiT
from mflux.models.qwen.model.qwen_vae.qwen_vae import QwenVAE
from mflux.models.qwen.weights.qwen_weight_mapping import QwenWeightMapping

REPO = "weights/Krea-2-Turbo"
CKPT = f"{REPO}/turbo.safetensors"
OUT = "release_mixed"
ART = f"{OUT}/transformer_mixed_4_8.safetensors"
os.makedirs(f"{OUT}/samples", exist_ok=True)


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
    nn.quantize(m, group_size=64, bits=4, class_predicate=mixed_4_8)
    mx.eval(m.parameters())
    flat = dict(tree_flatten(m.parameters()))
    mx.save_safetensors(ART, flat, metadata={
        "recipe": "mixed_4_8", "note": "Krea-2-Turbo transformer, mixed 4/8-bit MLX "
        "(down_proj + first/last 2 blocks' attn @8-bit, rest of 28-block attn+mlp @4-bit, g64); other paths bf16.",
    })
    print(f"[build] saved {ART}  ({os.path.getsize(ART)/1e9:.1f}GB, {len(flat)} tensors)")


def load_mixed():
    m = SingleStreamDiT(Krea2Config())
    nn.quantize(m, group_size=64, bits=4, class_predicate=mixed_4_8)
    m.load_weights(ART, strict=True)
    mx.eval(m.parameters())
    return m


def verify():
    enc = Qwen3VLConditioner(REPO, dtype=mx.bfloat16)
    vae = QwenVAE()
    vae.update(WeightLoader.load(weight_definition=_VaeDef, model_path=REPO).components["vae"])
    mx.eval(vae.parameters())
    m = load_mixed()
    nq = sum(1 for _, mod in m.named_modules() if hasattr(mod, "bits"))
    prompts = [("a fox in the snow", 0, "fox"),
               ("a neon city street at night in the rain, reflections", 1, "neon_city"),
               ("a close-up portrait of an old fisherman, weathered face", 2, "fisherman")]
    for prompt, seed, name in prompts:
        dec = sample(m, vae, enc, [prompt], width=1024, height=1024, steps=8, guidance=0.0, seed=seed)
        mx.eval(dec)
        to_pil(dec)[0].save(f"{OUT}/samples/{name}.png")
        o = np.array(dec.astype(mx.float32))
        print(f"[verify] {name}: finite={np.isfinite(o).all()} range=[{o.min():.3f},{o.max():.3f}]")
    print(f"[verify] reloaded qlinears={nq} (expect 224)")


if __name__ == "__main__":
    build()
    verify()
