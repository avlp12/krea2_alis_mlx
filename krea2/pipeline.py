"""End-to-end Krea-2-Turbo MLX pipeline.

Ships only the (quantized or bf16) transformer; the Qwen-Image VAE, Qwen3-VL-4B text
encoder, and tokenizer are pulled at runtime from `krea/Krea-2-Turbo` (you accept
Krea's license there). The VAE reuses mflux's `QwenVAE` — install with `pip install mflux`.
"""

from __future__ import annotations

import os

import mlx.core as mx
from mlx import nn
from mlx.utils import tree_map

from .quant_recipes import mixed_4_8, quantize_bulk
from .sampling import sample, to_pil
from .text_encoder import Qwen3VLConditioner
from .transformer import Krea2Config, SingleStreamDiT

BASE_REPO = "krea/Krea-2-Turbo"
DEFAULT_WEIGHTS_REPO = "avlp12/Krea-2-Turbo-Alis-MLX-8bit"
_WEIGHTS = {"8bit": "transformer_8bit.safetensors", "mixed-4-8": "transformer_mixed_4_8.safetensors"}


def resolve_weights(folder: str = ".", download: bool = True):
    """Find the transformer weights to use: a local file in `folder` if present,
    otherwise (download=True) fetch the 8-bit build from HF. Returns (precision, path)."""
    for prec, fname in _WEIGHTS.items():
        p = os.path.join(folder, fname)
        if os.path.exists(p):
            return prec, p
    if not download:
        return "8bit", None  # nothing local yet; the 8-bit build will be fetched on load
    from huggingface_hub import hf_hub_download

    return "8bit", hf_hub_download(DEFAULT_WEIGHTS_REPO, _WEIGHTS["8bit"])


def _base_dir() -> str:
    from huggingface_hub import snapshot_download

    return snapshot_download(
        BASE_REPO,
        allow_patterns=["vae/*", "text_encoder/*", "tokenizer/*", "model_index.json"],
    )


def _load_vae(base_dir: str):
    from mflux.models.common.weights.loading.weight_definition import ComponentDefinition
    from mflux.models.common.weights.loading.weight_loader import WeightLoader
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

    vae = QwenVAE()
    vae.update(WeightLoader.load(weight_definition=_VaeDef, model_path=base_dir).components["vae"])
    mx.eval(vae.parameters())
    return vae


class Krea2Pipeline:
    """precision:
    - '8bit'      : transformer_8bit.safetensors (28-block attn+mlp @ 8-bit)
    - 'mixed-4-8' : transformer_mixed_4_8.safetensors (down_proj+endpoints @8, rest @4)
    - 'bf16'      : krea/Krea-2-Turbo/turbo.safetensors (auto-downloaded)"""

    def __init__(self, transformer_path: str | None = None, precision: str = "8bit", base_dir: str | None = None):
        base = base_dir or _base_dir()
        m = SingleStreamDiT(Krea2Config())
        if precision == "8bit":
            nn.quantize(m, group_size=64, bits=8, class_predicate=quantize_bulk)
            m.load_weights(transformer_path, strict=True)
        elif precision == "mixed-4-8":
            nn.quantize(m, group_size=64, bits=4, class_predicate=mixed_4_8)
            m.load_weights(transformer_path, strict=True)
        elif precision == "bf16":
            if transformer_path is None:
                from huggingface_hub import hf_hub_download
                transformer_path = hf_hub_download(BASE_REPO, "turbo.safetensors")
            m.load_weights(transformer_path, strict=True)
            m.update(tree_map(lambda a: a.astype(mx.bfloat16), m.parameters()))
        else:
            raise ValueError(f"precision must be '8bit', 'mixed-4-8' or 'bf16', got {precision}")
        mx.eval(m.parameters())
        self.transformer = m
        self.vae = _load_vae(base)
        self.encoder = Qwen3VLConditioner(base, dtype=mx.bfloat16)

    def generate(self, prompt, *, width=1024, height=1024, steps=8, seed=0, num_images=1):
        dec = sample(self.transformer, self.vae, self.encoder, [prompt] * num_images,
                     width=width, height=height, steps=steps, guidance=0.0, seed=seed)
        return to_pil(dec)
