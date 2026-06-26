"""P6 VISUAL aid only: render bf16 vs quantized 1024² generations for eyeballing.

NOTE: the pixel-cos-vs-bf16 printed here is ILLUSTRATIVE, not the quality metric —
an 8-step ODE makes final-pixel cosine conflate benign trajectory divergence with
real degradation. The authoritative quality metric is per-step velocity cos in
`scripts/validate_quant.py`. Recipe = `krea2.quant_recipes.quantize_bulk` (eval==ship).
"""

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
W = H = 1024
STEPS = 8


class _VaeDef:
    @staticmethod
    def get_components():
        return [ComponentDefinition(name="vae", hf_subdir="vae", loading_mode="single",
                                    mapping_getter=QwenWeightMapping.get_vae_mapping)]
    @staticmethod
    def get_download_patterns():
        return ["vae/*.safetensors", "vae/*.json"]


def size_gb(model):
    return sum(v.nbytes for _, v in tree_flatten(model.parameters())) / 1e9


def load_bf16():
    m = SingleStreamDiT(Krea2Config())
    m.load_weights(CKPT, strict=True)
    m.update(tree_map(lambda a: a.astype(mx.bfloat16), m.parameters()))
    mx.eval(m.parameters())
    return m


def cos(a, b):
    a, b = a.ravel().astype(np.float64), b.ravel().astype(np.float64)
    return float((a @ b) / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-12))


def main():
    noise = np.random.RandomState(0).randn(1, 16, H // 8, W // 8).astype(np.float32)
    enc = Qwen3VLConditioner(REPO, dtype=mx.bfloat16)
    vae = QwenVAE()
    vae.update(WeightLoader.load(weight_definition=_VaeDef, model_path=REPO).components["vae"])
    mx.eval(vae.parameters())
    ctx = enc(["a fox in the snow"])  # encode once, reuse

    def gen(model):
        return np.array(sample(model, vae, lambda p: ctx, ["x"], width=W, height=H,
                               steps=STEPS, guidance=0.0, init_noise=noise,
                               dtype=mx.bfloat16).astype(mx.float32))

    m = load_bf16()
    print(f"[bf16] transformer size={size_gb(m):.1f}GB")
    ref = gen(m)
    to_pil(mx.array(ref))[0].save("out/q_bf16.png")
    del m

    for bits in (8, 4):
        m = load_bf16()
        nn.quantize(m, group_size=64, bits=bits, class_predicate=quantize_bulk)
        mx.eval(m.parameters())
        nq = sum(1 for p, mod in m.named_modules() if hasattr(mod, "bits"))
        out = gen(m)
        to_pil(mx.array(out))[0].save(f"out/q_{bits}bit.png")
        print(f"[{bits}bit] size={size_gb(m):.1f}GB  quantized_linears={nq}  "
              f"pixel_cos_vs_bf16(illustrative)={cos(out, ref):.5f}  "
              f"max|diff|={np.abs(out-ref).max():.4f}")
        del m


if __name__ == "__main__":
    main()
