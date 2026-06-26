# Validation harness

The scripts that verified this port against the **original PyTorch reference**
([`krea-ai/krea-2`](https://github.com/krea-ai/krea-2)) — every stage was cross-checked
before the next was built. These are the exact scripts used; they're for reproducing the
numbers in the model card, not for normal use (for that, see the top-level `app.py` / `generate.py`).

## Setup

```bash
pip install -r ../requirements.txt
pip install torch diffusers einops accelerate          # validation-only extras
git clone https://github.com/krea-ai/krea-2 krea-2-official   # PyTorch reference (run from repo root)
# base weights: a local krea/Krea-2-Turbo snapshot at ./weights/Krea-2-Turbo
huggingface-cli download krea/Krea-2-Turbo --local-dir weights/Krea-2-Turbo
```

Run from the **repo root** (so `import krea2` and the relative paths resolve), e.g.
`python validation/validate_e2e.py`.

## What each script checks

| Script | Checks | Reported result |
|---|---|---|
| `validate_encoder.py` | MLX Qwen3-VL encoder vs HF, 12 tapped layers | cos 1.000000 |
| `validate_transformer.py` | MLX transformer velocity vs PyTorch (float32) | cos 1.000000, rel-L2 3e-5 |
| `validate_vae.py` | MLX VAE decode vs 🤗 diffusers | cos 0.9994 |
| `validate_e2e.py` | full pipeline, MLX vs PyTorch, identical noise | pixel cos 1.000000 |
| `validate_quant.py` | per-step velocity cos vs bf16 (8-bit / mixed-4/8 / 4-bit) | see model card |
| `eval_mxfp.py` | MXFP4 / MXFP8 vs affine (why MXFP was rejected) | see model card |
| `test_*.py` | structural / weight-load / VAE-decode smoke tests | — |
| `build_release*.py` | build the 8-bit / mixed-4/8 HF artifacts | — |

PyTorch runs on CPU (the reference `rope` uses float64, which Apple MPS lacks).
