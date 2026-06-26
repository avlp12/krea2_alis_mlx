# Krea 2 Turbo — Apple MLX port (Alis)

Run **[Krea 2 Turbo](https://www.krea.ai/blog/krea-2-technical-report)** — a 12.9B
text-to-image model — **locally on your Mac**, with a one-command web UI. Pure
[Apple MLX](https://github.com/ml-explore/mlx), numerically validated **faithful to the
original PyTorch** (end-to-end pixel cosine **1.000000**).

[![8-bit](https://img.shields.io/badge/🤗%20model-8--bit%20(14GB)-orange)](https://huggingface.co/avlp12/Krea-2-Turbo-Alis-MLX-8bit)
[![mixed-4/8](https://img.shields.io/badge/🤗%20model-mixed--4%2F8%20(9.8GB)-orange)](https://huggingface.co/avlp12/Krea-2-Turbo-Alis-MLX-mixed-4-8)
[![license](https://img.shields.io/badge/license-Krea%202%20Community-blue)](https://krea.ai/krea-2-licensing)

![pipeline](assets/pipeline.png)

![samples](assets/samples.png)

> *Independent, unofficial port — not affiliated with or endorsed by Krea.*

---

## ⚡ Quick start (beginners — copy & paste)

**You need:** an **Apple-silicon Mac** (M1/M2/M3/M4), **Python 3.10+**, and ~20 GB free disk.

### Option A — Web UI (easiest)

```bash
git clone https://github.com/avlp12/krea2_alis_mlx
cd krea2_alis_mlx
pip install -r requirements.txt
python app.py
```

Your browser opens at **http://localhost:7860** — type a prompt, click **Generate**. ✨

> **First run** downloads the model (8-bit, ~14 GB) + the text encoder/VAE automatically,
> so it takes a few minutes. After that it's cached and instant to start. A 1024×1024 image
> takes ~50 s (8 steps).

### Option B — Command line

```bash
python generate.py "a red fox in the snow, photorealistic" --out fox.png
```

Useful flags: `--width/--height 512|768|1024`, `--steps 8`, `--seed 0`, `--num-images 2`.

> **Tip:** prefer the smaller, near-identical build? `pip install huggingface_hub` then
> `hf download avlp12/Krea-2-Turbo-Alis-MLX-mixed-4-8 --local-dir .` before running — the
> apps auto-detect whichever weights are in the folder.

---

## 📦 Models

The weights live on Hugging Face (the code here downloads them for you). Pick one:

| Build | Size | Quality | When to use |
|---|---|---|---|
| [**8-bit**](https://huggingface.co/avlp12/Krea-2-Turbo-Alis-MLX-8bit) *(default)* | 14.2 GB | near-lossless (vel-cos 0.99996) | best fidelity |
| [**mixed-4/8**](https://huggingface.co/avlp12/Krea-2-Turbo-Alis-MLX-mixed-4-8) | 9.8 GB | near-lossless (0.99849) | smaller download |
| bf16 (`--precision bf16`) | ~35 GB | reference | huge-RAM Macs |

All three run at the **same speed** — generation is attention-bound, so quantization only
shrinks the download, it doesn't speed things up. On a big-RAM Mac, bf16 is just as fast.

---

## ✅ Verified faithful to PyTorch

Every stage was cross-checked against the [original PyTorch code](https://github.com/krea-ai/krea-2)
(float32, fixed seed) **before** the next was built:

![validation](assets/validation.png)

| Stage | vs PyTorch | Result |
|---|---|---|
| Text encoder (Qwen3-VL-4B) | hidden states, 12 layers | **cos 1.000000** |
| Transformer (28-block DiT) | velocity field | **cos 1.000000** |
| VAE (Qwen-Image) | decoded pixels | **cos 0.9994** |
| **Full pipeline** | pixels, identical noise | **cos 1.000000** |

Reproduce it yourself: see [`validation/`](validation/).

---

## 🧠 What's inside

- **Transformer** `SingleStreamDiT` (12.9B): 28 blocks, GQA, per-head QK-norm, sigmoid output
  gate, SwiGLU, 3-axis RoPE, + a `text_fusion` adapter over 12 encoder layers. Pure MLX.
- **Text encoder** Qwen3-VL-4B (text-only), pure MLX.
- **VAE** Qwen-Image VAE — reused from [mflux](https://github.com/filipstrand/mflux).
- **Sampler** flow-matching Euler, 8-step Turbo (no CFG).

`krea2/` is the implementation; `app.py` / `generate.py` are the entry points; `validation/`
is the verification harness; `docs/PORT_PLAN.md` is the build journal.

---

## 📜 License

This code is an independent MLX implementation. The **model weights** are a modified
derivative of [`krea/Krea-2-Turbo`](https://huggingface.co/krea/Krea-2-Turbo) under the
**[Krea 2 Community License](https://krea.ai/krea-2-licensing)** (see [`LICENSE`](LICENSE),
[`NOTICE`](NOTICE)). Notably: commercial use requires annual revenue **under $1M USD**, you must
implement reasonable content filtering, and it's **not endorsed by Krea**. You own the images you generate.

## 🙏 Credits

[Krea.ai](https://www.krea.ai) (base model) · [Qwen](https://github.com/QwenLM) (VAE + text encoder) ·
[mflux](https://github.com/filipstrand/mflux) (MLX diffusion framework).

*Part of the **Alis** MLX line — see also [`lance_alis_mlx`](https://github.com/avlp12/lance_alis_mlx).*
