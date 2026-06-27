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

**You need:** an **Apple-silicon Mac** (M1/M2/M3/M4) with **≥ 24 GB unified memory** (**32 GB+
recommended**; 16 GB will run out of memory at 1024² — drop to `--width/--height 512` if you must),
**Python 3.10+**, and **~25 GB free disk** for the 8-bit build (bf16 needs ~40 GB).

> 🍎 **On macOS the commands are `python3` / `pip3`, not `python` / `pip`.** No Python yet?
> Install it from [python.org](https://www.python.org/downloads/macos/) (or `brew install python`),
> then reopen Terminal.

### Option A — Web UI (easiest)

```bash
git clone https://github.com/avlp12/krea2_alis_mlx
cd krea2_alis_mlx
python3 -m pip install -r requirements.txt
python3 app.py
```

Your browser opens at **http://localhost:7860** — type a prompt, click **Generate**. ✨

> **First run** downloads the model (8-bit, ~14 GB) + the text encoder/VAE + a small NSFW
> safety classifier automatically, so it takes a few minutes. After that it's cached and instant
> to start. A 1024×1024 image takes **~50 s on an M3 Ultra** (8 steps; slower chips take longer).

### Option B — Command line

```bash
python3 generate.py "a red fox in the snow, photorealistic" --out fox.png
```

Useful flags: `--width/--height 512|768|1024`, `--steps 8`, `--seed 0`, `--num-images 2`.

> **Choose your build:** in the web UI, pick **8-bit** or **mixed-4/8** from the **Model**
> dropdown — it downloads the chosen one on first use. On the CLI, add `--precision mixed-4-8`
> (or `8bit`). Default is 8-bit.

---

## 📦 Models

Two quantized **MLX builds** are published on Hugging Face — the code here downloads the one
you pick:

| Build | Transformer | Quality | When to use |
|---|---|---|---|
| [**8-bit**](https://huggingface.co/avlp12/Krea-2-Turbo-Alis-MLX-8bit) *(default)* | 14.2 GB | near-lossless (vel-cos 0.99996) | best fidelity |
| [**mixed-4/8**](https://huggingface.co/avlp12/Krea-2-Turbo-Alis-MLX-mixed-4-8) | 9.8 GB | near-lossless (0.99849) | smaller download |

<sub>*vel-cos = mean per-step velocity cosine vs the bf16 reference on a fixed trajectory, over 3 prompts × 8 steps (24 samples); reproduce with [`validation/validate_quant.py`](validation/validate_quant.py). Worst-case-min: 8-bit 0.99988, mixed-4/8 0.99229.*</sub>

> **What about bf16?** There's **no separate bf16 build** — `--precision bf16` runs **Krea's
> original weights** (`turbo.safetensors`, ~24 GB) through this *same* MLX code, pulled from
> [`krea/Krea-2-Turbo`](https://huggingface.co/krea/Krea-2-Turbo). It's the full-precision
> reference, but the quantized builds are near-lossless **and all builds run at ≈ the same speed**
> (generation is attention-bound — quantization only shrinks the download, it doesn't speed things
> up). So 8-bit / mixed-4/8 are the practical picks; reach for bf16 only to double-check fidelity.

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

> Why is the VAE's 0.9994 *lower* than the full pipeline's 1.000000? The VAE was tested on a
> **random** latent — an out-of-distribution torture test that amplifies tiny float differences at
> the saturating extremes (and the sampler clamps those anyway). On the **real** latents the
> pipeline actually produces, agreement rounds to 1.000000.

Reproduce it yourself: see [`validation/`](validation/).

---

## 🧠 What's inside

- **Transformer** `SingleStreamDiT` (12.9B): 28 blocks, GQA, per-head QK-norm, sigmoid output
  gate, SwiGLU, 3-axis RoPE, + a `text_fusion` adapter over 12 encoder layers. Pure MLX.
- **Text encoder** Qwen3-VL-4B (text-only), pure MLX.
- **VAE** Qwen-Image VAE — reused from [mflux](https://github.com/filipstrand/mflux).
- **Sampler** flow-matching Euler, 8-step Turbo (no CFG).
- **Safety** NSFW content filter (Falconsai/nsfw_image_detection) — on by default; redacts
  flagged outputs (see *Safety* below).

`krea2/` is the implementation; `app.py` / `generate.py` are the entry points; `validation/`
is the verification harness; `docs/PORT_PLAN.md` is the build journal.

---

## 🛠️ Troubleshooting

- **`zsh: command not found: python` / `pip`** → on macOS use **`python3`** and **`python3 -m pip`**.
- **First run downloads a lot** (~14 GB model + ~8 GB encoder/VAE) over the HF CDN — give it a
  few minutes; you'll see a `↓ … MB / MB` progress line. The app downloads via plain HTTP on
  purpose (it avoids HuggingFace's Xet client, which can **hang** behind some firewalls/ISPs).
- **Generation feels slow** → a 1024² image is ~50 s (8 steps) per image; the web UI shows a
  live step progress bar. Quantization doesn't speed generation up (it's attention-bound) — it
  only shrinks the download.
- **`Address already in use` (port 7860)** → another app has the port. Run on another one:
  `GRADIO_SERVER_PORT=7861 python3 app.py`.
- **Reach the UI from another device on your network** → the server binds to `127.0.0.1`
  (your Mac only) by default. To expose it on your LAN: `KREA2_HOST=0.0.0.0 python3 app.py`
  — only do this on networks you trust (anyone on the network can then generate images).
- **Out of memory / very slow + swapping** → you likely have < 24 GB RAM; use `--width/--height 512`
  or the smaller `mixed-4/8` build.
- **Verify your weights downloaded intact** → check the SHA-256 of the transformer file:
  ```bash
  shasum -a 256 transformer_8bit.safetensors       # b10f33f0dcd91772990e7cecfc8003ba4d3f1ba27f03010b6d17a1f490f80a6c  (14,244,836,620 bytes)
  shasum -a 256 transformer_mixed_4_8.safetensors  # 985d60722b339c3cd9df16a173f0cb504ae93d81ce9fbe2c3ab158cf5b60a5fb  (9,840,816,670 bytes)
  ```

## 🛡️ Safety &amp; responsible use

An NSFW content filter (**[Falconsai/nsfw_image_detection](https://huggingface.co/Falconsai/nsfw_image_detection)**,
one of the classifiers named in the license) runs **by default** and **redacts** explicit outputs.
It's reimplemented in **pure MLX** (`krea2/nsfw_mlx.py`, validated bit-for-bit against the PyTorch
reference — max |Δ| 2e-7), so it needs **no PyTorch** and works out of the box on a clean install.
It's tuned not to flag ordinary photos (e.g. swimwear). Turn it off with the web-UI checkbox,
`--no-safety` (CLI), or `KREA2_DISABLE_SAFETY=1`; adjust the cutoff with `KREA2_SAFETY_THRESHOLD`
(default 0.85).

The Krea 2 Community License **requires** deployments to implement reasonable content filtering
and to disclose AI-generated content where required by law. If you disable the filter or deploy
publicly, that obligation is **yours**. Do not generate or distribute non-consensual, sexual-abuse,
or otherwise illegal content.

## 📜 License

This code is an independent MLX implementation. The **model weights** are a modified
derivative of [`krea/Krea-2-Turbo`](https://huggingface.co/krea/Krea-2-Turbo) under the
**[Krea 2 Community License](https://krea.ai/krea-2-licensing)** (see [`LICENSE`](LICENSE),
[`NOTICE`](NOTICE)). Notably: commercial use requires total annual revenue **under $1M USD**
(otherwise an enterprise license from Krea), deployments must implement reasonable content
filtering (a built-in NSFW filter is included — see *Safety* above), and it's **not endorsed by
Krea**. You own the images you generate.

## 🙏 Credits

[Krea.ai](https://www.krea.ai) (base model) · [Qwen](https://github.com/QwenLM) (VAE + text encoder) ·
[mflux](https://github.com/filipstrand/mflux) (MLX diffusion framework).

*Part of the **Alis** MLX line — see also [`lance_alis_mlx`](https://github.com/avlp12/lance_alis_mlx).*
