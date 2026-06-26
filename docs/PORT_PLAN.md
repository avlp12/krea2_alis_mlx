# Krea-2-Turbo → MLX port (extend mflux)

**Goal:** native MLX text-to-image for `krea/Krea-2-Turbo`, integrated into mflux. bf16-correct first, then quantize.
**Hardware:** M3 Ultra 512GB (bf16 ~35GB fits trivially; quant is for speed only).

## Model (12.9B single-stream MMDiT, flow-matching)
- **Transformer** `Krea2Transformer2DModel` / official `SingleStreamDiT` (krea-2-official/mmdit.py — ground truth).
  - 28 blocks, width 6144 (48 q-heads × 128), GQA 12 kv-heads, per-head QK-RMSNorm, sigmoid output gate, SwiGLU(16384), 3-axis RoPE axes=[32,48,48] θ=1000.
  - `text_fusion`: 2 layerwise blocks over (B·12layers, n_tok, 2560) → Linear(12→1) projector → 2 refiner blocks (seq-axis, masked). heads=20, headdim=128.
  - tokens concatenated as **[txt ; img]**; output sliced back to img region.
  - mixed precision in ckpt: norms/embeds/mod = F32, big matmuls = BF16. Reference casts whole model to bf16 at inference; RMSNorm upcasts to f32 internally (weight = stored `scale` + 1.0, eps=1e-5).
- **Text encoder** `Qwen3-VL-4B-Instruct`, text-only. Fixed prompt template, output_hidden_states, stack 12 layers [2,5,…,35] → (B,seq,12,2560), strip 34-tok prefix. (encoder.py)
- **VAE** `AutoencoderKLQwenImage` (Qwen-Image VAE). **decode only** for txt2img. → reuse mflux `QwenVAE`.
- **Sampling** flow-matching Euler `x += (t_prev−t_curr)·v`. Turbo = 8 steps, guidance 0 (no CFG), distilled. (sampling.py)

## Reuse map (mflux)
| Piece | Source | Effort |
|---|---|---|
| VAE decode | `mflux.models.qwen…qwen_vae.QwenVAE` (latents_mean/std identical) | reuse as-is |
| Scheduler | flow-match Euler — port the ~10-line `timesteps()` from sampling.py | trivial |
| Text encoder | Qwen3-VL-4B text model — port (mlx-vlm Qwen3-VL as ref; differs from mflux's Qwen2.5-VL) | medium |
| Transformer | `SingleStreamDiT` — port from mmdit.py | core work |

## Checkpoint
Single file `turbo.safetensors` (430 tensors), official naming == mmdit.py module paths → **no weight remap** if MLX modules named identically. (Repo also has diffusers-named shards + bundled Qwen3-VL text_encoder + VAE.)
Downloaded to `weights/Krea-2-Turbo/`.

## Phases
- [x] P0 understand + env (mlx/mflux/transformers, clone refs, download weights)
- [x] P1 **transformer** MLX port (`krea2/transformer.py`) — VERIFIED: 12.82B params, param names/shapes EXACT match to turbo.safetensors (430/430, 0 missing/extra/mismatch, zero remap), random-init forward → correct (B,Limg,64) bf16 output. Test: `scripts/test_transformer_struct.py`. (Numerical match vs reference = P5.)
- [~] P2 text encoder (Qwen3-VL-4B text, 12-layer hidden states, prompt template, tokenizer)
      - [x] P2a HF-transformers oracle (`text_encoder_hf.py`) — drives e2e image; numerical oracle for P2b.
      - [x] P2b pure-MLX Qwen3 text model (`text_encoder.py`) — VERIFIED cos=1.000000 vs HF oracle across all 12 layers (`scripts/validate_encoder.py`), 398/398 tensors. Text-only ⇒ standard rope.
- [x] P3 VAE wiring — reuse mflux `QwenVAE` + `QwenWeightMapping.get_vae_mapping` via `Krea2WeightDefinition`. VERIFIED: 192/192 params loaded, pixel cos=0.9994 vs PT diffusers on random latent (max|diff| 0.31 at extremes = OOD random-latent drift, sampler clamps anyway; re-check real latents P5). Test: `scripts/validate_vae.py`.
- [x] P4 sampling loop + patchify/positions (`sampling.py`) — patchify∘unpatchify==identity verified; first e2e MLX image 1024² 8-step 55s (`out/fox_1024.png`, `scripts/generate.py`). CLI `mflux-generate-krea2` = TODO.
- [x] P5 **numerical validation** vs PyTorch reference — ALL PASS: transformer velocity cos=1.000000 (rel_l2=3e-5); encoder cos=1.000000; VAE cos=0.9994; **full-native e2e pixel cos=1.000000, max|diff|=0.005** (`scripts/validate_e2e.py`, 512² 8-step, identical noise). Visually identical (`out/e2e_{mlx,pt}_512.png`). Pure-MLX pipeline is faithful to the PT reference.
- [x] P6 quantize — recipe in `quant_recipes.py`, wired into `Krea2WeightDefinition.quantization_predicate` (eval==ship). Sensitivity-graded: quantize only 28-block attn+mlp bulk (224 Linears); keep first/last/tmlp/tproj/txtfusion/txtmlp + norms/modulation bf16.
      3-LENS REVIEW DONE (codex: no bugs; red-team + blank-slate converged → fixed the validation). Honest metric = per-step velocity cos vs bf16 on fixed trajectory (3 prompts×8 steps):
      - **8-bit** 14.2GB, vel-cos mean 0.99996 (near-lossless) — distribution default.
      - **mixed-4/8** (down_proj+endpoints@8) 9.8GB, vel-cos min 0.9923 — best "small" variant (beats plain 4-bit min 0.9885 @8.2GB).
      - **DECISION:** latency identical across bf16(5953ms)/8-bit/4-bit(~6050ms) @1024² → **quant gives NO speedup here** (attention-bound, unquantized) & 512GB ⇒ memory not a constraint. ⇒ **bf16 is the right inference path for this user**; quant value = portability/HF release only.
      - TODO if releasing: CLIP/multi-prompt FID, review Krea 2 Community License re: redistribution, card PNG-not-SVG.

## Validation strategy (P5) — Lance t2i methodology, hard-learned
Reference = krea-2-official on torch CPU/MPS (patch out CUDA-only `sdpa_kernel(CUDNN_ATTENTION)` + device="cuda"). Layered parity, **each pipeline fed its OWN native inputs from the raw prompt**:
1. encoder hidden states: MLX Qwen3-VL vs PT, per selected layer (cos, expect >0.999)
2. one transformer block output (cos >0.999)
3. single-step velocity v (cos >0.999)
4. full 8-step latent + decoded pixels (cos) + visual side-by-side
5. reproduce the repo's official example prompts/images

Rules (from Lance — do not repeat):
- **NO quality claim from vibes/eyeballing.** Side-by-side + cosine only. (Lance: claimed "official-grade" twice w/o pixel compare → user disappointed.)
- **Blind-test trap (#1170):** a harness feeding BOTH PT & MLX the same intermediate tensor can report cos=1.0 while a shared bug hides. Drive each pipeline from the RAW prompt. → the **HF-oracle tactic (P2) is an ISOLATION test of the transformer only, NOT end-to-end**; validate the MLX encoder separately + do a full-native run before any quality claim.
- **Interpretation matrix:** PT≈MLX & both good → port correct. PT good & MLX bad → MLX bug. PT≈MLX & both bad → model/setting limit (Krea-2 is commercial-grade → "both bad" unlikely → suspect bug or turbo mu/steps).
- Patchify/position-ids are classic bug sources → unit-test patchify∘unpatchify==identity and vs PT.

## Work mode (Lance #1165 — overrides ultracode default)
INLINE-first: porting, ref reading, harness, validation done inline so the user follows the flow. Parallel/Workflow only as narrow exception — (a) 3-lens [[critical-review-before-decisions]] at major decisions, (b) adversarial re-verify a surprising cos. No default fan-out.

## Open questions / gotchas
- Turbo timestep schedule: scheduler_config has dynamic shifting base_shift 0.5 / max_shift 1.15; README example passes only steps=8, cfg=0 (no mu). Confirm whether turbo uses resolution-aware mu or fixed 1.15. → pin during P5.
- GQA in MLX sdpa: repeat kv heads 12→48 for correctness-first (optimize later).
- GELU = tanh approximation (PyTorch approximate='tanh'); use explicit 0.5x(1+tanh(√(2/π)(x+0.044715x³))).
- 256-seq padding in mmdit.py is a torch.compile perf detail — skip in MLX (masked + sliced anyway, numerically identical).
- License: Krea 2 Community License (LICENSE.pdf in repo) — review redistribution terms before any public quant upload (P6).

## Paths
- refs: `krea-2-official/` (mmdit.py, encoder.py, autoencoder.py, sampling.py, inference.py), `mflux/`
- weights: `weights/Krea-2-Turbo/`
- code: `krea2/`
- venv: `.venv` (uv)
