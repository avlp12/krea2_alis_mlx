#!/usr/bin/env python
"""Generate images with Krea-2-Turbo on Apple MLX.

    pip install mlx transformers mflux huggingface_hub
    python generate.py "a fox in the snow"

8-bit (default) uses this repo's transformer_8bit.safetensors. Use --precision bf16
to run the full-precision transformer from krea/Krea-2-Turbo instead.
"""

import argparse
import os

from krea2.pipeline import Krea2Pipeline, resolve_weights


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("prompt")
    ap.add_argument("--precision", choices=["8bit", "mixed-4-8", "bf16"], default=None,
                    help="default: auto-detect the weights shipped in this folder")
    ap.add_argument("--transformer", default=None,
                    help="path to transformer weights (default: the matching file in this repo)")
    ap.add_argument("--width", type=int, default=1024)
    ap.add_argument("--height", type=int, default=1024)
    ap.add_argument("--steps", type=int, default=8)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--num-images", type=int, default=1)
    ap.add_argument("--out", default="out.png")
    args = ap.parse_args()

    here = os.path.dirname(os.path.abspath(__file__))
    precision, tpath = args.precision, args.transformer
    if precision is None:  # local weights if present, else fetch the 8-bit build from HF
        precision, tpath = resolve_weights(here, download=True)
    elif tpath is None and precision != "bf16":
        fname = "transformer_8bit.safetensors" if precision == "8bit" else "transformer_mixed_4_8.safetensors"
        tpath = os.path.join(here, fname)

    pipe = Krea2Pipeline(transformer_path=tpath, precision=precision)
    images = pipe.generate(args.prompt, width=args.width, height=args.height,
                           steps=args.steps, seed=args.seed, num_images=args.num_images)
    base, ext = os.path.splitext(args.out)
    ext = ext or ".png"
    for i, im in enumerate(images):
        path = args.out if len(images) == 1 else f"{base}_{i}{ext}"
        im.save(path)
        print(f"saved {path}")


if __name__ == "__main__":
    main()
