"""Parity check: pure-MLX NSFW ViT classifier vs the PyTorch (transformers) reference.

Runs both on the same images and compares P(nsfw). The MLX port must match the torch
pipeline to within a small tolerance, else the safety filter would mis-flag. torch is only
needed HERE (validation); the shipped safety path is torch-free.
"""

import os
import sys

import numpy as np
from PIL import Image

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from krea2.nsfw_mlx import load_classifier, nsfw_score  # noqa: E402

MODEL_DIR = os.path.expanduser("~/.cache/krea2_alis_mlx/Falconsai__nsfw_image_detection")


def make_images():
    rng = np.random.RandomState(0)
    imgs = {
        "solid_gray": Image.new("RGB", (512, 512), (128, 128, 128)),
        "noise": Image.fromarray(rng.randint(0, 256, (300, 400, 3), dtype=np.uint8)),
        "gradient": Image.fromarray(np.tile(np.linspace(0, 255, 256, dtype=np.uint8)[:, None, None], (1, 256, 3))),
    }
    fox = "/private/tmp/claude-501/-Users-gesicht-local-claude-code/25b3adad-e271-4ad5-aab1-02aaea6ff8c4/scratchpad/smoke/fox.png"
    if os.path.exists(fox):
        imgs["fox(real-gen)"] = Image.open(fox)
    return imgs


def main():
    imgs = make_images()

    # torch reference
    from transformers import pipeline
    ref = pipeline("image-classification", model=MODEL_DIR)

    def ref_nsfw(im):
        return {d["label"].lower(): d["score"] for d in ref(im)}.get("nsfw", 0.0)

    # MLX port
    model = load_classifier(MODEL_DIR)

    print(f"{'image':16s} {'torch P(nsfw)':>14s} {'mlx P(nsfw)':>12s} {'|diff|':>10s}")
    worst = 0.0
    for name, im in imgs.items():
        t = ref_nsfw(im)
        m = nsfw_score(model, im)
        d = abs(t - m)
        worst = max(worst, d)
        print(f"{name:16s} {t:14.6f} {m:12.6f} {d:10.2e}")
    print(f"\nworst |diff| = {worst:.2e}  ->  {'PASS' if worst < 2e-3 else 'FAIL'}")
    return 0 if worst < 2e-3 else 1


if __name__ == "__main__":
    sys.exit(main())
