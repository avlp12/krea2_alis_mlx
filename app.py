#!/usr/bin/env python
"""Local web UI for Krea-2-Turbo on Apple MLX.

    pip install mlx transformers "mflux>=0.18,<0.19" huggingface_hub gradio
    python app.py            # opens http://localhost:7860

Auto-detects the weights shipped in this folder (8-bit or mixed-4/8). The VAE /
Qwen3-VL-4B encoder / tokenizer are pulled from krea/Krea-2-Turbo on first run.
Set KREA2_BASE_DIR to a local Krea-2-Turbo snapshot to skip that download.
"""

import os

import gradio as gr

from krea2.pipeline import Krea2Pipeline, resolve_weights

HERE = os.path.dirname(os.path.abspath(__file__))
SIZES = ["512", "768", "1024"]
# selectable builds (label shown in the UI -> precision)
MODELS = [("8-bit · best quality (14 GB)", "8bit"), ("mixed-4/8 · smaller (9.8 GB)", "mixed-4-8")]
_PIPE = None
_PIPE_PREC = None


def _pipe(precision):
    """Return a pipeline for the chosen build, (re)loading it if the build changed.
    Downloads the build from HF on first use if it isn't already local."""
    global _PIPE, _PIPE_PREC
    if _PIPE is None or _PIPE_PREC != precision:
        prec, path = resolve_weights(HERE, precision=precision, download=True)
        _PIPE = Krea2Pipeline(path, precision=prec, base_dir=os.environ.get("KREA2_BASE_DIR"))
        _PIPE_PREC = prec
    return _PIPE


def generate(prompt, model, size, steps, seed, num_images):
    if not prompt or not prompt.strip():
        raise gr.Error("Enter a prompt.")
    s = int(size)
    return _pipe(model).generate(prompt.strip(), width=s, height=s, steps=int(steps),
                                 seed=int(seed), num_images=int(num_images))


with gr.Blocks(title="Krea 2 Turbo · Alis MLX") as demo:
    default_prec, _ = resolve_weights(HERE, download=False)  # the build already in this folder, if any
    gr.Markdown("# Krea&nbsp;2&nbsp;Turbo · Alis MLX\n"
                "Local text-to-image on Apple silicon · 8-step Turbo (no CFG). "
                "Switching **Model** downloads that build on first use.")
    with gr.Row():
        with gr.Column(scale=1):
            prompt = gr.Textbox(label="Prompt", lines=3, value="a fox in the snow")
            model = gr.Dropdown(MODELS, value=default_prec, label="Model")
            with gr.Row():
                size = gr.Dropdown(SIZES, value="1024", label="Size")
                steps = gr.Slider(4, 12, value=8, step=1, label="Steps")
            with gr.Row():
                seed = gr.Number(value=0, label="Seed", precision=0)
                num_images = gr.Slider(1, 4, value=1, step=1, label="Images")
            btn = gr.Button("Generate", variant="primary")
            gr.Examples(
                [["a fox in the snow"],
                 ["a neon city street at night in the rain, reflections"],
                 ["a close-up portrait of an old fisherman, weathered face"]],
                inputs=prompt,
            )
        with gr.Column(scale=1):
            gallery = gr.Gallery(label="Output", columns=2, height=560, object_fit="contain")
    btn.click(generate, [prompt, model, size, steps, seed, num_images], gallery)


if __name__ == "__main__":
    demo.launch(theme=gr.themes.Soft())
