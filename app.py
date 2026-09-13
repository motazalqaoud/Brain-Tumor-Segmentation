"""
Gradio demo for brain-tumor-segmentation-3d (Hugging Face Spaces entry point).

Real 3D volumetric input/output: upload a 4-modality NIfTI volume, get a full
3D segmentation mask back, visualized as an axial slice with the predicted
tumor regions overlaid (a full interactive 3D viewer isn't practical in a
lightweight web demo, but the underlying inference is genuinely volumetric --
sliding-window inference over the entire 3D volume, not a 2D slice classifier).

If checkpoints/best_model.pth is absent -- the state of this repo out of the
box, since training needs a GPU and the ~7GB dataset -- the app runs in a
clearly-labeled DEMO MODE: the full pipeline runs for real (NIfTI loading,
resampling, sliding-window 3D inference), but the network is untrained, so
the segmentation shown is not meaningful.

Note: CPU-tier Hugging Face Spaces will be slow for 3D sliding-window
inference (potentially 1-2 minutes per volume). A GPU-tier Space or running
locally/on the HPC is strongly recommended for anything beyond a quick demo.
"""

import os
import sys

import gradio as gr
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "src"))
from dataset import inference_transforms, REGION_NAMES  # noqa: E402
from model import load_model  # noqa: E402
from monai.inferers import sliding_window_inference  # noqa: E402
from monai.transforms import Activations, AsDiscrete, Compose  # noqa: E402

CHECKPOINT_PATH = os.path.join(os.path.dirname(__file__), "checkpoints", "best_model.pth")
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
ROI_SIZE = (128, 128, 128) if torch.cuda.is_available() else (64, 64, 64)

model, WEIGHTS_LOADED = load_model(checkpoint_path=CHECKPOINT_PATH, device=DEVICE)
post_pred = Compose([Activations(sigmoid=True), AsDiscrete(threshold=0.5)])
transform = inference_transforms()

MODE_BANNER = (
    "✅ **Trained model loaded** — this segmentation reflects a model fine-tuned on real BraTS-format volumes."
    if WEIGHTS_LOADED else
    "⚠️ **Demo mode — no trained checkpoint found.** The full 3D pipeline (NIfTI "
    "loading, resampling, sliding-window inference over the entire volume) runs "
    "for real, but the network is untrained, so the segmentation below isn't "
    "meaningful. Train on the HPC with `src/train.py` (see README) and this "
    "Space will automatically switch to real predictions once "
    "`checkpoints/best_model.pth` is uploaded."
)


def render_slice_overlay(image_vol, seg_masks, slice_frac=0.5):
    """image_vol: (4,H,W,D) input; seg_masks: (3,H,W,D) binary TC/WT/ET masks."""
    depth = image_vol.shape[-1]
    z = int(depth * slice_frac)

    base = image_vol[1, :, :, z]  # T1w channel for the background slice
    base = (base - base.min()) / (base.max() - base.min() + 1e-8)

    fig, ax = plt.subplots(figsize=(5, 5))
    ax.imshow(base.T, cmap="gray", origin="lower")

    colors = {"WT": (1, 1, 0, 0.25), "TC": (1, 0.5, 0, 0.35), "ET": (1, 0, 0, 0.5)}
    for name, mask in zip(REGION_NAMES, seg_masks):
        region_slice = mask[:, :, z]
        overlay = np.zeros((*region_slice.shape, 4))
        overlay[region_slice > 0] = colors[name]
        ax.imshow(overlay.transpose(1, 0, 2), origin="lower")

    ax.axis("off")
    ax.set_title(f"Axial slice {z}/{depth} — yellow=WT, orange=TC, red=ET")
    fig.tight_layout()

    fig.canvas.draw()
    buf = np.asarray(fig.canvas.buffer_rgba())
    plt.close(fig)
    return buf


def segment(nifti_file, slice_position):
    if nifti_file is None:
        return None, MODE_BANNER

    data = transform({"image": nifti_file})
    input_tensor = data["image"].unsqueeze(0).to(DEVICE)

    with torch.no_grad():
        output = sliding_window_inference(input_tensor, roi_size=ROI_SIZE, sw_batch_size=1,
                                           predictor=model, overlap=0.5)
        output = post_pred(output.squeeze(0)).cpu().numpy()

    image_vol = data["image"].numpy()
    overlay_img = render_slice_overlay(image_vol, output, slice_frac=slice_position)
    return overlay_img, MODE_BANNER


with gr.Blocks(title="Brain Tumor Segmentation 3D") as demo:
    gr.Markdown("# Brain Tumor Segmentation — Real 3D Volumetric Input")
    gr.Markdown(
        "Upload a 4-modality NIfTI volume (MSD Task01_BrainTumour layout: "
        "FLAIR, T1w, T1gd, T2w stacked as a single 4D `.nii.gz`). Inference "
        "runs on the **full 3D volume** via sliding-window prediction, not a "
        "2D slice classifier."
    )
    status = gr.Markdown(MODE_BANNER)

    with gr.Row():
        with gr.Column():
            file_input = gr.File(label="4-modality NIfTI volume (.nii.gz)", file_types=[".gz", ".nii"])
            slice_slider = gr.Slider(0.1, 0.9, value=0.5, label="Axial slice position (fraction through volume)")
            submit_btn = gr.Button("Segment", variant="primary")
        with gr.Column():
            output_image = gr.Image(label="Segmentation overlay (axial slice)")

    submit_btn.click(fn=segment, inputs=[file_input, slice_slider], outputs=[output_image, status])

    gr.Markdown(
        "---\n"
        "**This is a research and portfolio tool, not a medical device.** Not "
        "validated for clinical use. Any actual diagnosis requires a "
        "radiologist and the full clinical picture.\n\n"
        "Dataset: Medical Segmentation Decathlon Task01_BrainTumour "
        "([medicaldecathlon.com](http://medicaldecathlon.com/)), sourced from "
        "the BraTS 2016/2017 challenge."
    )

if __name__ == "__main__":
    demo.launch()
