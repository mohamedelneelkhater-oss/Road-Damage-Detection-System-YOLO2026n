"""
Road Damage Detection System - Gradio web app
Model: YOLO26n (fine-tuned on a Roboflow road damage dataset)
"""

from pathlib import Path

import gradio as gr
import torch
from PIL import Image
from ultralytics import YOLO

# --------------------------------------------------------------------------- #
# Configuration
# --------------------------------------------------------------------------- #
MODEL_PATH = Path("weights/best.pt")
IMG_SIZE = 640
DEFAULT_CONF = 0.25
DEFAULT_IOU = 0.45

# Use the GPU when available, otherwise fall back to the CPU.
DEVICE = 0 if torch.cuda.is_available() else "cpu"
USE_HALF = DEVICE != "cpu"  # FP16 is only supported on GPU

if not MODEL_PATH.exists():
    raise FileNotFoundError(
        f"Model weights not found at '{MODEL_PATH}'. "
        "Place your trained best.pt file in the 'weights/' folder."
    )

model = YOLO(str(MODEL_PATH))


# --------------------------------------------------------------------------- #
# Inference
# --------------------------------------------------------------------------- #
def predict(image: Image.Image, conf: float, iou: float):
    """Run detection on a single image and return the annotated result."""
    if image is None:
        raise gr.Error("Please upload a road image first.")

    results = model.predict(
        source=image,
        imgsz=IMG_SIZE,
        conf=conf,
        iou=iou,
        device=DEVICE,
        half=USE_HALF,
        verbose=False,
    )
    result = results[0]

    # result.plot() returns a BGR array, so reverse the channels to get RGB.
    annotated = Image.fromarray(result.plot()[..., ::-1])

    # Build a short summary of what was detected.
    if len(result.boxes) == 0:
        summary = "No road damage detected at the current confidence threshold."
    else:
        counts = {}
        for cls_id in result.boxes.cls.int().tolist():
            name = model.names[cls_id]
            counts[name] = counts.get(name, 0) + 1
        summary = "\n".join(f"{name}: {n}" for name, n in sorted(counts.items()))

    return annotated, summary


# --------------------------------------------------------------------------- #
# User interface
# --------------------------------------------------------------------------- #
demo = gr.Interface(
    fn=predict,
    inputs=[
        gr.Image(type="pil", label="Upload a road image"),
        gr.Slider(0.05, 0.95, value=DEFAULT_CONF, step=0.05, label="Confidence threshold"),
        gr.Slider(0.10, 0.95, value=DEFAULT_IOU, step=0.05, label="NMS IoU threshold"),
    ],
    outputs=[
        gr.Image(type="pil", label="Detection result"),
        gr.Textbox(label="Detected damage (count per class)", lines=6),
    ],
    title="Road Damage Detection System (YOLO26n)",
    description=(
        "Upload a road image to detect potholes, cracks, and other road defects. "
        "Adjust the confidence threshold to trade off precision and recall."
    ),
    flagging_mode="never",
)

if __name__ == "__main__":
    demo.launch()
