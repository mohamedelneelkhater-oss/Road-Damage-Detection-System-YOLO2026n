from pathlib import Path

import gradio as gr
import torch
from PIL import Image
from ultralytics import YOLO

# Change this path if your folder is somewhere else
BASE_DIR = Path.home() / "Documents" / "road_damage_yolo"
MODELS_DIR = BASE_DIR / "models"
SAMPLES_DIR = BASE_DIR / "samples"

DEVICE = 0 if torch.cuda.is_available() else "cpu"
USE_HALF = DEVICE != "cpu"

model_paths = {p.stem: p for p in sorted(MODELS_DIR.glob("*.pt"))}
if not model_paths:
    raise FileNotFoundError(f"No .pt files found in {MODELS_DIR}")

loaded_models = {}


def get_model(name):
    if name not in loaded_models:
        loaded_models[name] = YOLO(str(model_paths[name]))
    return loaded_models[name]


def predict_image(img, model_name, conf):
    if img is None:
        return None

    results = get_model(model_name).predict(
        source=img, imgsz=640, device=DEVICE, half=USE_HALF, conf=conf
    )
    annotated = results[0].plot()[..., ::-1]  # BGR -> RGB
    return Image.fromarray(annotated)


sample_images = sorted(
    str(p) for ext in ("jpg", "jpeg", "png") for p in SAMPLES_DIR.glob(f"*.{ext}")
)

with gr.Blocks(title="Road Damage Detection System") as demo:
    gr.Markdown("# Road Damage Detection System")
    gr.Markdown("Choose a model, upload an image or pick a sample, then press **Detect**.")

    with gr.Row():
        model_choice = gr.Dropdown(
            choices=list(model_paths), value=list(model_paths)[0], label="Model"
        )
        conf_slider = gr.Slider(0.05, 0.95, value=0.25, step=0.05, label="Confidence")

    with gr.Row():
        input_img = gr.Image(type="pil", label="Input Image")
        output_img = gr.Image(type="pil", label="Detection Result")

    detect_btn = gr.Button("Detect", variant="primary")
    detect_btn.click(predict_image, [input_img, model_choice, conf_slider], output_img)

    if sample_images:
        gr.Examples(examples=sample_images, inputs=input_img, label="Sample Images")

if __name__ == "__main__":
    demo.launch()