"""
Road Damage Detection System - Streamlit app
Model: YOLO26n (fine-tuned on a Roboflow road damage dataset)
"""

import io
from pathlib import Path

import requests
import streamlit as st
from PIL import Image

st.set_page_config(page_title="Road Damage Detection", page_icon="🛣️", layout="wide")

BASE_DIR = Path(__file__).parent
MODEL_PATH = BASE_DIR / "YOLO26n.pt"
EXAMPLES_DIR = BASE_DIR / "examples"
IMG_SIZE = 640
MAX_DOWNLOAD_BYTES = 10 * 1024 * 1024


@st.cache_resource(show_spinner="Loading YOLO26n model...")
def load_model():
    from ultralytics import YOLO

    if not MODEL_PATH.exists():
        raise FileNotFoundError(f"Model weights not found at '{MODEL_PATH.name}'.")
    return YOLO(str(MODEL_PATH))


def list_examples():
    if not EXAMPLES_DIR.exists():
        return []
    exts = {".jpg", ".jpeg", ".png", ".webp"}
    return sorted(p for p in EXAMPLES_DIR.iterdir() if p.suffix.lower() in exts)


def load_image_from_url(url: str) -> Image.Image:
    if not url.lower().startswith(("http://", "https://")):
        raise ValueError("The link must start with http:// or https://")

    headers = {"User-Agent": "Mozilla/5.0 (RoadDamageDetectionDemo)"}
    response = requests.get(url, headers=headers, timeout=15, stream=True)
    response.raise_for_status()

    data = response.raw.read(MAX_DOWNLOAD_BYTES + 1, decode_content=True)
    if len(data) > MAX_DOWNLOAD_BYTES:
        raise ValueError("The image is larger than 10 MB.")

    try:
        return Image.open(io.BytesIO(data)).convert("RGB")
    except Exception:
        raise ValueError(
            "This link is not a direct image. Right-click the image and choose "
            "'Copy image address'."
        )


def run_detection(model, image: Image.Image, conf: float, iou: float):
    result = model.predict(
        source=image, imgsz=IMG_SIZE, conf=conf, iou=iou, device="cpu", verbose=False
    )[0]
    annotated = Image.fromarray(result.plot()[..., ::-1])

    counts = {}
    for cls_id in result.boxes.cls.int().tolist():
        name = model.names[cls_id]
        counts[name] = counts.get(name, 0) + 1
    return annotated, counts


st.title("🛣️ Road Damage Detection System")
st.caption("Powered by YOLO26n | Detects potholes, cracks, and other road defects")

examples = list_examples()
modes = ["Upload image"]
if examples:
    modes.append("Sample images")
modes += ["Image URL", "Use camera"]

with st.sidebar:
    st.header("Settings")
    conf = st.slider("Confidence threshold", 0.05, 0.95, 0.25, 0.05,
                     help="Lower values detect more damage but may add false alarms.")
    iou = st.slider("NMS IoU threshold", 0.10, 0.95, 0.45, 0.05,
                    help="Controls how overlapping boxes are merged.")
    st.divider()
    input_mode = st.radio("Input source", modes)
    st.divider()
    st.markdown(
        "**About**  \n"
        "Fine-tuned YOLO26n trained on a 7-class road damage dataset "
        "(Roboflow). Built with Ultralytics, PyTorch, and Streamlit."
    )

try:
    model = load_model()
except Exception as exc:
    st.error(f"Could not load the model: {exc}")
    st.stop()

image = None

if input_mode == "Upload image":
    file = st.file_uploader("Upload a road image", type=["jpg", "jpeg", "png", "webp"])
    if file is not None:
        image = Image.open(file).convert("RGB")

elif input_mode == "Sample images":
    choice = st.selectbox("Choose a sample image", examples, format_func=lambda p: p.name)
    if choice is not None:
        image = Image.open(choice).convert("RGB")

elif input_mode == "Image URL":
    url = st.text_input("Paste a direct image link", placeholder="https://example.com/road.jpg")
    if url.strip():
        try:
            with st.spinner("Downloading image..."):
                image = load_image_from_url(url.strip())
        except Exception as exc:
            st.error(f"Could not load the image: {exc}")

else:
    shot = st.camera_input("Take a photo of the road")
    if shot is not None:
        image = Image.open(shot).convert("RGB")

if image is not None:
    with st.spinner("Detecting road damage..."):
        annotated, counts = run_detection(model, image, conf, iou)

    col1, col2 = st.columns(2)
    with col1:
        st.subheader("Original")
        st.image(image, use_container_width=True)
    with col2:
        st.subheader("Detection result")
        st.image(annotated, use_container_width=True)

    st.divider()

    if not counts:
        st.info("No road damage detected at the current confidence threshold. "
                "Try lowering the threshold from the sidebar, or use a clearer road image.")
    else:
        st.subheader(f"Detected damage ({sum(counts.values())} total)")
        st.table({"Class": list(counts.keys()), "Count": list(counts.values())})

    buffer = io.BytesIO()
    annotated.save(buffer, format="PNG")
    st.download_button("⬇️ Download result", data=buffer.getvalue(),
                       file_name="road_damage_result.png", mime="image/png")
else:
    st.info("Choose an input source from the sidebar to start.")
