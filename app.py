from pathlib import Path

import streamlit as st
from PIL import Image
from ultralytics import YOLO

st.set_page_config(page_title="Road Damage Detection", layout="wide")
st.title("Road Damage Detection System (YOLO26n)")
st.write("Upload a road image to detect potholes, cracks, and other road defects.")


@st.cache_resource
def load_model():
    return YOLO(str(Path("weights/best.pt")))


model = load_model()

conf = st.sidebar.slider("Confidence threshold", 0.05, 0.95, 0.25, 0.05)
iou = st.sidebar.slider("NMS IoU threshold", 0.10, 0.95, 0.45, 0.05)

file = st.file_uploader("Upload a road image", type=["jpg", "jpeg", "png"])

if file:
    image = Image.open(file).convert("RGB")
    result = model.predict(image, imgsz=640, conf=conf, iou=iou, device="cpu", verbose=False)[0]

    col1, col2 = st.columns(2)
    col1.image(image, caption="Original", use_container_width=True)
    col2.image(result.plot()[..., ::-1], caption="Detection result", use_container_width=True)

    if len(result.boxes) == 0:
        st.info("No road damage detected at the current threshold.")
    else:
        counts = {}
        for c in result.boxes.cls.int().tolist():
            counts[model.names[c]] = counts.get(model.names[c], 0) + 1
        st.subheader("Detected damage")
        st.table({"Class": list(counts.keys()), "Count": list(counts.values())})
