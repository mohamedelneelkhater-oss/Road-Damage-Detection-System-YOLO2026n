# Road Damage Detection System

An end-to-end computer vision project that detects road damage (potholes, cracks, and other surface defects) in images using modern YOLO architectures, with an interactive Gradio web app for real-time inference.

Three YOLO models were trained, fine-tuned, and benchmarked on the same dataset. The best-performing one, **YOLO26n**, powers the demo app.

## Highlights

- Trained and fine-tuned **YOLO11s**, **YOLOv12s**, and **YOLO26n** on a 7-class road damage dataset
- Two-stage training: 50 epochs of full training, then 5 epochs of fine-tuning with a lower learning rate (`lr0=0.001`) and a frozen backbone (`freeze=10`)
- Side-by-side evaluation on precision, recall, mAP@50, mAP@50-95, and inference speed
- Deployed as a **Gradio** app with adjustable confidence and IoU thresholds

## Results

Evaluated on the validation split (1,146 images), image size 640, NVIDIA Tesla T4.

| Model     | Params | mAP@50 | mAP@50-95 | FPS* |
|-----------|-------:|-------:|----------:|-----:|
| **YOLO26n** | 2.4M | **0.4379** | **0.2551** | 256.7 |
| YOLOv12s  | 9.2M   | 0.4254 | 0.2465    | 132.1 |
| YOLO11s   | 9.4M   | 0.3135 | 0.1724    | 179.7 |

\*FPS is derived from Ultralytics' validation timing (preprocess + inference + postprocess) and varies between runs depending on caching and hardware load. Treat it as a relative comparison, not an absolute benchmark.

**Key takeaway:** the smallest model (YOLO26n, about 4x fewer parameters than the "s" variants) achieved the best accuracy and the best speed, which makes it a strong candidate for edge and real-time deployment.

## Dataset

- Source: Roboflow Universe (`road-damage`, version 1), exported in YOLO format
- 7 classes
- Please review the dataset's license on Roboflow before any commercial use

## Project Structure

```
road-damage-detection/
├── app.py                # Gradio web application
├── requirements.txt      # Python dependencies
├── weights/
│   └── best.pt           # Trained YOLO26n weights (add your own)
├── notebook/
│   └── road-damage-detection-system.ipynb
└── README.md
```

## Getting Started

```bash
# 1. Clone the repository
git clone https://github.com/<your-username>/road-damage-detection.git
cd road-damage-detection

# 2. Install dependencies
pip install -r requirements.txt

# 3. Make sure weights/best.pt exists, then launch the app
python app.py
```

Open the local URL printed in the terminal. The app uses the GPU automatically when CUDA is available and falls back to the CPU otherwise.

## How the App Works (`app.py`)

1. Loads the fine-tuned YOLO26n weights once at startup
2. Selects the device (GPU with FP16, or CPU)
3. Accepts an uploaded image plus confidence and NMS IoU thresholds
4. Runs inference at 640 px and draws bounding boxes and labels
5. Returns the annotated image and a per-class count of detections

## Training Pipeline

```python
from ultralytics import YOLO

# Stage 1: full training
model = YOLO("yolo26n.pt")
model.train(data="data.yaml", epochs=50, imgsz=640)

# Stage 2: fine-tuning with a frozen backbone
model = YOLO("runs/detect/train/weights/best.pt")
model.train(data="data.yaml", epochs=5, imgsz=640, lr0=0.001, freeze=10, close_mosaic=0)
```

## Limitations and Future Work

- Absolute scores (mAP@50 around 0.44) leave clear room for improvement; some classes are likely under-represented
- Next steps: class-balance analysis, stronger augmentation, larger or higher-resolution models, longer training, and video or dashcam inference
- Export to ONNX or TensorRT for faster edge deployment

## Tech Stack

Python, Ultralytics YOLO, PyTorch, Gradio, OpenCV, Pandas, Roboflow, Kaggle (Tesla T4)

## Author

**<Your Name>**: [LinkedIn](https://linkedin.com/in/<your-profile>) | [GitHub](https://github.com/<your-username>)

If you found this project useful, consider giving it a star.
