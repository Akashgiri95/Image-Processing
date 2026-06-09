# ANPR — Automatic Number Plate Recognition

A Streamlit app that runs **two competing detection pipelines** on the same image and displays results side by side — built to compare deep learning vs classical computer vision for plate detection.

---

## Pipelines

| | Pipeline 1 — Deep Learning | Pipeline 2 — Classical CV |
|---|---|---|
| **Detection** | YOLOv8 (fine-tuned `plate_yolov8.pt`) | Canny edges → morphological close → contour hierarchy |
| **OCR** | EasyOCR | EasyOCR |
| **Strength** | Handles varied angles, lighting, partial occlusion | Fast, no GPU required, interpretable steps |
| **Output** | Bounding box, confidence, cropped plate, OCR text | Bounding box, contour score, cropped plate, OCR text |

---

## Architecture

```
Upload image (Streamlit UI)
       │
       ├── Pipeline 1: YOLO detect → crop → enhance → EasyOCR → text + confidence
       └── Pipeline 2: Canny → close → contour filter → crop → enhance → EasyOCR → text + score

Both results displayed side by side with quality metrics
```

---

## Run

```bash
pip install ultralytics easyocr streamlit opencv-python imutils
streamlit run app.py
```

> `weights/plate_yolov8.pt` must be present in the same directory.

---

## Tech Stack

Python · YOLOv8 (Ultralytics) · OpenCV · EasyOCR · Streamlit · imutils · NumPy
