# Image Processing

Hands-on image processing work — from pixel-level fundamentals to a production-grade ANPR system. Covers classical techniques using OpenCV, and a real-world applied project comparing deep learning vs classical computer vision.

---

## Contents

| File / Folder | Topics |
|---|---|
| [`Lab Assignment.ipynb`](Lab%20Assignment.ipynb) | Image I/O with OpenCV & Pillow · Color spaces (BGR, RGB, Grayscale, HSV) · Resize, crop, rotate, flip · Channel splitting |
| [`Assignment 2.ipynb`](Assignment%202.ipynb) | Manual histogram via NumPy (no `cv2.calcHist`) · Contrast stretching (min-max normalization) · PIL vs OpenCV workflow comparison |
| [`IPP_Assignmet_5&6.ipynb`](IPP_Assignmet_5%266.ipynb) | Thresholding (binary / inverse) · Contour detection · Object counting · Area statistics (mean, median, min, max) · Shape classification (triangle, rectangle, circle) via `approxPolyDP` · Pandas summary table |
| [`Lab_Assignment_7_8.ipynb`](Lab_Assignment_7_8.ipynb) | Advanced contour analysis · Convex hull · Convexity defect detection · Defective part identification with area thresholding |
| [`APNR YOLO+Traditional method/`](APNR%20YOLO%2BTraditional%20method) | **Project** — Dual-pipeline ANPR: YOLOv8 plate detection vs contour-based detection · EasyOCR text extraction · Streamlit comparison UI |

---

## Progression

```
Fundamentals         → Image I/O, color spaces, geometric transforms
Intensity Processing → Histograms, contrast enhancement, thresholding
Structural Analysis  → Contour detection, shape classification, defect analysis
Applied Project      → ANPR: deep learning pipeline vs classical CV pipeline
```

---

## Skills Demonstrated

- **Image fundamentals** — pixel manipulation, color channel math, affine transforms
- **Intensity processing** — histogram analysis, contrast stretching, binary thresholding
- **Morphological analysis** — contour detection, shape classification, convexity defect detection
- **Applied project** — real-world ANPR with two competing pipelines and Streamlit UI

---

## Tech Stack

Python · OpenCV · NumPy · Pillow · Matplotlib · Pandas · scikit-image · YOLOv8 · EasyOCR · Streamlit
