# Image Processing

Hands-on image processing work — from pixel-level fundamentals to a production-grade ANPR system. Covers classical techniques using OpenCV, and a real-world applied project comparing deep learning vs classical computer vision.

---

## Contents

| Folder | Topics |
|---|---|
| [`01_fundamentals/`](01_fundamentals) | Image I/O with OpenCV & Pillow · Color spaces (BGR, RGB, Grayscale, HSV) · Resize, crop, rotate, flip · Channel splitting |
| [`02_intensity_processing/`](02_intensity_processing) | Manual histogram via NumPy (no `cv2.calcHist`) · Contrast stretching (min-max normalization) · Histogram equalization |
| [`03_morphological_analysis/`](03_morphological_analysis) | Thresholding · Contour detection · Object counting & area statistics · Shape classification · Convexity defect detection |
| [`anpr/`](anpr) | **Project** — Dual-pipeline ANPR: YOLOv8 vs contour-based plate detection · EasyOCR · Streamlit UI · tutorial notebooks |

Each folder has its own README with a detailed breakdown.

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
