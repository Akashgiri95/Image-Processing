# 03 — Morphological Analysis

Object detection, shape classification, and defect analysis using thresholding and contours.

## `thresholding_and_contours.ipynb`

- Binary thresholding (`THRESH_BINARY` vs `THRESH_BINARY_INV`) and why object detection needs a binary image
- Contour extraction (`cv2.findContours`, `RETR_EXTERNAL`, `CHAIN_APPROX_SIMPLE`)
- Object counting and area statistics (mean, median, largest, smallest)
- Shape classification — triangle / rectangle / circle — via `cv2.approxPolyDP`
- Tabulating per-object results (shape, area, perimeter) with Pandas

## `contour_defect_detection.ipynb`

- Builds on contour detection for a quality-inspection use case
- Convex hull and convexity defect detection
- Flags defective parts using a defect-count + area threshold
- Reports total parts, defective parts, and defect percentage

## Tech Stack

Python · OpenCV · NumPy · Pandas · Matplotlib
