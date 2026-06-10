# 02 — Intensity Processing

Pixel-intensity techniques for analysing and enhancing image contrast.

## `histogram_and_contrast.ipynb`

- Manual histogram computation with NumPy (`np.histogram`, no `cv2.calcHist`)
- Histogram comparison: manual vs `cv2.calcHist` vs `plt.hist`
- Contrast stretching using min-max normalization:
  `stretched = (img - r_min) * (255 / (r_max - r_min))`
- Histogram equalization (`cv2.equalizeHist`) for low-light / uneven-lighting images

## Tech Stack

Python · OpenCV · NumPy · Matplotlib
