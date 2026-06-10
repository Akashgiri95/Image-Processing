import cv2
import numpy as np
import imutils
from ultralytics import YOLO
import easyocr


# ---------- 1. MODEL LOADING ----------

def load_models(
    yolo_weights_path: str = "weights/plate_yolov8.pt",
    ocr_languages: list = None,
):
    if ocr_languages is None:
        ocr_languages = ["en"]

    yolo_model = YOLO(yolo_weights_path)
    ocr_reader = easyocr.Reader(ocr_languages)
    return yolo_model, ocr_reader


# ---------- 2. YOLO PLATE DETECTION (PIPELINE 1) ----------

def detect_plates_yolo(img_bgr, yolo_model, conf_thresh=0.4):
    """
    YOLOv8 detector on full image.

    Returns list of dicts:
      {"bbox": (x1, y1, x2, y2), "conf": conf, "crop": roi_bgr}
    """
    img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
    results = yolo_model.predict(
        source=img_rgb,
        conf=conf_thresh,
        verbose=False
    )[0]

    plates = []
    h, w = img_bgr.shape[:2]

    for box in results.boxes:
        conf = float(box.conf[0])
        x1, y1, x2, y2 = box.xyxy[0].cpu().numpy().astype(int)

        x1 = max(0, min(x1, w - 1))
        x2 = max(0, min(x2, w - 1))
        y1 = max(0, min(y1, h - 1))
        y2 = max(0, min(y2, h - 1))
        if x2 <= x1 or y2 <= y1:
            continue

        roi = img_bgr[y1:y2, x1:x2].copy()
        plates.append({"bbox": (x1, y1, x2, y2), "conf": conf, "crop": roi})

    return plates


# ---------- 3. CONTOUR / HIERARCHY PLATE DETECTION (PIPELINE 2) ----------

def detect_plates_contour(
    img_bgr,
    canny_low=50,
    canny_high=150,
    close_ksize=5,
    close_iters=2,
):
    """
    Pure OpenCV contour-based detection on full image (no YOLO).

    Returns list of dicts:
      {"bbox": (x1, y1, x2, y2), "score": score, "crop": roi_bgr}
    """
    # resize for speed + scale back later
    img_resized = imutils.resize(img_bgr, width=600)
    scale_x = img_bgr.shape[1] / img_resized.shape[1]
    scale_y = img_bgr.shape[0] / img_resized.shape[0]

    gray = cv2.cvtColor(img_resized, cv2.COLOR_BGR2GRAY)

    # --- improved pre-processing ---
    # denoise a bit; Bilateral is enough here
    gray_blur = cv2.bilateralFilter(gray, 9, 75, 75)


    # global + Otsu binary threshold to separate plate/background
    _, thresh = cv2.threshold(
        gray_blur, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU
    )

    # Canny edges on thresholded image (clearer boundaries)[web:80]
    edges = cv2.Canny(thresh, canny_low, canny_high)

    # morphological closing with rectangular kernel tuned for plate aspect
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (close_ksize, 3))
    closed = cv2.morphologyEx(edges, cv2.MORPH_CLOSE, kernel, iterations=close_iters)

    # optional dilation to connect broken edges
    closed = cv2.dilate(closed, None, iterations=1)

    cnts, hierarchy = cv2.findContours(
        closed.copy(), cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE
    )

    H, W = gray.shape
    img_area = H * W
    plates = []

    if hierarchy is not None:
        hierarchy = hierarchy[0]

    for i, c in enumerate(cnts):
        x, y, w, h = cv2.boundingRect(c)
        area = w * h

        # stricter area range: avoid tiny noise and huge blobs[web:75]
        if area < 0.01 * img_area or area > 0.25 * img_area:
            continue

        aspect = w / float(h)
        # typical LP aspect (adjustable for your region)
        if not (2.0 <= aspect <= 6.5):
            continue

        # approximate polygon, require near-rectangular contour[web:64]
        peri = cv2.arcLength(c, True)
        approx = cv2.approxPolyDP(c, 0.02 * peri, True)
        if len(approx) < 4:
            continue

        # child check (contours inside -> characters)
        has_child = hierarchy[i][2] != -1 if hierarchy is not None else False

        roi_gray = gray[y:y + h, x:x + w]
        if roi_gray.size == 0:
            continue
        bright = roi_gray.mean() / 255.0

        # score combines size, brightness, and hierarchy
        score = area / img_area + 0.5 * bright + (0.3 if has_child else 0.0)

        # rescale coordinates to original image
        x1 = int(x * scale_x)
        y1 = int(y * scale_y)
        x2 = int((x + w) * scale_x)
        y2 = int((y + h) * scale_y)

        x1 = max(0, min(x1, img_bgr.shape[1] - 1))
        x2 = max(0, min(x2, img_bgr.shape[1] - 1))
        y1 = max(0, min(y1, img_bgr.shape[0] - 1))
        y2 = max(0, min(y2, img_bgr.shape[0] - 1))
        if x2 <= x1 or y2 <= y1:
            continue

        roi = img_bgr[y1:y2, x1:x2].copy()
        plates.append({"bbox": (x1, y1, x2, y2), "score": score, "crop": roi})

    # sort by score descending
    plates = sorted(plates, key=lambda d: d["score"], reverse=True)
    return plates


# ---------- 4. COMMON LIGHTING-ROBUST ENHANCEMENT ----------

def _lighting_robust_gray(roi_bgr, target_height=80):
    roi_bgr = imutils.resize(roi_bgr, height=target_height)
    gray = cv2.cvtColor(roi_bgr, cv2.COLOR_BGR2GRAY)

    mean_intensity = gray.mean()
    if mean_intensity < 80:
        gamma = 0.6
    elif mean_intensity > 180:
        gamma = 1.6
    else:
        gamma = 1.0

    if gamma != 1.0:
        g = gray.astype(np.float32) / 255.0
        g = np.power(g, gamma)
        gray = np.clip(g * 255.0, 0, 255).astype(np.uint8)

    gray_blur = cv2.bilateralFilter(gray, 9, 75, 75)

    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    gray_contrast = clahe.apply(gray_blur)

    return gray_contrast


# ---------- 5. YOLO PIPELINE ENHANCEMENT ----------

def enhance_plate_yolo(
    roi_bgr,
    canny_low=30,
    canny_high=120,
    close_ksize=5,
    close_iters=2,
):
    gray_contrast = _lighting_robust_gray(roi_bgr)

    edges = cv2.Canny(gray_contrast, canny_low, canny_high)

    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (close_ksize, 3))
    closed = cv2.morphologyEx(edges, cv2.MORPH_CLOSE, kernel, iterations=close_iters)

    _, binary = cv2.threshold(
        gray_contrast, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU
    )

    binary_closed = cv2.bitwise_or(binary, closed)

    debug = {"edges": edges, "closed": closed, "binary": binary_closed}
    return gray_contrast, edges, binary_closed, debug


# ---------- 6. HIERARCHY PIPELINE ENHANCEMENT ----------

def enhance_plate_hierarchy(
    roi_bgr,
    canny_low=30,
    canny_high=120,
    close_ksize=5,
    close_iters=2,
):
    """
    Enhancement for the contour-detected plate ROI.
    Similar to YOLO enhancement, but kept separate if later you
    want different defaults.
    """
    gray_contrast = _lighting_robust_gray(roi_bgr)

    edges = cv2.Canny(gray_contrast, canny_low, canny_high)

    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (close_ksize, 3))
    closed = cv2.morphologyEx(edges, cv2.MORPH_CLOSE, kernel, iterations=close_iters)

    _, binary = cv2.threshold(
        gray_contrast, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU
    )
    binary_closed = cv2.bitwise_or(binary, closed)

    debug = {"edges": edges, "closed": closed, "binary": binary_closed}
    return gray_contrast, edges, binary_closed, debug


# ---------- 7. OCR & QUALITY METRICS ----------

def ocr_plate(img_gray_or_bin, ocr_reader):
    if len(img_gray_or_bin.shape) == 2:
        img_for_ocr = cv2.cvtColor(img_gray_or_bin, cv2.COLOR_GRAY2BGR)
    else:
        img_for_ocr = img_gray_or_bin

    result = ocr_reader.readtext(img_for_ocr)
    full_text = "".join(r[1] for r in result)
    clean_text = "".join(ch for ch in full_text if ch.isalnum())
    return full_text, clean_text, len(clean_text)


def variance_of_laplacian(img_gray):
    return cv2.Laplacian(img_gray, cv2.CV_64F).var()


def edge_density(edges):
    return float(np.sum(edges > 0)) / float(edges.size)


def quality_metrics(gray_enhanced, edges):
    blur = variance_of_laplacian(gray_enhanced)
    edens = edge_density(edges)
    contrast = gray_enhanced.std()
    return blur, edens, contrast
