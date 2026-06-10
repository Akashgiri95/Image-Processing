import cv2
import numpy as np
import streamlit as st

from backend import (
    load_models,
    detect_plates_yolo,
    detect_plates_contour,
    enhance_plate_yolo,
    enhance_plate_hierarchy,
    ocr_plate,
    quality_metrics,
)

# ----------------- PAGE CONFIG -----------------

st.set_page_config(
    page_title="Car Number Plate Detection & Enhancement",
    layout="wide",
    page_icon="🚗",
)

st.markdown(
    """
    <style>
        .main-title {
            font-size: 2.2rem;
            font-weight: 700;
            color: #1f2933;
        }
        .subtitle {
            font-size: 0.9rem;
            color: #6b7280;
        }
        .pipeline-title {
            font-size: 1.0rem;
            font-weight: 600;
            margin-top: 0.5rem;
        }
        .plate-header {
            padding: 0.4rem 0.8rem;
            border-radius: 0.4rem;
            background-color: #f3f4ff;
            color: #1f2933;
            font-weight: 600;
            margin-bottom: 0.4rem;
        }
        .metric-box {
            border-radius: 0.4rem;
            padding: 0.6rem 0.9rem;
            background-color: #f9fafb;
            border: 1px solid #e5e7eb;
        }
    </style>
    """,
    unsafe_allow_html=True,
)

st.markdown(
    '<div class="main-title">Car Number Plate Detection & Enhancement</div>',
    unsafe_allow_html=True,
)
st.markdown(
    '<div class="subtitle">Two pipelines: YOLOv8-based detection vs pure contour/hierarchical detection, both with enhancement, OCR, and quality analytics.</div>',
    unsafe_allow_html=True,
)
st.markdown("---")


# ----------------- SIDEBAR CONFIG PANEL -----------------

st.sidebar.header("YOLO pipeline config")
conf_thresh = st.sidebar.slider(
    "YOLO confidence threshold", 0.10, 0.90, 0.40, 0.05
)

st.sidebar.header("Contour pipeline config")
contour_canny_low = st.sidebar.slider(
    "Contour Canny low", 0, 150, 50, 5
)
contour_canny_high = st.sidebar.slider(
    "Contour Canny high", 50, 300, 150, 5
)
contour_close_ksize = st.sidebar.slider(
    "Contour closing kernel width", 1, 15, 5, 1
)
contour_close_iters = st.sidebar.slider(
    "Contour closing iterations", 1, 5, 2, 1
)

st.sidebar.header("Enhancement parameters (for both)")
enh_canny_low = st.sidebar.slider(
    "Enhancement Canny low", 0, 150, 30, 5
)
enh_canny_high = st.sidebar.slider(
    "Enhancement Canny high", 50, 300, 120, 5
)
enh_close_ksize = st.sidebar.slider(
    "Enhancement closing kernel width", 1, 15, 5, 1
)
enh_close_iters = st.sidebar.slider(
    "Enhancement closing iterations", 1, 5, 2, 1
)

st.sidebar.header("View options")
show_edges = st.sidebar.checkbox("Show Canny edges", value=False)
show_binary = st.sidebar.checkbox("Show binary view", value=True)

st.sidebar.markdown("---")
st.sidebar.markdown(
    "YOLO pipeline: uses trained detector.\n"
    "Contour pipeline: uses only Canny + closing + contour filtering on full image.\n"
    "Compare how both behave as you adjust parameters."
)


# ----------------- MODEL LOADING (CACHED) -----------------

@st.cache_resource
def load_all_models():
    return load_models()


yolo_model, ocr_reader = load_all_models()


# ----------------- FILE UPLOAD -----------------

uploaded = st.file_uploader(
    "Upload a car image (JPG / PNG)", type=["jpg", "jpeg", "png"]
)

if uploaded is None:
    st.info("Upload an image with one or more cars to start both pipelines.")
    st.stop()

file_bytes = np.asarray(bytearray(uploaded.read()), dtype=np.uint8)
img_bgr = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)

if img_bgr is None:
    st.error("Could not decode the uploaded image. Please try a different file.")
    st.stop()


# ----------------- STEP 1: INPUT IMAGE -----------------

st.subheader("Step 1 · Input image")
st.image(img_bgr, channels="BGR", use_column_width=True)
st.markdown("---")


# ----------------- STEP 2: YOLO vs CONTOUR DETECTION -----------------

st.subheader("Step 2 · Plate detection by two pipelines")

# YOLO detection (pipeline 1)
yolo_plates = detect_plates_yolo(img_bgr, yolo_model, conf_thresh=conf_thresh)

img_yolo = img_bgr.copy()
for i, p in enumerate(yolo_plates, start=1):
    x1, y1, x2, y2 = p["bbox"]
    conf = p["conf"]
    cv2.rectangle(img_yolo, (x1, y1), (x2, y2), (34, 197, 94), 3)
    cv2.putText(
        img_yolo,
        f"{i}:{conf:.2f}",
        (x1, max(0, y1 - 8)),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.6,
        (34, 197, 94),
        2,
        cv2.LINE_AA,
    )

# Contour detection (pipeline 2)
contour_plates = detect_plates_contour(
    img_bgr,
    canny_low=contour_canny_low,
    canny_high=contour_canny_high,
    close_ksize=contour_close_ksize,
    close_iters=contour_close_iters,
)

img_contour = img_bgr.copy()
for i, p in enumerate(contour_plates, start=1):
    x1, y1, x2, y2 = p["bbox"]
    score = p["score"]
    cv2.rectangle(img_contour, (x1, y1), (x2, y2), (255, 140, 0), 3)
    cv2.putText(
        img_contour,
        f"{i}:{score:.2f}",
        (x1, max(0, y1 - 8)),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.6,
        (255, 140, 0),
        2,
        cv2.LINE_AA,
    )

col_det1, col_det2 = st.columns(2)
with col_det1:
    st.markdown('<div class="pipeline-title">YOLO pipeline detection</div>', unsafe_allow_html=True)
    st.image(
        img_yolo,
        channels="BGR",
        caption=f"YOLO detected plates: {len(yolo_plates)}",
        use_column_width=True,
    )

with col_det2:
    st.markdown('<div class="pipeline-title">Contour/hierarchical pipeline detection</div>', unsafe_allow_html=True)
    st.image(
        img_contour,
        channels="BGR",
        caption=f"Contour-detected plates: {len(contour_plates)}",
        use_column_width=True,
    )

st.markdown("---")


# ----------------- STEP 3–5: PER‑PLATE ENHANCEMENT, OCR, METRICS -----------------

st.subheader("Step 3–5 · Per‑plate enhancement, OCR, and metrics for both pipelines")

max_pairs = max(len(yolo_plates), len(contour_plates))

if max_pairs == 0:
    st.warning("Neither YOLO nor contour pipeline detected any plate.")
    st.stop()

for idx in range(max_pairs):
    st.markdown(
        f'<div class="plate-header">Plate index {idx + 1}</div>',
        unsafe_allow_html=True,
    )

    col_y, col_c = st.columns(2)

    # ----- YOLO pipeline for this index -----
    with col_y:
        st.markdown("### YOLO pipeline output")

        if idx < len(yolo_plates):
            p = yolo_plates[idx]
            roi = p["crop"]
            st.image(
                roi,
                channels="BGR",
                caption="YOLO ROI",
                use_column_width=True,
            )

            gray_y, edges_y, bin_y, dbg_y = enhance_plate_yolo(
                roi,
                canny_low=enh_canny_low,
                canny_high=enh_canny_high,
                close_ksize=enh_close_ksize,
                close_iters=enh_close_iters,
            )

            vcols = st.columns(2)
            if show_edges:
                with vcols[0]:
                    st.image(
                        dbg_y["edges"],
                        caption="YOLO Canny edges",
                        clamp=True,
                        use_column_width=True,
                    )
            else:
                with vcols[0]:
                    st.image(
                        dbg_y["closed"],
                        caption="YOLO Canny + closing",
                        clamp=True,
                        use_column_width=True,
                    )

            if show_binary:
                with vcols[1]:
                    st.image(
                        dbg_y["binary"],
                        caption="YOLO enhanced binary",
                        clamp=True,
                        use_column_width=True,
                    )

            full_y, clean_y, cnt_y = ocr_plate(bin_y, ocr_reader)
            blur_y, edens_y, contrast_y = quality_metrics(gray_y, edges_y)

            st.markdown("**YOLO OCR**")
            st.markdown(f"- Text: `{full_y}`")
            st.markdown(f"- Clean: `{clean_y}` (length: {cnt_y})")

            st.markdown("**YOLO quality metrics**")
            st.markdown(
                f'<div class="metric-box">Blur: <strong>{blur_y:.2f}</strong></div>',
                unsafe_allow_html=True,
            )
            st.markdown(
                f'<div class="metric-box">Edge density: <strong>{edens_y:.4f}</strong></div>',
                unsafe_allow_html=True,
            )
            st.markdown(
                f'<div class="metric-box">Contrast: <strong>{contrast_y:.2f}</strong></div>',
                unsafe_allow_html=True,
            )
        else:
            st.info("No YOLO plate at this index.")

    # ----- CONTOUR pipeline for this index -----
    with col_c:
        st.markdown("### Contour/hierarchical pipeline output")

        if idx < len(contour_plates):
            p = contour_plates[idx]
            roi = p["crop"]
            st.image(
                roi,
                channels="BGR",
                caption="Contour-detected ROI",
                use_column_width=True,
            )

            gray_c, edges_c, bin_c, dbg_c = enhance_plate_hierarchy(
                roi,
                canny_low=enh_canny_low,
                canny_high=enh_canny_high,
                close_ksize=enh_close_ksize,
                close_iters=enh_close_iters,
            )

            vcols = st.columns(2)
            if show_edges:
                with vcols[0]:
                    st.image(
                        dbg_c["edges"],
                        caption="Contour Canny edges",
                        clamp=True,
                        use_column_width=True,
                    )
            else:
                with vcols[0]:
                    st.image(
                        dbg_c["closed"],
                        caption="Contour Canny + closing",
                        clamp=True,
                        use_column_width=True,
                    )

            if show_binary:
                with vcols[1]:
                    st.image(
                        dbg_c["binary"],
                        caption="Contour enhanced binary",
                        clamp=True,
                        use_column_width=True,
                    )

            full_c, clean_c, cnt_c = ocr_plate(bin_c, ocr_reader)
            blur_c, edens_c, contrast_c = quality_metrics(gray_c, edges_c)

            st.markdown("**Contour OCR**")
            st.markdown(f"- Text: `{full_c}`")
            st.markdown(f"- Clean: `{clean_c}` (length: {cnt_c})")

            st.markdown("**Contour quality metrics**")
            st.markdown(
                f'<div class="metric-box">Blur: <strong>{blur_c:.2f}</strong></div>',
                unsafe_allow_html=True,
            )
            st.markdown(
                f'<div class="metric-box">Edge density: <strong>{edens_c:.4f}</strong></div>',
                unsafe_allow_html=True,
            )
            st.markdown(
                f'<div class="metric-box">Contrast: <strong>{contrast_c:.2f}</strong></div>',
                unsafe_allow_html=True,
            )
        else:
            st.info("No contour-detected plate at this index.")

    st.markdown("---")
