"""
Dental Cavity Detection — Streamlit app
=======================================
Upload a dental X-ray (or pick a sample) and a U-Net model segments candidate
cavity regions, overlays them on the image and reports a verdict.

Inference runs with ONNX Runtime (no TensorFlow needed at serving time), which
keeps the hosted app light and installs cleanly on any Python version.

Run locally:   streamlit run app.py
"""

from pathlib import Path

import cv2
import numpy as np
import streamlit as st
import onnxruntime as ort

# --------------------------------------------------------------------------- #
# Config
# --------------------------------------------------------------------------- #
IMG_SIZE = 128
ROOT = Path(__file__).parent
MODEL_PATH = ROOT / "unet_cavity_final.onnx"
SAMPLES_DIR = ROOT / "samples" / "images"

st.set_page_config(
    page_title="Dental Cavity Detection",
    page_icon="🦷",
    layout="wide",
)

# --------------------------------------------------------------------------- #
# Model loading (cached so the session is created once)
# --------------------------------------------------------------------------- #
@st.cache_resource(show_spinner="Loading the model…")
def load_session():
    sess = ort.InferenceSession(str(MODEL_PATH), providers=["CPUExecutionProvider"])
    return sess, sess.get_inputs()[0].name


def predict_mask(sess, input_name, gray: np.ndarray) -> np.ndarray:
    """Return a HxW probability map (same size as `gray`) in [0, 1]."""
    small = (cv2.resize(gray, (IMG_SIZE, IMG_SIZE)) / 255.0).astype(np.float32)
    x = small[np.newaxis, ..., np.newaxis]
    prob = sess.run(None, {input_name: x})[0][0, ..., 0]
    return cv2.resize(prob, (gray.shape[1], gray.shape[0]))


def build_overlay(gray, binary, min_area):
    """Red cavity overlay + a bounding box around the largest region."""
    rgb = cv2.cvtColor(gray, cv2.COLOR_GRAY2RGB)
    tint = rgb.copy()
    tint[binary > 0] = [255, 45, 45]
    overlay = cv2.addWeighted(rgb, 0.65, tint, 0.35, 0)

    detected, area = False, 0
    contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if contours:
        largest = max(contours, key=cv2.contourArea)
        area = cv2.contourArea(largest)
        if area >= min_area:
            x, y, w, h = cv2.boundingRect(largest)
            cv2.rectangle(overlay, (x, y), (x + w, y + h), (255, 215, 0), 2)
            detected = True
    return overlay, detected, area


def heatmap(prob: np.ndarray) -> np.ndarray:
    """Colourised probability map for display."""
    hm = cv2.applyColorMap((prob * 255).astype(np.uint8), cv2.COLORMAP_MAGMA)
    return cv2.cvtColor(hm, cv2.COLOR_BGR2RGB)


# --------------------------------------------------------------------------- #
# Sidebar
# --------------------------------------------------------------------------- #
with st.sidebar:
    st.header("⚙️ Settings")
    threshold = st.slider(
        "Probability threshold", 0.05, 0.95, 0.50, 0.05,
        help="Pixels above this probability are counted as cavity.",
    )
    min_area = st.slider(
        "Min region area (px)", 0, 500, 100, 10,
        help="Ignore detected blobs smaller than this to suppress noise.",
    )
    st.divider()
    st.markdown(
        "**About**\n\n"
        "A U-Net trained on dental X-ray patches to segment cavity regions. "
        "It is a **screening / second-opinion aid**, not a diagnostic tool."
    )
    st.caption("Model: U-Net · 128×128 grayscale · focal + dice loss · ONNX Runtime")

# --------------------------------------------------------------------------- #
# Header
# --------------------------------------------------------------------------- #
st.title("🦷 Dental Cavity Detection")
st.markdown(
    "Upload a dental X-ray image (or choose a sample) to segment and highlight "
    "candidate cavity regions using a U-Net semantic-segmentation model."
)

# --------------------------------------------------------------------------- #
# Input selection
# --------------------------------------------------------------------------- #
col_up, col_sample = st.columns([2, 1])
with col_up:
    uploaded = st.file_uploader(
        "Upload an X-ray", type=["png", "jpg", "jpeg", "bmp"]
    )
with col_sample:
    sample_files = sorted(p.name for p in SAMPLES_DIR.glob("*.png")) if SAMPLES_DIR.exists() else []
    sample_choice = st.selectbox(
        "…or try a sample", ["—"] + sample_files,
        help="Sample patches shipped with the repo.",
    )

# Resolve the chosen image into a BGR numpy array
image_bgr = None
if uploaded is not None:
    data = np.frombuffer(uploaded.read(), np.uint8)
    image_bgr = cv2.imdecode(data, cv2.IMREAD_COLOR)
elif sample_choice != "—":
    image_bgr = cv2.imread(str(SAMPLES_DIR / sample_choice), cv2.IMREAD_COLOR)

# --------------------------------------------------------------------------- #
# Run inference
# --------------------------------------------------------------------------- #
if image_bgr is None:
    st.info("👆 Upload an image or pick a sample to run detection.")
    st.stop()

if not MODEL_PATH.exists():
    st.error(f"Model file not found at `{MODEL_PATH.name}`. Make sure it sits next to app.py.")
    st.stop()

sess, input_name = load_session()
gray = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2GRAY)

prob = predict_mask(sess, input_name, gray)
binary = (prob > threshold).astype(np.uint8)
overlay, detected, area = build_overlay(gray, binary, min_area)

# --------------------------------------------------------------------------- #
# Verdict
# --------------------------------------------------------------------------- #
if detected:
    st.error(f"### 🔴 Cavity region detected  ·  largest area ≈ {int(area)} px")
else:
    st.success("### 🟢 No cavity region above the current threshold")

# --------------------------------------------------------------------------- #
# Visual results
# --------------------------------------------------------------------------- #
c1, c2, c3, c4 = st.columns(4)
c1.image(cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB), caption="Original", use_container_width=True)
c2.image(heatmap(prob), caption="Predicted probability", use_container_width=True)
c3.image((binary * 255), caption="Binary mask", use_container_width=True, clamp=True)
c4.image(overlay, caption="Detection overlay", use_container_width=True)

m1, m2, m3 = st.columns(3)
m1.metric("Max probability", f"{prob.max():.2f}")
m2.metric("Cavity pixels", f"{(100 * binary.mean()):.1f}%")
m3.metric("Largest region", f"{int(area)} px")

st.caption(
    "⚠️ This tool is for research and educational purposes only. It is **not** a "
    "certified medical device and must not be used for clinical diagnosis."
)
