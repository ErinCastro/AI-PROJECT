import os
import sys
import types

# ---- Disable OpenCV usage & GUI backends BEFORE importing ultralytics ----
os.environ["YOLO_NO_CV2"] = "1"
os.environ["QT_QPA_PLATFORM"] = "offscreen"  # Disable any Qt display
os.environ["DISPLAY"] = ""  # No X server
os.environ["OPENCV_VIDEOIO_PRIORITY_MSMF"] = "0"
os.environ["OPENCV_VIDEOIO_PRIORITY_GSTREAMER"] = "0"
os.environ["OPENCV_FFMPEG_CAPTURE_OPTIONS"] = "0"

# ---- Fake cv2 to avoid errors when it's imported inside ultralytics ----
if "cv2" not in sys.modules:
    cv2_stub = types.ModuleType("cv2")
    # Add no-op functions for the ones that Ultralytics might call
    cv2_stub.imshow = lambda *args, **kwargs: None
    cv2_stub.imwrite = lambda *args, **kwargs: None
    cv2_stub.imread = lambda *args, **kwargs: None
    cv2_stub.IMREAD_COLOR = 1  # Fake the IMREAD_COLOR constant
    cv2_stub.setNumThreads = lambda *args, **kwargs: None  # Fake the setNumThreads function
    cv2_stub.__version__ = "0.0.0-stub"  # Fake version
    sys.modules["cv2"] = cv2_stub

# ---- Completely intercept any OpenCV imports and make them "no-op" ----
def no_cv2_import(*args, **kwargs):
    raise ImportError("cv2 is not available, OpenCV functions are disabled.")

sys.modules['cv2'] = types.ModuleType('cv2')
sys.modules['cv2'].imshow = no_cv2_import
sys.modules['cv2'].imwrite = no_cv2_import
sys.modules['cv2'].imread = no_cv2_import
sys.modules['cv2'].setNumThreads = no_cv2_import
sys.modules['cv2'].IMREAD_COLOR = 1

# ---- Import the rest of the packages after the patches ----
import streamlit as st
from ultralytics import YOLO
import numpy as np
import tempfile
from PIL import Image

st.set_page_config(page_title="YOLO Deploy", page_icon="🤖", layout="centered")
st.title("YOLO Object Detection (Streamlit + Colab) 🤖")
st.write("Upload an image and run detection using your YOLO weights.")

WEIGHTS_PATH = st.text_input("Path to your weights (.pt):", "/content/drive/MyDrive/FINALMORINGA/yolov12/runs/segment/train3/weights/best.pt")

# Lazy-load model when needed
@st.cache_resource
def load_model(weights):
    return YOLO(weights)

uploaded = st.file_uploader("Choose an image", type=["jpg","jpeg","png"])
conf = st.slider("Confidence threshold", 0.1, 0.9, 0.25, 0.05)

if uploaded:
    img = Image.open(uploaded).convert("RGB")
    st.image(img, caption="Uploaded", use_column_width=True)

    if st.button("Run detection"):
        model = load_model(WEIGHTS_PATH)
        with tempfile.NamedTemporaryFile(delete=False, suffix=".png") as tmp:
            img.save(tmp.name)
            results = model.predict(source=tmp.name, conf=conf, save=False, verbose=False)
        # results[0].plot() returns a numpy array with annotations
        annotated = results[0].plot()
        st.image(annotated, caption="Detections", use_column_width=True)
        # Show raw boxes/classes (optional)
        with st.expander("Show raw results"):
            st.write(results[0].boxes)
