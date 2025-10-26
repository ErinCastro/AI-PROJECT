
import os
os.environ["YOLO_NO_CV2"] = "1"
os.environ["QT_QPA_PLATFORM"] = "offscreen"
os.environ["DISPLAY"] = ""
os.environ["OPENCV_VIDEOIO_PRIORITY_MSMF"] = "0"
os.environ["OPENCV_VIDEOIO_PRIORITY_GSTREAMER"] = "0"
os.environ["OPENCV_FFMPEG_CAPTURE_OPTIONS"] = "0"

import streamlit as st
from ultralytics import YOLO
import numpy as np
import tempfile, os
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
