
import os
import time
import json
import warnings
import tempfile
from pathlib import Path

import streamlit as st
import cv2
import numpy as np
import pandas as pd
from PIL import Image
import textwrap
import traceback

# Ultralytics YOLOv11
try:
    from ultralytics import YOLO
except Exception as e:
    YOLO = None

#############################
# Helpers & Utilities       #
#############################

def fmt_err(prefix, e):
    """Return a robust multi-line error message."""
    return textwrap.dedent(f"""
        {prefix}
        Original error: {repr(e)}
        Traceback (most recent call last):
        {traceback.format_exc()}
    """)


def save_uploaded_file(uploaded_file, suffix=""):
    """Save an uploaded file to a temporary location and return its path."""
    tmp_dir = tempfile.mkdtemp()
    filename = uploaded_file.name
    path = os.path.join(tmp_dir, f"{Path(filename).stem}{suffix}{Path(filename).suffix}")
    with open(path, "wb") as f:
        f.write(uploaded_file.getbuffer())
    return path


def parse_selected_classes(selected_names, model_names):
    """Map selected class names to indices for Ultralytics 'classes' filter."""
    if not selected_names:
        return None
    # model_names is typically a dict {idx: name}
    if isinstance(model_names, dict):
        name_to_idx = {v: k for k, v in model_names.items()}
    else:
        name_to_idx = {v: i for i, v in enumerate(model_names)}
    indices = []
    for name in selected_names:
        if name in name_to_idx:
            indices.append(name_to_idx[name])
        else:
            st.warning(f"Class name '{name}' not found in model.names")
    return sorted(set(indices)) if indices else None


@st.cache_resource(show_spinner=True)
def load_yolov11_model(weights_path, device=None):
    """Load Ultralytics YOLO (v11) model from weights."""
    if YOLO is None:
        raise RuntimeError("Ultralytics package not found. Please install with: pip install ultralytics")
    try:
        model = YOLO(weights_path)
    except Exception as e:
        raise RuntimeError(fmt_err("Failed to load YOLOv11 model from given weights.", e))
    # device is handled at predict-time in Ultralytics
    return model


def run_inference_on_image(model, img_path, out_dir, imgsz, conf, iou, 
                           classes=None, agnostic_nms=False, device=None, out_ext="jpg"):
    """Run inference and save annotated IR image (jpg/png) + CSV of detections."""
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    try:
        results = model.predict(
            source=str(img_path),
            conf=float(conf),
            iou=float(iou),
            imgsz=int(imgsz),
            device=None if (device is None or device == 'auto') else device,
            classes=classes,  # list of class indices or None
            agnostic_nms=bool(agnostic_nms),
            verbose=False,
        )
    except Exception as e:
        raise RuntimeError(fmt_err(f"Prediction failed for image: {img_path}", e))

    # Expect one result per image
    if not results:
        warnings.warn("No results returned.")
        return None

    r = results[0]

    # Annotated image with bounding boxes
    try:
        im_annot = r.plot()  # numpy array (BGR)
        stem = Path(img_path).stem
        ext = ".png" if str(out_ext).lower() == "png" else ".jpg"
        out_img_path = out_dir / f"{stem}{ext}"
        cv2.imwrite(str(out_img_path), im_annot)
    except Exception as e:
        warnings.warn(f"Could not render/save annotated image: {e}")

    # Save CSV of detections
    try:
        boxes = r.boxes  # Boxes object
        xyxy = boxes.xyxy.cpu().numpy() if hasattr(boxes.xyxy, 'cpu') else boxes.xyxy
        confs = boxes.conf.cpu().numpy() if hasattr(boxes.conf, 'cpu') else boxes.conf
        clss = boxes.cls.cpu().numpy() if hasattr(boxes.cls, 'cpu') else boxes.cls

        # model.names may be dict
        names_map = getattr(model, 'names', None)
        if names_map is None:
            names_map = getattr(model.model, 'names', None)
        def cls_to_name(i):
            try:
                if isinstance(names_map, dict):
                    return names_map.get(int(i), str(int(i)))
                else:
                    return names_map[int(i)] if names_map is not None else str(int(i))
            except Exception:
                return str(int(i))

        rows = []
        for j in range(len(xyxy)):
            x1, y1, x2, y2 = xyxy[j]
            rows.append({
                'xmin': float(x1), 'ymin': float(y1), 'xmax': float(x2), 'ymax': float(y2),
                'confidence': float(confs[j]), 'class': int(clss[j]), 'name': cls_to_name(clss[j])
            })
        df = pd.DataFrame(rows)
        stem = Path(img_path).stem
        (out_dir / f"{stem}.csv").write_text(df.to_csv(index=False))
    except Exception as e:
        warnings.warn(f"Could not save detection CSV: {e}")

    return r


#############################
# Streamlit UI              #
#############################

st.set_page_config(page_title="IR Thermography Defect Detection (YOLOv11)", layout="wide")

st.title("IR Thermography Defect Detection using YOLOv11 (best.pt)")
st.write(
    "Upload IR thermal images and a trained Ultralytics YOLOv11 weights file (best.pt). Configure confidence, IOU, classes, and output format, "
    "then run detection. Annotated images (JPG/PNG with bounding boxes) and CSVs will be saved to a run folder."
)

with st.sidebar:
    st.header("Model & Settings")
    weights_upload = st.file_uploader("Upload YOLOv11 weights (.pt)", type=["pt"], accept_multiple_files=False)

    device_choice = st.selectbox("Device", options=["auto", "cpu", "cuda"], index=0)

    conf_thres = st.slider("Confidence threshold", 0.0, 1.0, 0.25, 0.01)
    iou_thres = st.slider("NMS IoU threshold", 0.0, 1.0, 0.45, 0.01)
    imgsz = st.number_input("Image size (pixels)", min_value=320, max_value=4096, value=1280, step=64)
    agnostic_nms = st.checkbox("Agnostic NMS", value=False)

    max_det = st.number_input("Max detections (predict-time)", min_value=1, max_value=10000, value=1000, step=10,
                              help="Ultralytics caps per-image detections internally; this control is informational.")

    out_root = st.text_input("Output root folder", value="./runs/ir_streamlit")
    out_ext = st.radio("Output image format", options=["jpg", "png"], index=0)

st.divider()

st.subheader("Upload IR images")
uploaded_images = st.file_uploader(
    "Upload one or more IR images (JPG/PNG/BMP/TIFF)", type=["jpg", "jpeg", "png", "bmp", "tif", "tiff"], accept_multiple_files=True
)

run_btn = st.button("Run Detection", type="primary")

model = None
weights_path = None

if weights_upload is not None:
    weights_path = save_uploaded_file(weights_upload)

if run_btn:
    if YOLO is None:
        st.error("Ultralytics package not found. Please install: pip install ultralytics")
        st.stop()
    if weights_path is None:
        st.error("Please upload a YOLOv11 weights file (best.pt) to proceed.")
        st.stop()
    if not uploaded_images:
        st.error("Please upload at least one IR image.")
        st.stop()

    # Load model
    try:
        with st.spinner("Loading YOLOv11 model..."):
            model = load_yolov11_model(weights_path, device=device_choice)
    except Exception as e:
        st.error(fmt_err("Loading YOLOv11 model failed.", e))
        st.stop()

    st.success("Model loaded.")

    # Class selection UI (after model is loaded so we can read model.names)
    names_map = getattr(model, 'names', None)
    if names_map is None:
        names_map = getattr(model.model, 'names', None)
    if isinstance(names_map, dict):
        cls_display = [names_map[i] for i in sorted(names_map.keys())]
    else:
        cls_display = list(names_map) if names_map is not None else []

    selected_cls_names = st.multiselect("Select classes to detect (optional)", options=cls_display)
    selected_cls_indices = parse_selected_classes(selected_cls_names, names_map)

    # Prepare output directory
    ts = time.strftime("%Y%m%d_%H%M%S")
    out_dir = Path(out_root) / f"run_{ts}"
    out_dir.mkdir(parents=True, exist_ok=True)

    # Save run config
    cfg = {
        "weights": weights_upload.name if weights_upload else None,
        "device": device_choice,
        "conf": conf_thres,
        "iou": iou_thres,
        "imgsz": imgsz,
        "agnostic_nms": agnostic_nms,
        "selected_classes": selected_cls_names,
        "out_dir": str(out_dir),
        "out_ext": out_ext,
    }
    (out_dir / "run_config.json").write_text(json.dumps(cfg, indent=2))

    # Layout for preview and tables
    cols = st.columns(2)
    left, right = cols

    results_summary = []
    for i, upl in enumerate(uploaded_images):
        with st.spinner(f"Processing image {i+1}/{len(uploaded_images)}: {upl.name}"):
            img_path = save_uploaded_file(upl, suffix="")
            try:
                res = run_inference_on_image(
                    model=model,
                    img_path=img_path,
                    out_dir=out_dir,
                    imgsz=int(imgsz),
                    conf=conf_thres,
                    iou=iou_thres,
                    classes=selected_cls_indices,
                    agnostic_nms=agnostic_nms,
                    device=device_choice,
                    out_ext=out_ext,
                )
            except Exception as e:
                st.error(fmt_err(f"Inference failed for {upl.name}.", e))
                continue

            # Display annotated image
            stem = Path(img_path).stem
            annotated_path = Path(out_dir) / f"{stem}.{out_ext}"
            if annotated_path.exists():
                try:
                    img = Image.open(annotated_path)
                    left.image(img, caption=f"Annotated: {upl.name}", use_column_width=True)
                except Exception:
                    im_bgr = cv2.imread(str(annotated_path))
                    if im_bgr is not None:
                        im_rgb = cv2.cvtColor(im_bgr, cv2.COLOR_BGR2RGB)
                        left.image(im_rgb, caption=f"Annotated: {upl.name}", use_column_width=True)
                    else:
                        st.warning("Annotated image could not be opened for preview.")
            else:
                st.warning("Annotated image not found; check output folder.")

            # Show table of detections
            try:
                if res is not None:
                    boxes = res.boxes
                    xyxy = boxes.xyxy.cpu().numpy() if hasattr(boxes.xyxy, 'cpu') else boxes.xyxy
                    confs = boxes.conf.cpu().numpy() if hasattr(boxes.conf, 'cpu') else boxes.conf
                    clss = boxes.cls.cpu().numpy() if hasattr(boxes.cls, 'cpu') else boxes.cls

                    def cls_to_name(i):
                        try:
                            if isinstance(names_map, dict):
                                return names_map.get(int(i), str(int(i)))
                            else:
                                return names_map[int(i)] if names_map is not None else str(int(i))
                        except Exception:
                            return str(int(i))

                    rows = []
                    for j in range(len(xyxy)):
                        x1, y1, x2, y2 = xyxy[j]
                        rows.append({
                            'xmin': float(x1), 'ymin': float(y1), 'xmax': float(x2), 'ymax': float(y2),
                            'confidence': float(confs[j]), 'class': int(clss[j]), 'name': cls_to_name(clss[j])
                        })
                    df = pd.DataFrame(rows)
                    right.dataframe(df, use_container_width=True)
                    results_summary.append({"image": upl.name, "detections": len(df)})
                else:
                    st.warning("No detections returned.")
            except Exception as e:
                st.warning(f"No detections table available: {e}")

    # Summary & download
    st.success(f"Completed. Outputs saved to: {out_dir}")

    if results_summary:
        st.subheader("Detection summary")
        st.table(pd.DataFrame(results_summary))

    # Zip download
    try:
        import shutil
        zip_path = str(out_dir) + ".zip"
        shutil.make_archive(str(out_dir), "zip", str(out_dir))
        with open(zip_path, "rb") as f:
            st.download_button(
                label="Download all outputs (ZIP)",
                data=f,
                file_name=os.path.basename(zip_path),
                mime="application/zip",
            )
    except Exception as e:
        st.warning(fmt_err("Could not create ZIP archive for outputs.", e))

