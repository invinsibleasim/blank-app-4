
import os
import io
import time
import json
import cv2
import math
import zipfile
import numpy as np
import streamlit as st
from PIL import Image
from pathlib import Path
from typing import Dict, Any, List, Tuple

# ---------------------------
# Ultralytics YOLO (YOLOv11)
# ---------------------------
ULTRA_AVAILABLE = False
try:
    from ultralytics import YOLO
    ULTRA_AVAILABLE = True
except Exception as e:
    ULTRA_AVAILABLE = False
    st.error(f"Ultralytics import failed. Install ultralytics + CPU torch in requirements.txt.\nError: {e}")
    st.stop()

# ---------------------------
# Utility helpers
# ---------------------------
def ensure_dir(path: Path):
    path.mkdir(parents=True, exist_ok=True)

def pil_to_numpy_rgb(img: Image.Image) -> np.ndarray:
    """Convert PIL -> numpy RGB (uint8)."""
    return np.array(img.convert("RGB"))

def to_three_channel_if_gray(arr: np.ndarray) -> np.ndarray:
    """Make 3-channel RGB from single-channel IR (repeat grayscale)."""
    if arr.ndim == 2:  # H, W
        return np.stack([arr, arr, arr], axis=-1)
    if arr.ndim == 3 and arr.shape[2] == 1:
        return np.repeat(arr, 3, axis=2)
    return arr  # already RGB/BGR-like

def draw_boxes(img_bgr: np.ndarray,
               boxes: List[List[int]],
               classes: List[int],
               scores: List[float],
               class_names: List[str]) -> np.ndarray:
    """Overlay bounding boxes + labels."""
    vis = img_bgr.copy()
    for (x1, y1, x2, y2), cls, score in zip(boxes, classes, scores):
        color = (0, 255, 0)
        cv2.rectangle(vis, (x1, y1), (x2, y2), color, 2)
        name = class_names[cls] if cls < len(class_names) else str(cls)
        label = f"{name} {score:.2f}"
        cv2.putText(vis, label, (x1, max(0, y1 - 6)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2, cv2.LINE_AA)
    return vis

def parse_yolo_results(result) -> Dict[str, Any]:
    """
    Parse a single ultralytics result (boxes only).
    Returns dict with boxes (xyxy int list), classes, scores.
    """
    out = {"boxes": [], "classes": [], "scores": []}
    if result is None or result.boxes is None:
        return out
    # xyxy in tensor -> numpy
    b = result.boxes
    xyxy = b.xyxy.cpu().numpy().astype(int)  # shape [N,4]
    cls = b.cls.cpu().numpy().astype(int).tolist()
    conf = b.conf.cpu().numpy().astype(float).tolist()
    out["boxes"] = xyxy.tolist()
    out["classes"] = cls
    out["scores"] = conf
    return out

def zip_in_memory(file_tuples: List[Tuple[str, bytes]]) -> io.BytesIO:
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w", zipfile.ZIP_DEFLATED) as zf:
        for zpath, data in file_tuples:
            zf.writestr(zpath, data)
    buf.seek(0)
    return buf

# ---------------------------
# Streamlit UI
# ---------------------------
st.set_page_config(page_title="IR Thermography Defect Detection (YOLOv11)", layout="wide")
st.title("🌡️ IR Thermography Defect Detection — YOLOv11 (Streamlit)")

st.markdown("""
Upload **IR thermography images/videos** and a **YOLOv11 `.pt` model** (`best.pt`) to detect defects.
The app shows overlays, counts per class, and saves outputs.
""")

# Sidebar: model upload & settings
st.sidebar.header("🤖 Model")
model_file = st.sidebar.file_uploader("Upload pretrained YOLOv11 model (.pt)", type=["pt"])
conf_thres = st.sidebar.slider("Confidence (conf)", 0.05, 0.95, 0.25, 0.05)
iou_thres  = st.sidebar.slider("IoU (NMS)", 0.10, 0.95, 0.45, 0.05)
imgsz      = st.sidebar.selectbox("Inference size (imgsz)", [640, 512, 416], index=0)
device     = st.sidebar.selectbox("Device", ["cpu"], index=0)  # keep CPU for Cloud

st.sidebar.header("🖼️ Image & 🎥 Video")
ir_is_grayscale = st.sidebar.checkbox("Inputs may be grayscale IR", True)
save_dir_str    = st.sidebar.text_input("Output directory", "output")
start_btn       = st.sidebar.button("🚀 Run detection")

# Uploaders in main area
st.subheader("Upload Images")
img_files = st.file_uploader("Select IR images", type=["jpg", "jpeg", "png", "bmp", "tif", "tiff"], accept_multiple_files=True)

st.subheader("Upload Videos")
vid_files = st.file_uploader("Select IR videos", type=["mp4", "avi", "mov", "mkv"], accept_multiple_files=True)

# ---------------------------
# Run
# ---------------------------
if start_btn:
    # Load model
    if model_file is None:
        st.error("Please upload a YOLOv11 `.pt` model first.")
        st.stop()

    # Persist model to disk
    models_dir = Path("models"); ensure_dir(models_dir)
    model_path = models_dir / model_file.name
    with open(model_path, "wb") as f:
        f.write(model_file.read())

    try:
        model = YOLO(str(model_path))
    except Exception as e:
        st.error(f"Failed to load YOLOv11 model: {e}")
        st.stop()

    # Show model classes
    try:
        model_names = model.names  # dict {id: name}
        class_names = [model_names[i] for i in sorted(model_names.keys())]
    except Exception:
        # fallback if names missing
        num_classes = int(st.number_input("Enter number of model classes", min_value=1, value=1))
        class_names = [f"class_{i}" for i in range(num_classes)]

    st.success(f"Model loaded: **{model_path.name}**")
    st.info(f"Number of classes in model: **{len(class_names)}** — {class_names}")

    # Output dir
    out_base = Path(save_dir_str); ensure_dir(out_base)

    # -----------------------
    # Process Images
    # -----------------------
    if img_files:
        st.subheader("Image results")
        img_zip_tuples = []  # (zip_path, bytes)
        agg_counts_img: Dict[str, int] = {}
        for upl in img_files:
            # Load image
            img = Image.open(io.BytesIO(upl.read()))
            rgb = pil_to_numpy_rgb(img)
            if ir_is_grayscale:
                # If original was single-channel, ensure 3-channel
                if rgb.ndim == 2 or (rgb.ndim == 3 and rgb.shape[2] == 1):
                    rgb = to_three_channel_if_gray(rgb)

            # Ultralytics expects RGB array
            # Run inference
            results = model.predict(
                source=rgb,
                imgsz=int(imgsz),
                conf=conf_thres,
                iou=iou_thres,
                device=device,
                verbose=False
            )

            r = results[0]
            parsed = parse_yolo_results(r)

            # Overlay on BGR for drawing then convert back
            bgr = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
            vis_bgr = draw_boxes(bgr, parsed["boxes"], parsed["classes"], parsed["scores"], class_names)
            vis_rgb = cv2.cvtColor(vis_bgr, cv2.COLOR_BGR2RGB)

            # Save to disk
            save_dir = out_base / Path(upl.name).stem
            ensure_dir(save_dir)
            out_img_path = save_dir / "overlay.jpg"
            Image.fromarray(vis_rgb).save(out_img_path, format="JPEG", quality=95)

            # Aggregate counts
            for cid in parsed["classes"]:
                cname = class_names[cid] if cid < len(class_names) else str(cid)
                agg_counts_img[cname] = agg_counts_img.get(cname, 0) + 1

            # Show in UI
            st.image(vis_rgb, caption=f"{upl.name} — detections: {len(parsed['boxes'])}", use_column_width=True)
            st.json({
                "file": upl.name,
                "num_boxes": len(parsed["boxes"]),
                "classes": [class_names[c] if c < len(class_names) else str(c) for c in parsed["classes"]],
                "scores": parsed["scores"]
            })

            # Add to ZIP (overlay + a JSON summary)
            with open(out_img_path, "rb") as f:
                img_zip_tuples.append((f"{Path(upl.name).stem}/overlay.jpg", f.read()))
            summary_json = json.dumps({
                "file": upl.name,
                "num_boxes": len(parsed["boxes"]),
                "classes": [class_names[c] if c < len(class_names) else str(c) for c in parsed["classes"]],
                "scores": parsed["scores"]
            }, indent=2).encode("utf-8")
            img_zip_tuples.append((f"{Path(upl.name).stem}/summary.json", summary_json))

        # Download ZIP with all image overlays
        if img_zip_tuples:
            img_zip = zip_in_memory(img_zip_tuples)
            st.download_button("📦 Download all image overlays (ZIP)", data=img_zip, file_name="images_results.zip")

        # Show aggregated counts
        if agg_counts_img:
            st.info("Aggregated image detections per class:")
            st.json(agg_counts_img)

    # -----------------------
    # Process Videos
    # -----------------------
    if vid_files:
        st.subheader("Video results")
        frame_stride = st.slider("Process every Nth frame (stride)", 1, 10, 2, 1)
        max_frames   = st.slider("Max frames to process (per video)", 50, 2000, 300, 50)

        for upl in vid_files:
            # Persist video to disk for OpenCV
            vid_tmp = Path("tmp_videos"); ensure_dir(vid_tmp)
            vid_path = vid_tmp / upl.name
            with open(vid_path, "wb") as f:
                f.write(upl.read())

            cap = cv2.VideoCapture(str(vid_path))
            if not cap.isOpened():
                st.error(f"Cannot open video: {upl.name}")
                continue

            fps = cap.get(cv2.CAP_PROP_FPS) or 25
            width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

            # Prepare writer
            out_dir = out_base / Path(upl.name).stem
            ensure_dir(out_dir)
            out_video_path = out_dir / "overlay.mp4"
            fourcc = cv2.VideoWriter_fourcc(*"mp4v")
            writer = cv2.VideoWriter(str(out_video_path), fourcc, fps, (width, height))

            # Aggregated counts for this video
            agg_counts_vid: Dict[str, int] = {}
            processed = 0
            pbar = st.progress(0.0)

            frame_idx = 0
            while cap.isOpened() and processed < max_frames:
                ret, frame_bgr = cap.read()
                if not ret:
                    break
                if frame_idx % frame_stride != 0:
                    frame_idx += 1
                    continue

                # Convert to RGB for YOLO
                frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)

                # If IR grayscale expected (rare in videos), convert if needed
                if ir_is_grayscale and frame_rgb.ndim == 2:
                    frame_rgb = to_three_channel_if_gray(frame_rgb)

                # Run inference
                results = model.predict(
                    source=frame_rgb,
                    imgsz=int(imgsz),
                    conf=conf_thres,
                    iou=iou_thres,
                    device=device,
                    verbose=False
                )
                r = results[0]
                parsed = parse_yolo_results(r)

                # Draw and write
                vis_bgr = draw_boxes(frame_bgr, parsed["boxes"], parsed["classes"], parsed["scores"], class_names)
                writer.write(vis_bgr)

                # Aggregate class counts
                for cid in parsed["classes"]:
                    cname = class_names[cid] if cid < len(class_names) else str(cid)
                    agg_counts_vid[cname] = agg_counts_vid.get(cname, 0) + 1

                processed += 1
                pbar.progress(min(1.0, processed / max_frames))
                frame_idx += 1

            cap.release()
            writer.release()

            st.success(f"Processed video {upl.name}: {processed} frames (stride={frame_stride})")
            st.video(str(out_video_path))

            # Save per-video summary
            with open(out_dir / "summary.json", "w") as f:
                json.dump({
                    "video": upl.name,
                    "processed_frames": processed,
                    "stride": frame_stride,
                    "imgsz": imgsz,
                    "conf": conf_thres,
                    "iou": iou_thres,
                    "class_counts": agg_counts_vid
                }, f, indent=2)

            st.info(f"Classes detected in {upl.name}:")
            st.json(agg_counts_vid)

            # Offer ZIP of video outputs
            zip_buf = io.BytesIO()
            with zipfile.ZipFile(zip_buf, "w", zipfile.ZIP_DEFLATED) as zf:
                # add overlay video
                with open(out_video_path, "rb") as vf:
                    zf.writestr(f"{Path(upl.name).stem}/overlay.mp4", vf.read())
                # add summary
                with open(out_dir / "summary.json", "rb") as sf:
                    zf.writestr(f"{Path(upl.name).stem}/summary.json", sf.read())
            zip_buf.seek(0)
            st.download_button(
                f"📦 Download outputs for {upl.name} (ZIP)",
                data=zip_buf,
                file_name=f"{Path(upl.name).stem}_results.zip"
            )

st.markdown("---")
st.caption("Notes: IR images may be grayscale; this app repeats the channel to match YOLO’s RGB input. Adjust confidence/IoU for your data. For long videos, increase stride or set max frames.")
