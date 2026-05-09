"""
video_processor.py — Video upload, frame extraction, and basic analysis via OpenCV.

Usage:
    from modules.video_processor import extract_frames, get_video_metadata, render_video_tab
"""

import os
import tempfile
import cv2
import numpy as np
import streamlit as st
from PIL import Image


def get_video_metadata(video_path: str) -> dict:
    """
    Return basic metadata for the video at ``video_path``.

    Returns a dict with keys:
        width, height, fps, frame_count, duration_sec, codec
    """
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        return {}

    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    duration = frame_count / fps if fps > 0 else 0
    fourcc_int = int(cap.get(cv2.CAP_PROP_FOURCC))
    codec = "".join([chr((fourcc_int >> (8 * i)) & 0xFF) for i in range(4)])

    cap.release()
    return {
        "width": width,
        "height": height,
        "fps": round(fps, 2),
        "frame_count": frame_count,
        "duration_sec": round(duration, 2),
        "codec": codec.strip(),
    }


def extract_frames(
    video_path: str,
    max_frames: int = 10,
    strategy: str = "uniform",
) -> list[Image.Image]:
    """
    Extract frames from a video file.

    Args:
        video_path:  Path to the video file.
        max_frames:  Maximum number of frames to return.
        strategy:    "uniform" — evenly spaced frames across the video.
                     "first"   — frames from the first few seconds.

    Returns:
        List of PIL Image objects (RGB).
    """
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        return []

    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    if total_frames == 0:
        cap.release()
        return []

    max_frames = min(max_frames, total_frames)

    if strategy == "uniform":
        indices = np.linspace(0, total_frames - 1, max_frames, dtype=int).tolist()
    else:  # "first"
        indices = list(range(min(max_frames, total_frames)))

    frames = []
    for idx in indices:
        cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
        ret, frame = cap.read()
        if ret:
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            frames.append(Image.fromarray(rgb))

    cap.release()
    return frames


def analyze_brightness(frames: list[Image.Image]) -> list[float]:
    """Return the mean pixel brightness (0–255) for each frame."""
    values = []
    for img in frames:
        arr = np.array(img.convert("L"))  # grayscale
        values.append(round(float(arr.mean()), 1))
    return values


def save_frames_to_output(
    frames: list[Image.Image],
    out_dir: str = "outputs",
    prefix: str = "frame",
) -> list[str]:
    """Save extracted frames as PNG files and return their paths."""
    os.makedirs(out_dir, exist_ok=True)
    paths = []
    for i, img in enumerate(frames):
        path = os.path.join(out_dir, f"{prefix}_{i:04d}.png")
        img.save(path)
        paths.append(path)
    return paths


# ── Streamlit UI component ────────────────────────────────────────────────────

def render_video_tab():
    """Render the full video processing tab inside the Streamlit app."""
    st.header("🎥 Video Processing")
    st.caption("Upload a video to extract frames and run basic analysis.")

    uploaded = st.file_uploader(
        "Upload video",
        type=["mp4", "avi", "mov", "mkv", "webm"],
        help="Supported formats: MP4, AVI, MOV, MKV, WebM",
    )

    if not uploaded:
        st.info("Upload a video file to get started.")
        return

    # Write to a temp file so OpenCV can read it
    with tempfile.NamedTemporaryFile(suffix=os.path.splitext(uploaded.name)[1], delete=False) as tmp:
        tmp.write(uploaded.read())
        tmp_path = tmp.name

    # ── Metadata ──────────────────────────────────────────────────────────────
    meta = get_video_metadata(tmp_path)
    if meta:
        cols = st.columns(4)
        cols[0].metric("Resolution", f"{meta['width']}×{meta['height']}")
        cols[1].metric("FPS", meta["fps"])
        cols[2].metric("Duration", f"{meta['duration_sec']}s")
        cols[3].metric("Total frames", meta["frame_count"])
    else:
        st.error("Could not read video metadata. The file may be corrupted.")
        os.unlink(tmp_path)
        return

    # ── Extraction controls ───────────────────────────────────────────────────
    st.divider()
    col1, col2 = st.columns(2)
    with col1:
        max_frames = st.slider("Frames to extract", 3, 30, 10)
    with col2:
        strategy = st.radio("Sampling strategy", ["uniform", "first"], horizontal=True)

    if st.button("Extract frames", type="primary"):
        with st.spinner("Extracting frames…"):
            frames = extract_frames(tmp_path, max_frames, strategy)

        if not frames:
            st.error("No frames could be extracted.")
        else:
            st.success(f"Extracted {len(frames)} frames.")

            # Brightness chart data
            brightness = analyze_brightness(frames)

            st.subheader("Extracted frames")
            cols_per_row = 5
            for row_start in range(0, len(frames), cols_per_row):
                row_frames = frames[row_start: row_start + cols_per_row]
                cols = st.columns(len(row_frames))
                for col, (img, bv) in zip(cols, zip(row_frames, brightness[row_start:])):
                    col.image(img, caption=f"Brightness: {bv}", use_container_width=True)

            st.subheader("Brightness across frames")
            st.line_chart({"Brightness": brightness})

            # Allow downloading all frames as a zip
            import zipfile, io as _io
            zip_buf = _io.BytesIO()
            with zipfile.ZipFile(zip_buf, "w") as zf:
                for i, img in enumerate(frames):
                    img_buf = _io.BytesIO()
                    img.save(img_buf, format="PNG")
                    zf.writestr(f"frame_{i:04d}.png", img_buf.getvalue())
            zip_buf.seek(0)
            st.download_button(
                "Download frames (.zip)",
                data=zip_buf,
                file_name="storyflow_frames.zip",
                mime="application/zip",
            )

    # Clean up temp file
    try:
        os.unlink(tmp_path)
    except OSError:
        pass
