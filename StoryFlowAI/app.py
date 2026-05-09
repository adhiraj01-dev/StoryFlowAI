"""
app.py — StoryFlow AI · Main Streamlit entry point.

Run with:
    streamlit run app.py
"""

import streamlit as st

# ── Page config (must be first Streamlit call) ────────────────────────────────
st.set_page_config(
    page_title="StoryFlow AI",
    page_icon="🚀",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Internal imports ──────────────────────────────────────────────────────────
from modules.gpu_utils import log_device_info, get_device, get_dtype
from modules.summarizer import render_summarizer_tab
from modules.image_generator import render_image_tab
from modules.tts import render_tts_tab
from modules.video_processor import render_video_tab
from utils.helpers import ensure_output_dir

# ── Init ──────────────────────────────────────────────────────────────────────
ensure_output_dir("outputs")


# ── Sidebar ───────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("# 🚀 StoryFlow AI")
    st.markdown(
        "An all-in-one generative AI toolkit: "
        "summarize text, generate images, convert to speech, and process video."
    )
    st.markdown("---")
    st.markdown("### Navigation")
    active_tab = st.radio(
        "Choose a module",
        ["📝 Summarize", "🎨 Image Gen", "🎙️ Text-to-Speech", "🎥 Video"],
        label_visibility="collapsed",
    )

    # Show device info at the bottom of the sidebar
    device, dtype = log_device_info()


# ── Main content ──────────────────────────────────────────────────────────────
device_str = str(device)   # "cuda", "mps", or "cpu"

if active_tab == "📝 Summarize":
    render_summarizer_tab(device=device_str)

elif active_tab == "🎨 Image Gen":
    render_image_tab(device=device_str, dtype=dtype)

elif active_tab == "🎙️ Text-to-Speech":
    render_tts_tab()

elif active_tab == "🎥 Video":
    render_video_tab()
