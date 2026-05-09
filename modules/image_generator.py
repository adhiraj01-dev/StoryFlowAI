"""
image_generator.py — AI image generation using Stable Diffusion v1.5.

Usage:
    from modules.image_generator import load_sd_pipeline, generate_image
"""

import io
import torch
import streamlit as st
from diffusers import StableDiffusionPipeline, DPMSolverMultistepScheduler
from PIL import Image


MODEL_ID = "runwayml/stable-diffusion-v1-5"


@st.cache_resource(show_spinner=False)
def load_sd_pipeline(device: str = "cpu", dtype: torch.dtype = torch.float32):
    """
    Load Stable Diffusion v1.5 with DPM-Solver++ scheduler (faster convergence).
    Cached by Streamlit — loaded once per session.
    """
    pipe = StableDiffusionPipeline.from_pretrained(
        MODEL_ID,
        torch_dtype=dtype,
        safety_checker=None,          # disable NSFW filter for speed; re-enable if needed
        requires_safety_checker=False,
    )

    # Swap to DPM-Solver++ for ~2× faster inference vs default PNDM
    pipe.scheduler = DPMSolverMultistepScheduler.from_config(pipe.scheduler.config)

    pipe = pipe.to(device)

    # Memory optimizations
    if device == "cuda":
        pipe.enable_attention_slicing()   # reduces peak VRAM usage
    if device == "cpu":
        pipe.enable_sequential_cpu_offload()  # streams layers to RAM

    return pipe


def generate_image(
    prompt: str,
    pipeline,
    negative_prompt: str = "",
    num_inference_steps: int = 25,
    guidance_scale: float = 7.5,
    width: int = 512,
    height: int = 512,
    seed: int = -1,
) -> Image.Image | None:
    """
    Generate an image from a text prompt.

    Args:
        prompt:               Positive text description.
        pipeline:             Loaded StableDiffusionPipeline.
        negative_prompt:      Things to avoid in the image.
        num_inference_steps:  Denoising steps (15–50 typical range).
        guidance_scale:       Prompt adherence strength (1–20; 7–8 is default).
        width / height:       Output dimensions — must be multiples of 8.
        seed:                 RNG seed for reproducibility; -1 = random.

    Returns:
        PIL Image, or None on error.
    """
    if not prompt.strip():
        st.error("Prompt cannot be empty.")
        return None

    generator = None
    if seed >= 0:
        generator = torch.Generator(device=pipeline.device).manual_seed(seed)

    try:
        result = pipeline(
            prompt=prompt,
            negative_prompt=negative_prompt or None,
            num_inference_steps=num_inference_steps,
            guidance_scale=guidance_scale,
            width=width,
            height=height,
            generator=generator,
        )
        return result.images[0]
    except Exception as exc:
        st.error(f"Generation failed: {exc}")
        return None


def image_to_bytes(img: Image.Image, fmt: str = "PNG") -> bytes:
    """Convert a PIL image to a bytes object for download."""
    buf = io.BytesIO()
    img.save(buf, format=fmt)
    return buf.getvalue()


# ── Streamlit UI component ────────────────────────────────────────────────────

def render_image_tab(device: str = "cpu", dtype: torch.dtype = torch.float32):
    """Render the full image generation tab inside the Streamlit app."""
    st.header("🎨 AI Image Generation")
    st.caption("Generate images from text prompts using Stable Diffusion v1.5.")

    prompt = st.text_area(
        "Prompt",
        placeholder="A futuristic city at sunset, cyberpunk, detailed, 4K",
        height=100,
    )
    negative_prompt = st.text_area(
        "Negative prompt (optional)",
        placeholder="blurry, low quality, watermark, text",
        height=70,
    )

    col1, col2, col3 = st.columns(3)
    with col1:
        steps = st.slider("Inference steps", 10, 50, 25)
    with col2:
        guidance = st.slider("Guidance scale", 1.0, 20.0, 7.5, step=0.5)
    with col3:
        seed = st.number_input("Seed (-1 = random)", value=-1, step=1)

    size_options = {"512 × 512": (512, 512), "768 × 512": (768, 512), "512 × 768": (512, 768)}
    size_label = st.selectbox("Output size", list(size_options.keys()))
    w, h = size_options[size_label]

    if st.button("Generate image", type="primary", disabled=not prompt):
        with st.spinner("Loading Stable Diffusion and generating…"):
            pipe = load_sd_pipeline(device, dtype)
            img = generate_image(
                prompt, pipe, negative_prompt, steps, guidance, w, h, int(seed)
            )

        if img:
            st.image(img, caption=prompt[:80], use_container_width=True)
            st.download_button(
                "Download image (PNG)",
                data=image_to_bytes(img),
                file_name="storyflow_generated.png",
                mime="image/png",
            )
