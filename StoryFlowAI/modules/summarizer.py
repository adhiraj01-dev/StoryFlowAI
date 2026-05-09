"""
summarizer.py — AI text summarization using facebook/bart-large-cnn.

Usage:
    from modules.summarizer import load_summarizer, summarize_text
"""

import streamlit as st
from transformers import pipeline, Pipeline


@st.cache_resource(show_spinner=False)
def load_summarizer(device: str = "cpu") -> Pipeline:
    """
    Load the BART summarization pipeline.
    Cached by Streamlit so the model is only downloaded/loaded once per session.
    """
    # Map torch device string to transformers device index
    # transformers pipeline expects -1 for CPU, 0+ for CUDA device index
    device_index = -1
    if device == "cuda":
        device_index = 0
    elif device == "mps":
        # transformers >= 4.30 supports "mps" string directly
        device_index = "mps"

    return pipeline(
        "summarization",
        model="facebook/bart-large-cnn",
        device=device_index,
    )


def summarize_text(
    text: str,
    summarizer: Pipeline,
    max_length: int = 200,
    min_length: int = 50,
) -> str:
    """
    Summarize ``text`` and return the summary string.

    Args:
        text:        Input text (≥ 50 tokens recommended).
        summarizer:  Loaded HuggingFace pipeline from load_summarizer().
        max_length:  Maximum token length of the generated summary.
        min_length:  Minimum token length of the generated summary.

    Returns:
        Summary string, or an error message prefixed with "Error:".
    """
    text = text.strip()
    if not text:
        return "Error: Input text is empty."

    word_count = len(text.split())
    if word_count < 30:
        return "Error: Text is too short to summarize (need at least 30 words)."

    # BART max input is 1024 tokens — truncate to ~900 words to stay safe
    if word_count > 900:
        text = " ".join(text.split()[:900])

    try:
        result = summarizer(
            text,
            max_length=max_length,
            min_length=min_length,
            do_sample=False,
            truncation=True,
        )
        return result[0]["summary_text"]
    except Exception as exc:
        return f"Error: {exc}"


# ── Streamlit UI component ────────────────────────────────────────────────────

def render_summarizer_tab(device: str = "cpu"):
    """Render the full summarization tab inside the Streamlit app."""
    st.header("📝 Text Summarization")
    st.caption("Condense long content using BART (facebook/bart-large-cnn).")

    col1, col2 = st.columns([3, 1])
    with col1:
        input_text = st.text_area(
            "Paste your text here",
            height=250,
            placeholder="Paste an article, report, or any long-form text...",
        )
    with col2:
        max_len = st.slider("Max summary length (tokens)", 50, 400, 200, step=10)
        min_len = st.slider("Min summary length (tokens)", 20, 100, 50, step=5)
        min_len = min(min_len, max_len - 10)

    word_count = len(input_text.split()) if input_text else 0
    st.caption(f"Word count: **{word_count}**")

    if st.button("Summarize", type="primary", disabled=not input_text):
        with st.spinner("Loading model and summarizing…"):
            summarizer = load_summarizer(device)
            summary = summarize_text(input_text, summarizer, max_len, min_len)

        if summary.startswith("Error:"):
            st.error(summary)
        else:
            st.subheader("Summary")
            st.write(summary)
            st.download_button(
                "Download summary (.txt)",
                data=summary,
                file_name="summary.txt",
                mime="text/plain",
            )
