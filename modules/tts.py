"""
tts.py — Text-to-Speech conversion using gTTS.

Usage:
    from modules.tts import text_to_speech, render_tts_tab
"""

import io
import streamlit as st
from gtts import gTTS
from gtts.lang import tts_langs


# Curated subset of commonly used languages with display names
LANGUAGE_MAP = {
    "English": "en",
    "Hindi": "hi",
    "French": "fr",
    "German": "de",
    "Spanish": "es",
    "Portuguese": "pt",
    "Italian": "it",
    "Japanese": "ja",
    "Korean": "ko",
    "Chinese (Mandarin)": "zh-CN",
    "Arabic": "ar",
    "Russian": "ru",
    "Dutch": "nl",
    "Polish": "pl",
    "Turkish": "tr",
}


def text_to_speech(text: str, lang_code: str = "en", slow: bool = False) -> bytes | None:
    """
    Convert ``text`` to speech and return the MP3 as bytes.

    Args:
        text:      The text to synthesize.
        lang_code: BCP-47 language code (e.g. "en", "hi", "fr").
        slow:      If True, speak at a slower pace (useful for language learning).

    Returns:
        MP3 bytes, or None on error.
    """
    text = text.strip()
    if not text:
        return None

    # gTTS has a soft limit around ~5000 chars per request
    if len(text) > 5000:
        text = text[:5000]
        st.warning("Text truncated to 5 000 characters for synthesis.")

    try:
        tts = gTTS(text=text, lang=lang_code, slow=slow)
        buf = io.BytesIO()
        tts.write_to_fp(buf)
        buf.seek(0)
        return buf.read()
    except Exception as exc:
        st.error(f"TTS failed: {exc}")
        return None


def get_supported_languages() -> dict[str, str]:
    """Return all languages supported by gTTS as {display_name: lang_code}."""
    try:
        raw = tts_langs()  # {code: name}
        return {name: code for code, name in sorted(raw.items(), key=lambda x: x[1])}
    except Exception:
        return LANGUAGE_MAP


# ── Streamlit UI component ────────────────────────────────────────────────────

def render_tts_tab():
    """Render the full TTS tab inside the Streamlit app."""
    st.header("🎙️ Text-to-Speech")
    st.caption("Convert any text to natural speech and download as MP3.")

    input_text = st.text_area(
        "Text to speak",
        height=200,
        placeholder="Type or paste the text you want to convert to speech…",
    )

    col1, col2 = st.columns(2)
    with col1:
        lang_display = st.selectbox("Language", list(LANGUAGE_MAP.keys()), index=0)
        lang_code = LANGUAGE_MAP[lang_display]
    with col2:
        slow_mode = st.checkbox("Slow speech", value=False)
        st.caption("Useful for language learning or accessibility.")

    char_count = len(input_text)
    st.caption(f"Characters: **{char_count}** / 5 000")

    if st.button("Convert to speech", type="primary", disabled=not input_text):
        with st.spinner("Synthesizing speech…"):
            mp3_bytes = text_to_speech(input_text, lang_code, slow_mode)

        if mp3_bytes:
            st.audio(mp3_bytes, format="audio/mp3")
            st.download_button(
                "Download MP3",
                data=mp3_bytes,
                file_name="storyflow_speech.mp3",
                mime="audio/mpeg",
            )
            st.success(f"Generated {len(mp3_bytes) / 1024:.1f} KB of audio in {lang_display}.")
