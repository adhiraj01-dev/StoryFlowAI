# utils/helpers.py — shared utility functions for StoryFlow AI

import os
import time
import streamlit as st


def ensure_output_dir(path: str = "outputs") -> str:
    """Create the outputs directory if it doesn't exist and return its path."""
    os.makedirs(path, exist_ok=True)
    return path


def human_readable_size(num_bytes: int) -> str:
    """Convert a byte count to a human-readable string (KB / MB / GB)."""
    for unit in ("B", "KB", "MB", "GB"):
        if num_bytes < 1024:
            return f"{num_bytes:.1f} {unit}"
        num_bytes /= 1024
    return f"{num_bytes:.1f} TB"


def timer(label: str = ""):
    """Context manager that logs elapsed time to the Streamlit sidebar."""
    class _Timer:
        def __enter__(self):
            self._start = time.perf_counter()
            return self
        def __exit__(self, *_):
            elapsed = time.perf_counter() - self._start
            msg = f"{label}: `{elapsed:.2f}s`" if label else f"`{elapsed:.2f}s`"
            st.sidebar.caption(msg)
    return _Timer()
