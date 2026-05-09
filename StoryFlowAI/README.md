# 🚀 StoryFlow AI

An all-in-one AI-powered application built with Python and Streamlit that combines multiple generative AI capabilities into a single modern interface.

---

## ✨ Features

| Module | Technology | Capability |
|--------|-----------|------------|
| 📝 Text Summarization | `facebook/bart-large-cnn` | Condense long-form content instantly |
| 🎨 Image Generation | Stable Diffusion v1.5 | Generate images from text prompts |
| 🎙️ Text-to-Speech | gTTS | Multi-language MP3 synthesis |
| 🎥 Video Processing | OpenCV | Frame extraction & brightness analysis |
| ⚡ GPU Optimization | PyTorch CUDA | Auto-detect GPU, fp16, memory management |

---

## 🛠 Tech Stack

- **Python 3.10**
- **Streamlit** — UI framework
- **PyTorch 2.2** — deep learning backend
- **HuggingFace Transformers & Diffusers** — pre-trained models
- **Accelerate** — device management
- **OpenCV** — video frame processing
- **Pillow / NumPy** — image utilities
- **gTTS** — text-to-speech synthesis

---

## 📂 Project Structure

```
StoryFlowAI/
│
├── app.py                  # Main Streamlit entry point
├── requirements.txt
│
├── modules/
│   ├── __init__.py
│   ├── gpu_utils.py        # CUDA detection, fp16, VRAM stats
│   ├── summarizer.py       # BART summarization + UI
│   ├── image_generator.py  # Stable Diffusion + UI
│   ├── tts.py              # gTTS text-to-speech + UI
│   └── video_processor.py  # OpenCV frame extraction + UI
│
├── utils/
│   ├── __init__.py
│   └── helpers.py          # Shared utilities (timer, sizes, dirs)
│
├── assets/                 # Static files (logos, icons)
├── models/                 # Local model cache (optional)
└── outputs/                # Generated files (images, audio, frames)
```

---

## 🚀 Installation

### 1. Clone the repository
```bash
git clone https://github.com/yourusername/StoryFlowAI.git
cd StoryFlowAI
```

### 2. Create a Conda environment
```bash
conda create -n storyflow python=3.10
conda activate storyflow
```

### 3. Install dependencies
```bash
pip install -r requirements.txt
```

> **GPU users:** Install the CUDA-enabled PyTorch wheel first:
> ```bash
> pip install torch==2.2.2 torchvision==0.17.2 --index-url https://download.pytorch.org/whl/cu121
> ```
> Then run `pip install -r requirements.txt` as usual.

### 4. Run the app
```bash
streamlit run app.py
```

Open `http://localhost:8501` in your browser.

---

## ⚡ GPU / CPU Behaviour

| Device | Dtype | Notes |
|--------|-------|-------|
| CUDA GPU | fp16 | Fastest; ~4 GB VRAM minimum for SD v1.5 |
| Apple MPS | fp32 | Supported on M1/M2/M3 Macs |
| CPU | fp32 | Slowest; SD image gen may take several minutes |

The app auto-detects the best device at startup and shows it in the sidebar.

---

## 🔥 Learning Outcomes

This project covers:
- Multi-model AI integration in a single UI
- CUDA acceleration and graceful CPU fallback
- GPU memory management (`attention_slicing`, `sequential_cpu_offload`)
- Streamlit `@st.cache_resource` for model caching
- OpenCV video I/O and frame sampling
- Packaging Python AI projects cleanly

---

## 📌 Future Improvements

- [ ] Real-time AI chat (Claude / GPT integration)
- [ ] Voice cloning (Coqui TTS / ElevenLabs)
- [ ] Advanced video AI (object detection, scene summarization)
- [ ] Cloud deployment (Hugging Face Spaces / AWS)
- [ ] REST API layer (FastAPI)
- [ ] Multi-user authentication

---

## 📜 License

MIT License — open-source and free to use.
