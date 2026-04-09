# VoiceClone: Hindi XTTS-v2 Fine-Tuning Pipeline

A specialized pipeline for high-quality Hindi voice cloning using XTTS-v2. This repository contains the specific fixes required to handle the Hindi token mismatch and common Windows environments blockers.

## 🚀 Key Features
- **Hindi Support**: Fixes the 6153 vs 6681 token mismatch in the official XTTS-v2 model.
- **Windows Stability**: Patches for `num_loader_workers` and file locking.
- **Recursion Fix**: Robust retry logic for the XTTS Dataset loader.
- **Optimized Workflow**: Integrated scripts for base model synchronization and metadata sanitization.

## 🛠 Prerequisites
- **Python 3.11** (Recommended for stability).
- **GPU** with at least 8GB VRAM (12GB+ recommended for fine-tuning).
- **Git** and **FFmpeg** installed.

## 📋 Step-by-Step Setup

### 1. Clone & Environment
```bash
git clone https://github.com/Jalpan04/VoiceClone
cd VoiceClone
python -m venv venv
.\venv\Scripts\activate
pip install -r xtts-finetune-webui/requirements.txt
```

### 2. The "Hindi Problem" (Token Mismatch)
The official XTTS-v2 base model often defaults to a 6153-token vocabulary, but Hindi support requires **6681 tokens**. Running with mismatched weights will cause a `size mismatch` error.

**The Fix:**
Run our synchronization tool to pull the correct 6681-token weights from HuggingFace:
```bash
python tools/sync_base_models.py
```
This script will download the matching `model.pth`, `config.json`, and `vocab.json` into the `base_models/v2.0.2/` directory.

### 3. Applying Code Patches
We have provided a critical patch for the `TTS` library to prevent infinite recursion on failed audio samples.
1. Locate your `venv` site-packages: `venv\Lib\site-packages\TTS\tts\layers\xtts\trainer\dataset.py`.
2. Replace its contents with `/patches/dataset_patch.py` (or apply the changes manually).

### 4. Training on Windows
In `xtts-finetune-webui/utils/gpt_train.py`:
- Always set `num_loader_workers=0` to avoid Windows file-lock errors.
- Ensure the `hi` language key is injected into the training samples.

## 🎙 Preparing Your Data
1. **Rename Files**: Use `tools/sanitize_metadata.py` to convert emoji/special character filenames into sequential ASCII (e.g., `s00001.wav`).
2. **Metadata**: Ensure your `metadata.csv` is in the format: `audio_file|text|speaker_name`.

## 🧪 Inference & Optimization
After training (~20+ epochs recommended for Hindi):
1. Use **Step 3 - Optimize** in the WebUI to convert your 5.6GB model into a production-ready 1.8GB version.
2. Use **Speaker Reference `s00129.wav`** (or any 6-10 second clear clip) for the best pitch matching.

## 📜 Credits
Based on the [XTTS-finetune-webui](https://github.com/daswer123/xtts-finetune-webui) project with specialized stability patches.

---
**Disclaimer**: This project is for educational purposes. Always obtain permission before cloning a voice.
