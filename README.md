# VoiceClone: A Technical Guide to High-Fidelity Hindi Voice Cloning

![GitHub top language](https://img.shields.io/github/languages/top/Jalpan04/VoiceClone) ![GitHub repo size](https://img.shields.io/github/repo-size/Jalpan04/VoiceClone) [![License](https://img.shields.io/badge/License-Apache_2.0-blue.svg)](LICENSE)

This article documents the comprehensive process of creating a high-quality Hindi voice clone using the XTTS-v2 architecture. It chronicles the journey from raw call recordings to a fully optimized, production-ready model, detailing every technical obstacle encountered and the specialized solutions developed to overcome them.

## Table of Contents
1. [Introduction and Project Scope](#introduction-and-project-scope)
2. [Data Acquisition: Processing Call Recordings](#data-acquisition-processing-call-recordings)
    - [Diarization and Speaker Splitting](#diarization-and-speaker-splitting)
    - [The Sanitization Battle: ASCII vs Emojis](#the-sanitization-battle-ascii-vs-emojis)
3. [Architectural Challenges and Failed Methods](#architectural-challenges-and-failed-methods)
    - [The 6681-Token Size Mismatch](#the-6681-token-size-mismatch)
    - [Experimental Failures: Manual Weight Resizing](#experimental-failures-manual-weight-resizing)
4. [Stability Patches for Windows Training](#stability-patches-for-windows-training)
    - [Resolving Infinite Recursion in Dataset Loaders](#resolving-infinite-recursion-in-dataset-loaders)
    - [Handling Windows Multiprocessing and File Locks](#handling-windows-multiprocessing-and-file-locks)
5. [Inference Optimization: Fixing the Mumbling Output](#inference-optimization-fixing-the-mumbling-output)
6. [Conclusion: The Final Replicable Workflow](#conclusion-the-final-replicable-workflow)

---

## Introduction and Project Scope

The objective of this project was to achieve a natural-sounding Hindi voice clone using a limited dataset derived from real-world call recordings. The primary constraints included Windows-based development, an 8GB VRAM GPU limit, and the inherent complexities of the Hindi language within the XTTS-v2 framework.

## Data Acquisition: Processing Call Recordings

The process began with raw, single-channel call recordings. These recordings contained multiple speakers and varying noise floors, making them unsuitable for direct training.

### Diarization and Speaker Splitting
To isolate the target voice, we utilized a specialized diarization script (`diarize_and_split.py`). This script performed the following:
- Voice Activity Detection (VAD) to remove silence.
- Speaker Embeddings to distinguish the target speaker from the caller.
- Automated splitting of the long recording into 2-10 second chunks optimized for training.

### The Sanitization Battle: ASCII vs Emojis
A significant early hurdle was the Windows file system. Original files were named using descriptive text and emojis, which caused the training metadata formatter to fail. 
- **Problem**: Python encoding errors when reading paths containing non-ASCII characters.
- **Solution**: We implemented `tools/sanitize_metadata.py`, which systematically renamed every audio file to a sequential format (e.g., `s00001.wav`) and rebuilt the metadata CSV from scratch.

## Architectural Challenges and Failed Methods

### The 6681-Token Size Mismatch
The most critical technical blocker was a `RuntimeError: size mismatch` when loading the Hindi base model. Specifically, the model expected a vocabulary size of 6681 tokens, but standard XTTS-v2 distributions often defaulted to 6153.
- **Root Cause**: XTTS-v2 uses different tokenizers for different languages, and the Hindi-compatible weights were not being correctly synchronized by the default model manager.

### Experimental Failures: Manual Weight Resizing
Before finding the correct solution, we attempted several failed methods:
- **Resizing base weights**: Attempting to pad the embedding layer from 6153 to 6681. This resulted in "hallucinating" audio and garbled speech.
- **Partial Downloads**: Attempting to swap only the `vocab.json` while keeping old weights. This led to immediate crashes during the first training epoch.
- **Final Solution**: The creation of `tools/sync_base_models.py`, which utilizes the HuggingFace Hub API to pull the verified 6681-token weights, configuration, and vocabulary files.

## Stability Patches for Windows Training

Training on Windows presented unique stability issues that are not present in Linux environments.

### Resolving Infinite Recursion in Dataset Loaders
During training, if a single audio sample was slightly corrupted or too short, the `TTS` library would recursively attempt to load the next sample.
- **The Bug**: `RecursionError: maximum recursion depth exceeded`.
- **The Patch**: We modified `patches/dataset_patch.py` (specifically `venv/Lib/site-packages/TTS/tts/layers/xtts/trainer/dataset.py`) to replace the recursive call with a robust `while` loop and retry logic.

### Handling Windows Multiprocessing and File Locks
Windows does not support the same `fork` mechanism as Linux for data loading.
- **Problem**: `PermissionError` and file locks when `num_loader_workers` was set above 0.
- **Solution**: We patched `xtts-finetune-webui/utils/gpt_train.py` to force `num_loader_workers=0` on Windows, ensuring single-threaded but stable data loading.

## Inference Optimization: Fixing the Mumbling Output

Even with a successfully trained model, early inference was characterized by "mumbling" or skipping phonemes. This was traced back to two specific configuration issues:
1. **Repetition Penalty**: The default value of `5.0` was too aggressive for Hindi, forcing the model to skip repetitive vowels common in the language. We lowered this to `1.1`.
2. **JSON Syntax Errors**: The optimized `config.json` sometimes contained `Infinity` values (an artifact of the training loss). This is invalid in standard JSON and caused the inference engine to fail. We sanitized these values to `null`.

## Conclusion: The Final Replicable Workflow

To replicate this process for any other voice, follow these steps:
1. **Diarize**: Isolate the speaker using `diarize_and_split.py`.
2. **Sanitize**: Rename files to ASCII using `tools/sanitize_metadata.py`.
3. **Sync**: Download the correct 6681-token base weights using `tools/sync_base_models.py`.
4. **Patch**: Apply the recursion patch in `patches/dataset_patch.py`.
5. **Train**: Run the fine-tuner with `num_loader_workers=0` and language set to `hi`.
6. **Optimize**: Use Step 3 in the WebUI and set the repetition penalty to `1.1`.

---
*Documentation maintained by Jalpan04*

## License

This project is licensed under the Apache-2.0 License - see the [LICENSE](LICENSE) file for details.
