import os
import torch
from TTS.api import TTS

# =========================================================
# RUN THIS SCRIPT AFTER YOU FINISH TRAINING IN PINOKIO
# =========================================================

# When Pinokio finishes training, it will output a folder containing:
# 1. config.json
# 2. best_model.pth or model_file.pth
# 3. vocab.json

# Update these paths to where your new model is located!
MODEL_PATH = "C:/path/to/your/trained/best_model.pth"
CONFIG_PATH = "C:/path/to/your/trained/config.json"
VOCAB_PATH = "C:/path/to/your/trained/vocab.json"

# You need one clean clip of your voice to act as the "Speaker Reference"
# Pick one of your enhanced audio files from the dataset.
REFERENCE_AUDIO = "TTS_Dataset/wavs/YOUR_PREFERRED_CLIP.wav"

OUTPUT_FILE = "my_first_clone.wav"

def main():
    if not os.path.exists(MODEL_PATH):
        print(f"Waiting for your trained model at: {MODEL_PATH}")
        print("Please train your model in Pinokio first and update the paths in this script!")
        return

    print("Loading your Custom XTTS v2 Model...")
    
    # Load your fine-tuned model
    tts = TTS(model_path=MODEL_PATH, config_path=CONFIG_PATH)

    # -----------------------------------------------------
    # Type your text here! 
    # Since your dataset was transcribed in Hindi, write the
    # prompt in Hindi (Devanagari) to get the best cloned output!
    # -----------------------------------------------------
    text_to_speak = "नमस्ते, यह मेरी अपनी एआई आवाज़ है जिसे मैंने सफलतापूर्वक क्लोन किया है।"

    print(f"\nGenerating Speech -> '{text_to_speak}'")

    # The language must match one of XTTS v2's supported languages.
    # Set to "hi" matching your training dataset
    tts.tts_to_file(
        text=text_to_speak,
        file_path=OUTPUT_FILE,
        speaker_wav=REFERENCE_AUDIO,
        language="hi" 
    )

    print(f"\nSuccess! Check out your cloned voice here: {os.path.abspath(OUTPUT_FILE)}")

if __name__ == "__main__":
    main()
