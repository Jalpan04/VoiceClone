import os
import torch
import warnings
import whisper
from df.enhance import enhance, init_df, load_audio, save_audio
from pathlib import Path
import csv

# Suppress warnings for cleaner output
warnings.filterwarnings("ignore")

RAW_DIR = "TTS_Dataset/raw"
CLEAN_DIR = "TTS_Dataset/wavs"
METADATA_CSV = "TTS_Dataset/metadata.csv"

def main():
    os.makedirs(CLEAN_DIR, exist_ok=True)
    
    raw_files = list(Path(RAW_DIR).glob("*.wav"))
    if not raw_files:
        print(f"Error: No files found in {RAW_DIR}")
        return

    print("Loading DeepFilterNet model for studio-quality audio enhancement...")
    df_model, df_state, _ = init_df()

    print("Loading Whisper model (large-v3-turbo) for transcription... (This will download if first time)")
    # Using small or medium is also an option, but large-v3-turbo handles code-switching best and fits in 8GB VRAM
    whisper_model = whisper.load_model("large-v3-turbo", device="cuda" if torch.cuda.is_available() else "cpu")

    metadata = []
    
    print(f"\nProcessing {len(raw_files)} clips (~30 minutes of audio)...")
    print("This will enhance the audio and transcribe the text for XTTS v2.\n")

    for i, file_path in enumerate(raw_files, 1):
        filename = file_path.name
        out_path = Path(CLEAN_DIR) / filename
        
        # 1. Enhance the audio (Remove phone static/noise)
        try:
            audio, _ = load_audio(file_path, sr=df_state.sr())
            enhanced = enhance(df_model, df_state, audio)
            save_audio(out_path, enhanced, df_state.sr())
        except Exception as e:
            print(f"[{i}/{len(raw_files)}] Error enhancing {filename}: {e}")
            continue

        # 2. Transcribe using Whisper
        try:
            # We enforce Indian English / Hindi / Gujarati context by not forcing a language,
            # letting Whisper auto-detect the best transcription script.
            result = whisper_model.transcribe(str(out_path))
            text = result["text"].strip().replace("\n", " ").replace("|", " ")
        except Exception as e:
            print(f"[{i}/{len(raw_files)}] Error transcribing {filename}: {e}")
            continue

        # XTTS formatting (metadata.csv): wav_file_name|transcript|transcript
        # (Where transcript is duplicated for XTTS v2 standard LJSpeech mapping)
        metadata.append((filename, text, text))
        
        print(f"[{i}/{len(raw_files)}] {filename} -> {text}")

    # Write metadata.csv
    with open(METADATA_CSV, 'w', encoding='utf-8', newline='') as f:
        writer = csv.writer(f, delimiter='|')
        writer.writerows(metadata)

    print("\n" + "=" * 50)
    print("DATASET PREPARATION COMPLETE!")
    print("=" * 50)
    print(f"Enhanced wavs saved to: {Path(CLEAN_DIR).absolute()}")
    print(f"Metadata file saved to: {Path(METADATA_CSV).absolute()}")
    print("\nNEXT STEPS:")
    print("1. Open TTS_Dataset/metadata.csv in Excel/Notepad.")
    print("2. Whisper did its best to transcribe your Hinglish/Gujarati, but it may have hallucinated or used wrong spellings.")
    print("3. CAREFULLY review and correct the text to match exactly what you hear in each clip.")
    print("4. You are ready to fine-tune XTTS v2!")

if __name__ == "__main__":
    main()
