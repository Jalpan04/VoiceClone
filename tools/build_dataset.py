import os
import shutil
import wave
from pathlib import Path

INPUT_DIR = "MyVoice_Consolidated"
OUTPUT_DIR = "TTS_Dataset/raw"
TARGET_MINUTES = 30
TARGET_SECONDS = TARGET_MINUTES * 60

MIN_CLIP_LEN = 4.0
MAX_CLIP_LEN = 10.0

def get_duration(wav_path):
    try:
        with wave.open(str(wav_path), 'rb') as w:
            frames = w.getnframes()
            rate = w.getframerate()
            return frames / float(rate)
    except Exception as e:
        return 0

def main():
    if not os.path.exists(INPUT_DIR):
        print(f"Error: {INPUT_DIR} not found.")
        return

    os.makedirs(OUTPUT_DIR, exist_ok=True)
    
    # Get all wav files
    all_wavs = list(Path(INPUT_DIR).glob("*.wav"))
    print(f"Found {len(all_wavs)} total clips in {INPUT_DIR}.")

    # Gather durations
    valid_clips = []
    for w in all_wavs:
        duration = get_duration(w)
        if MIN_CLIP_LEN <= duration <= MAX_CLIP_LEN:
            valid_clips.append((w, duration))
            
    # Sort clips (maybe longest within the 4-10 window is better? Or just take them)
    # Let's sort by duration descending to get good meaty sentences rather than short grunts
    valid_clips.sort(key=lambda x: x[1], reverse=True)
    
    total_duration = 0.0
    selected_clips = []
    
    for clip_path, duration in valid_clips:
        if total_duration + duration <= TARGET_SECONDS:
            selected_clips.append((clip_path, duration))
            total_duration += duration
        elif total_duration >= TARGET_SECONDS:
            break
            
    print(f"\nSelected {len(selected_clips)} clips matching perfectly (4-10s) to build exactly {total_duration / 60:.2f} minutes of training data.")
    
    # Copying
    print(f"Copying files to {OUTPUT_DIR}...")
    copied = 0
    for clip_path, _ in selected_clips:
        dest_path = Path(OUTPUT_DIR) / clip_path.name
        try:
            shutil.copy2(clip_path, dest_path)
            copied += 1
        except Exception as e:
            print(f"Failed to copy {clip_path}: {e}")
            
    print(f"Done! Copied {copied} files to {OUTPUT_DIR}.")

if __name__ == "__main__":
    main()
