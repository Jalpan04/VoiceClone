import json
import os
import shutil
from pathlib import Path

# Paths
ANNOTATIONS_FILE = "annotations.json"
FINAL_OUTPUT_DIR = "MyVoice_Consolidated"

def main():
    if not os.path.exists(ANNOTATIONS_FILE):
        print(f"Error: {ANNOTATIONS_FILE} not found.")
        return

    # Load annotations
    with open(ANNOTATIONS_FILE, "r", encoding="utf-8") as f:
        annotations = json.load(f)

    # Find all "mine" folders
    mine_folders = [folder for folder, label in annotations.items() if label == "mine"]
    print(f"Found {len(mine_folders)} folders marked as 'mine'.")

    # Create consolidation directory
    os.makedirs(FINAL_OUTPUT_DIR, exist_ok=True)
    
    total_clips = 0
    copied_clips = 0

    print("Copying files...")
    for folder in mine_folders:
        folder_path = Path(folder)
        if not folder_path.exists():
            print(f"Warning: Folder {folder} does not exist.")
            continue
            
        # Find all .wav files that aren't the temp file
        wav_files = list(folder_path.glob("*.wav"))
        wav_files = [w for w in wav_files if not w.name.endswith("_temp.wav")]
        
        for wav_file in wav_files:
            total_clips += 1
            # We construct a unique filename just in case two files have identical names
            # by putting the folder name (speaker name) in front.
            # Example original: +917048..._SPEAKER_00_clip0001_1.0s-3.0s.wav
            destination_name = wav_file.name
            destination_path = Path(FINAL_OUTPUT_DIR) / destination_name
            
            # If there's a name collision, we'll append a counter
            counter = 1
            while destination_path.exists():
                name_parts = destination_name.rsplit(".", 1)
                base_name = name_parts[0]
                ext = name_parts[1] if len(name_parts) > 1 else "wav"
                destination_path = Path(FINAL_OUTPUT_DIR) / f"{base_name}_{counter}.{ext}"
                counter += 1
            
            try:
                shutil.copy2(wav_file, destination_path)
                copied_clips += 1
            except Exception as e:
                print(f"Failed to copy {wav_file.name}: {e}")

    print("=" * 50)
    print("CONSOLIDATION COMPLETE")
    print("=" * 50)
    print(f"Total Folders: {len(mine_folders)}")
    print(f"Clips Found:   {total_clips}")
    print(f"Clips Copied:  {copied_clips}")
    print(f"\nAll files have been saved to: {Path(FINAL_OUTPUT_DIR).absolute()}")

if __name__ == "__main__":
    main()
