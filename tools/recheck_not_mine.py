import os
import json
import numpy as np
from pathlib import Path
from scipy.spatial.distance import cosine
from dotenv import load_dotenv
import torchaudio
from pyannote.audio import Inference, Model

# Disable HF network calls to prevent issues
os.environ["HF_HUB_OFFLINE"] = "1"
os.environ["HF_HUB_DISABLE_HTTP2"] = "1"

load_dotenv()
HF_TOKEN = os.getenv("HUGGINGFACE_TOKEN")

THRESHOLD = 0.45  # Lower threshold to catch more potential maybes

def get_clip_duration(filename):
    try:
        parts = filename.replace(".wav", "").split("_")[-1]
        start, end = parts.split("-")
        return float(end.replace("s", "")) - float(start.replace("s", ""))
    except:
        return 0

def get_longest_clip(folder_path):
    clips = list(Path(folder_path).glob("*.wav"))
    if not clips: return None
    return max(clips, key=lambda p: get_clip_duration(p.name))

def main():
    print("Loading Embedding Model from Local Cache...")
    model = Model.from_pretrained("pyannote/wespeaker-voxceleb-resnet34-LM", use_auth_token=HF_TOKEN)
    inference = Inference(model, window="whole")

    ann_path = Path("annotations.json")
    auto_path = Path("autoaot.json")

    with open(ann_path, "r", encoding="utf-8") as f:
        ann = json.load(f)
    
    if auto_path.exists():
        with open(auto_path, "r", encoding="utf-8") as f:
            auto = json.load(f)
    else:
        auto = {}

    def get_embedding(file_path):
        try:
            waveform, sample_rate = torchaudio.load(file_path)
            return inference({"waveform": waveform, "sample_rate": sample_rate})
        except Exception as e:
            print(f"Error on {file_path}: {e}")
            return None

    print("Building User Fingerprint from 'mine' annotations...")
    mine_folders = [p for p, label in ann.items() if label == "mine"]
    mine_clips = []
    for fp in mine_folders:
        lc = get_longest_clip(fp)
        if lc: mine_clips.append(lc)
    
    mine_clips.sort(key=lambda p: get_clip_duration(p.name), reverse=True)
    embeddings = []
    # Up to 15 longest clips to get a solid fingerprint
    for c in mine_clips[:15]:
        emb = get_embedding(c)
        if emb is not None:
            embeddings.append(emb)
    
    if not embeddings:
        print("Error: No valid 'mine' embeddings found.")
        return

    ref_emb = np.mean(embeddings, axis=0)
    print("Fingerprint built.")

    not_mine_keys = [k for k, v in ann.items() if v == "not_mine"]
    print(f"\nChecking {len(not_mine_keys)} 'not_mine' folders for similarity >= {THRESHOLD}...")

    new_maybes = 0

    for idx, folder in enumerate(not_mine_keys):
        lc = get_longest_clip(folder)
        if not lc: continue
        curr_emb = get_embedding(lc)
        if curr_emb is None: continue
        
        sim = 1 - cosine(ref_emb, curr_emb)
        
        if sim >= THRESHOLD:
            auto[folder] = {
                "label": "unsure",
                "similarity": round(float(sim), 3),
                "clip": str(lc).replace("\\", "/")
            }
            del ann[folder]
            new_maybes += 1
            print(f"[{sim:.2f}] Move to maybe -> {folder}")

        if (idx + 1) % 100 == 0:
            print(f"  Processed {idx + 1}/{len(not_mine_keys)}...")

    with open(ann_path, "w", encoding="utf-8") as f:
        json.dump(ann, f, indent=2, ensure_ascii=False)
    with open(auto_path, "w", encoding="utf-8") as f:
        json.dump(auto, f, indent=2, ensure_ascii=False)

    print("\n" + "="*50)
    print(f"Found {new_maybes} new 'maybe' folders!")
    print("="*50)
    print("They have been moved to autoaot.json and will appear in the UI as Maybes.")

if __name__ == '__main__':
    main()
