import os
import json
import numpy as np
from pathlib import Path
from scipy.spatial.distance import cosine
from dotenv import load_dotenv
import torchaudio
from pyannote.audio import Inference, Model

os.environ["HF_HUB_OFFLINE"] = "1"
os.environ["HF_HUB_DISABLE_HTTP2"] = "1"

load_dotenv()
HF_TOKEN = os.getenv("HUGGINGFACE_TOKEN")

THRESHOLD = 0.50

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
    print("Loading Embedding Model...")
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
            # Ensure it's valid
            if waveform.numel() == 0: return None
            return np.squeeze(inference({"waveform": waveform, "sample_rate": sample_rate}))
        except Exception as e:
            return None

    print("Building GOLD STANDARD Fingerprint from ALL 'mine' annotations...")
    mine_folders = [p for p, label in ann.items() if label == "mine"]
    mine_clips = []
    for fp in mine_folders:
        lc = get_longest_clip(fp)
        if lc: mine_clips.append(lc)
    
    # Take top 30 longest 
    mine_clips.sort(key=lambda p: get_clip_duration(p.name), reverse=True)
    embeddings = []
    
    for c in mine_clips[:30]: 
        emb = get_embedding(c)
        if emb is not None:
            # Flatten to 1D to avoid Scipy ValueError
            embeddings.append(np.array(emb).flatten())
    
    if not embeddings:
        print("Error: No valid 'mine' embeddings found.")
        return

    ref_emb = np.mean(embeddings, axis=0)
    print(f"GOLD STANDARD built using top {len(embeddings)} longest 'mine' clips.")

    folders_to_check = [k for k, v in ann.items() if v == "not_mine"] + list(auto.keys())
    print(f"\nChecking {len(folders_to_check)} remaining folders against Gold Standard (>= {THRESHOLD})...")

    new_maybes = 0

    for idx, folder in enumerate(folders_to_check):
        lc = get_longest_clip(folder)
        if not lc: continue
        curr_emb = get_embedding(lc)
        if curr_emb is None: continue
        
        curr_emb = np.array(curr_emb).flatten()
        
        try:
            sim = 1 - cosine(ref_emb, curr_emb)
        except Exception as e:
            print(f"Skipping {folder} due to distance error: {e}")
            continue
            
        if sim >= THRESHOLD:
            auto[folder] = {
                "label": "unsure",
                "similarity": round(float(sim), 3),
                "clip": str(lc).replace("\\", "/")
            }
            if folder in ann:
                del ann[folder]
                new_maybes += 1
            print(f"[{sim:.2f}] MATCH -> {folder}")

        if (idx + 1) % 50 == 0:
            print(f"  Processed {idx + 1}/{len(folders_to_check)}...")

    with open(ann_path, "w", encoding="utf-8") as f:
        json.dump(ann, f, indent=2, ensure_ascii=False)
    with open(auto_path, "w", encoding="utf-8") as f:
        json.dump(auto, f, indent=2, ensure_ascii=False)

    print("\n" + "="*50)
    print(f"Found {new_maybes} NEW stray folders that match the Gold Standard!")
    print(f"Total Maybes now pending review: {len(auto)}")
    print("="*50)

if __name__ == '__main__':
    main()
