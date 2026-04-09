import os
import json
import torch
import numpy as np
from pathlib import Path
from pyannote.audio import Inference, Model
from scipy.spatial.distance import cosine
from dotenv import load_dotenv

# Disable HF network calls to prevent httpx crashes on Windows
os.environ["HF_HUB_OFFLINE"] = "1"
os.environ["HF_HUB_DISABLE_HTTP2"] = "1"

# Load env for HG_TOKEN
load_dotenv()
HF_TOKEN = os.getenv("HUGGINGFACE_TOKEN")

# Thresholds
SIMILARITY_HIGH = 0.85  # High confidence
SIMILARITY_LOW = 0.65   # Below this is definitely not mine

def get_clip_duration(filename):
    """Extract duration from suffix like 0.72s-1.90s.wav"""
    try:
        parts = filename.replace(".wav", "").split("_")[-1]
        start, end = parts.split("-")
        return float(end.replace("s", "")) - float(start.replace("s", ""))
    except:
        return 0

import torchaudio

def get_longest_clip(folder_path):
    clips = list(Path(folder_path).glob("*.wav"))
    if not clips:
        return None
    return max(clips, key=lambda p: get_clip_duration(p.name))

class AutoAnnotator:
    def __init__(self):
        print("Loading Embedding Model from Local Cache...")
        try:
            model = Model.from_pretrained("pyannote/wespeaker-voxceleb-resnet34-LM", use_auth_token=HF_TOKEN)
            self.model = Inference(model, window="whole")
        except Exception as e:
            print("Failed to load model locally. Ensure it's cached.")
            raise e
        self.annotations_path = Path("annotations.json")
        self.auto_path = Path("autoaot.json")
        
        self.annotations = {}
        if self.annotations_path.exists():
            with open(self.annotations_path, "r") as f:
                self.annotations = json.load(f)

    def get_embedding(self, file_path):
        try:
            # Bypass torchcodec warning using torchaudio
            waveform, sample_rate = torchaudio.load(file_path)
            # Inference expects (num_channels, num_samples) shaped tensor wrapped as dictionary
            embedding = self.model({"waveform": waveform, "sample_rate": sample_rate})
            return embedding
        except Exception as e:
            print(f"Error extracting embedding for {file_path}: {e}")
            return None

    def build_reference(self):
        print("Building User Fingerprint from existing annotations...")
        mine_folders = [p for p, label in self.annotations.items() if label == "mine"]
        
        embeddings = []
        # Take up to 10 longest "mine" clips to build a robust average
        mine_clips = []
        for folder in mine_folders:
            longest = get_longest_clip(folder)
            if longest:
                mine_clips.append(longest)
        
        # Sort by presumed duration (implied in name) and take top 10
        mine_clips.sort(key=lambda p: get_clip_duration(p.name), reverse=True)
        
        for clip in mine_clips[:10]:
            emb = self.get_embedding(clip)
            if emb is not None:
                embeddings.append(emb)
        
        if not embeddings:
            raise ValueError("No 'mine' annotations found to build reference!")
            
        return np.mean(embeddings, axis=0)

    def run(self):
        ref_emb = self.build_reference()
        print("Fingerprint built. Scanning for new clips...")
        
        output_dir = Path("output")
        auto_aot = {}
        new_annotations = {}
        
        # Count stats
        automated_mine = 0
        automated_not_mine = 0
        unsure_count = 0

        # Walk through output folders
        for source_folder in output_dir.iterdir():
            if not source_folder.is_dir(): continue
            
            for speaker_folder in source_folder.iterdir():
                if not speaker_folder.is_dir(): continue
                
                folder_str = str(speaker_folder).replace("\\", "/")
                
                # Skip if already annotated
                if folder_str in self.annotations:
                    continue
                
                longest_clip = get_longest_clip(speaker_folder)
                if not longest_clip:
                    continue
                
                # Get similarity
                current_emb = self.get_embedding(longest_clip)
                if current_emb is None: continue
                
                # Cosine similarity (1 - cosine distance)
                similarity = 1 - cosine(ref_emb, current_emb)
                
                if similarity >= SIMILARITY_HIGH:
                    new_annotations[folder_str] = "mine"
                    automated_mine += 1
                    print(f"[AUTO-MINE] {folder_str} (Sim: {similarity:.2f})")
                elif similarity <= SIMILARITY_LOW:
                    new_annotations[folder_str] = "not_mine"
                    automated_not_mine += 1
                else:
                    auto_aot[folder_str] = {
                        "label": "unsure",
                        "similarity": round(float(similarity), 3),
                        "clip": str(longest_clip).replace("\\", "/")
                    }
                    unsure_count += 1
                    print(f"[UNSURE] {folder_str} (Sim: {similarity:.2f})")

        # Update annotations.json
        if new_annotations:
            self.annotations.update(new_annotations)
            with open(self.annotations_path, "w") as f:
                json.dump(self.annotations, f, indent=2)
            print(f"Updated annotations.json with {len(new_annotations)} automated tags.")

        # Write autoaot.json
        with open(self.auto_path, "w") as f:
            json.dump(auto_aot, f, indent=2)
            
        print("\n--- Automation Summary ---")
        print(f"High-Confidence Mine:     {automated_mine}")
        print(f"High-Confidence Not Mine: {automated_not_mine}")
        print(f"Unsure (to review):      {unsure_count}")
        print(f"Results saved to annotations.json and {self.auto_path}")

if __name__ == "__main__":
    annotator = AutoAnnotator()
    annotator.run()
