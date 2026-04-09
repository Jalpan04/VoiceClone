"""
Pipeline loader -- handles loading the pyannote speaker diarization model.
"""

import torch
from pyannote.audio import Pipeline


def load_pipeline(hf_token: str) -> Pipeline:
    """Load the pyannote speaker diarization pipeline onto GPU if available."""
    print("[INFO] Loading pyannote/speaker-diarization-3.1 pipeline...")
    print("       (First run downloads ~1GB of model weights)")

    pipeline = Pipeline.from_pretrained(
        "pyannote/speaker-diarization-3.1",
        token=hf_token,
    )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    pipeline.to(device)
    print(f"[INFO] Running on: {device}")

    return pipeline
