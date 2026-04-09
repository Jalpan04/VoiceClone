"""
Diarizer -- runs speaker diarization on a single audio file.

Loads audio via torchaudio (bypassing pyannote's broken torchcodec reader)
and passes a waveform dict to the pipeline.
"""

import torchaudio
from pyannote.audio import Pipeline


def _get_annotation(diarization):
    """Extract the Annotation object, handling both pyannote 3.x and 4.x output."""
    if hasattr(diarization, "speaker_diarization"):
        return diarization.speaker_diarization
    return diarization


def diarize(pipeline: Pipeline, audio_path: str, num_speakers: int = None):
    """
    Run speaker diarization on a single audio file.

    Args:
        pipeline: loaded pyannote Pipeline
        audio_path: path to audio file (WAV recommended)
        num_speakers: if known, fix the number of speakers

    Returns:
        (raw_output, annotation) -- raw pipeline output and the Annotation object
    """
    print(f"[INFO] Loading waveform: {audio_path}")
    waveform, sample_rate = torchaudio.load(audio_path)

    # pyannote expects mono
    if waveform.shape[0] > 1:
        waveform = waveform.mean(dim=0, keepdim=True)

    audio_input = {"waveform": waveform, "sample_rate": sample_rate}

    params = {}
    if num_speakers is not None:
        params["num_speakers"] = num_speakers

    print(f"[INFO] Running diarization...")
    result = pipeline(audio_input, **params)
    annotation = _get_annotation(result)

    return result, annotation


def parse_segments(annotation) -> list[dict]:
    """
    Convert an Annotation into a flat list of segment dicts.

    Returns:
        [{"speaker": "SPEAKER_00", "start": 0.0, "end": 2.5, "duration": 2.5}, ...]
    """
    segments = []
    for turn, _, speaker in annotation.itertracks(yield_label=True):
        segments.append({
            "speaker": speaker,
            "start": round(turn.start, 3),
            "end": round(turn.end, 3),
            "duration": round(turn.end - turn.start, 3),
        })
    return segments
