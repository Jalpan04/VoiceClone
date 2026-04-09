"""
Utilities -- segment merging, temp file helpers.
"""

import os
from pathlib import Path

from pydub import AudioSegment


def merge_segments(segments: list[dict], gap_threshold: float) -> list[dict]:
    """Merge consecutive segments from the same speaker when the gap is small."""
    if not segments:
        return segments

    merged = [segments[0].copy()]
    for seg in segments[1:]:
        last = merged[-1]
        gap = seg["start"] - last["end"]
        if seg["speaker"] == last["speaker"] and gap <= gap_threshold:
            last["end"] = seg["end"]
            last["duration"] = round(last["end"] - last["start"], 3)
        else:
            merged.append(seg.copy())

    print(f"[INFO] Merged {len(segments)} -> {len(merged)} segments (gap <= {gap_threshold}s)")
    return merged


def convert_to_wav(audio_path: str, output_dir: str) -> str:
    """
    Convert any audio file to a temporary WAV for reliable torchaudio loading.
    Returns the path to the WAV file.
    """
    source = Path(audio_path)
    wav_path = os.path.join(output_dir, f"{source.stem}_temp.wav")

    if source.suffix.lower() == ".wav":
        return str(source)  # already WAV, no conversion needed

    # Determine pydub format hint
    fmt_map = {
        ".mp3": "mp3", ".flac": "flac", ".ogg": "ogg",
        ".m4a": "mp4", ".aac": "aac", ".wma": "asf", ".opus": "ogg",
    }
    fmt = fmt_map.get(source.suffix.lower(), source.suffix.lower().lstrip("."))

    print(f"[INFO] Converting {source.name} -> WAV for processing...")
    audio = AudioSegment.from_file(str(source), format=fmt)
    audio.export(wav_path, format="wav")
    return wav_path


def cleanup_temp(path: str, original_path: str):
    """Remove a temp WAV file if it was created by convert_to_wav."""
    if path != original_path and os.path.exists(path):
        try:
            os.remove(path)
        except OSError:
            pass
