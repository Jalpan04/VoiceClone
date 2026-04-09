"""
Splitter -- cuts audio into per-speaker clips using pydub.
"""

import os
from collections import defaultdict
from pathlib import Path

from pydub import AudioSegment

from .utils import merge_segments

# Format hint map for pydub
FORMAT_MAP = {
    "mp3": "mp3", "wav": "wav", "flac": "flac",
    "ogg": "ogg", "m4a": "mp4", "aac": "aac",
    "wma": "asf", "opus": "ogg",
}


def load_audio(audio_path: str) -> AudioSegment:
    """Load an audio file into a pydub AudioSegment."""
    ext = Path(audio_path).suffix.lower().lstrip(".")
    fmt = FORMAT_MAP.get(ext, ext)
    audio = AudioSegment.from_file(audio_path, format=fmt)
    print(f"[INFO] Audio: {len(audio) / 1000:.2f}s | {audio.channels}ch | {audio.frame_rate}Hz")
    return audio


def split_audio(
    audio: AudioSegment,
    segments: list[dict],
    output_dir: str,
    source_name: str,
    min_duration: float = 0.5,
    output_format: str = "wav",
    merge_consecutive: bool = True,
    merge_gap_threshold: float = 0.3,
) -> dict[str, list[str]]:
    """
    Split audio into per-speaker clips.

    Args:
        audio: pydub AudioSegment
        segments: parsed diarization segments
        output_dir: root output dir for this file (e.g. output/my_interview)
        source_name: stem of the original file, used in clip filenames
        min_duration: skip clips shorter than this (seconds)
        output_format: wav / mp3 / flac
        merge_consecutive: merge same-speaker segments with small gaps
        merge_gap_threshold: max gap (seconds) to merge

    Returns:
        dict mapping speaker -> list of output file paths
    """
    os.makedirs(output_dir, exist_ok=True)

    if merge_consecutive:
        segments = merge_segments(segments, merge_gap_threshold)

    # Create speaker subdirs
    speakers = sorted(set(s["speaker"] for s in segments))
    for speaker in speakers:
        os.makedirs(os.path.join(output_dir, speaker), exist_ok=True)

    output_files: dict[str, list[str]] = defaultdict(list)
    speaker_counters: dict[str, int] = defaultdict(int)
    skipped = 0

    for seg in segments:
        if seg["duration"] < min_duration:
            skipped += 1
            continue

        speaker = seg["speaker"]
        start_ms = int(seg["start"] * 1000)
        end_ms = int(seg["end"] * 1000)
        clip = audio[start_ms:end_ms]

        speaker_counters[speaker] += 1
        clip_num = speaker_counters[speaker]

        filename = (
            f"{source_name}_{speaker}_clip{clip_num:04d}"
            f"_{seg['start']:.2f}s-{seg['end']:.2f}s.{output_format}"
        )
        filepath = os.path.join(output_dir, speaker, filename)

        export_params = {}
        if output_format == "mp3":
            export_params["bitrate"] = "192k"

        clip.export(filepath, format=output_format, **export_params)
        output_files[speaker].append(filepath)

    print(f"[INFO] Exported {sum(len(v) for v in output_files.values())} clips, skipped {skipped}")
    return dict(output_files)
