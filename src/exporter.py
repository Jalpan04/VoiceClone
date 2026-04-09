"""
Exporter -- saves diarization results to disk (JSON, RTTM) and prints summaries.
"""

import json
import os
from collections import defaultdict


def save_json(segments: list[dict], output_dir: str, source_name: str) -> str:
    """Save parsed segments as a JSON file. Returns the file path."""
    path = os.path.join(output_dir, f"{source_name}_diarization.json")
    with open(path, "w", encoding="utf-8") as f:
        json.dump(segments, f, indent=2)
    print(f"[INFO] JSON saved: {path}")
    return path


def save_rttm(annotation, output_dir: str, source_name: str) -> str:
    """Save the Annotation in RTTM format. Returns the file path."""
    path = os.path.join(output_dir, f"{source_name}.rttm")
    with open(path, "w", encoding="utf-8") as f:
        annotation.write_rttm(f)
    print(f"[INFO] RTTM saved: {path}")
    return path


def print_summary(segments: list[dict], output_files: dict[str, list[str]], source_name: str):
    """Print a human-readable summary of diarization results for one file."""
    print(f"\n{'=' * 60}")
    print(f"  RESULTS: {source_name}")
    print(f"{'=' * 60}")

    speaker_stats = defaultdict(lambda: {"count": 0, "total_duration": 0.0})
    for seg in segments:
        sp = seg["speaker"]
        speaker_stats[sp]["count"] += 1
        speaker_stats[sp]["total_duration"] += seg["duration"]

    total_duration = sum(s["total_duration"] for s in speaker_stats.values())

    for speaker in sorted(speaker_stats.keys()):
        stats = speaker_stats[speaker]
        clips_saved = len(output_files.get(speaker, []))
        pct = (stats["total_duration"] / total_duration * 100) if total_duration > 0 else 0
        print(f"  {speaker}:")
        print(f"    Segments   : {stats['count']}")
        print(f"    Duration   : {stats['total_duration']:.2f}s ({pct:.1f}%)")
        print(f"    Clips saved: {clips_saved}")

    print(f"\n  Total speaking time : {total_duration:.2f}s")
    print(f"  Total clips exported: {sum(len(v) for v in output_files.values())}")
    print(f"{'=' * 60}")
