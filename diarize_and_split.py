"""
Speaker Diarization and Audio Splitter
======================================
Uses pyannote.audio for state-of-the-art speaker diarization and
pydub/ffmpeg for audio splitting.

Requirements:
    pip install pyannote.audio pydub tqdm python-dotenv

    You also need a Hugging Face token with access to:
      - pyannote/speaker-diarization-3.1
      - pyannote/segmentation-3.0

    Token is auto-loaded from .env file (HF_TOKEN=...) in the same directory.
    Get a free token at: https://hf.co/settings/tokens
    Then accept model terms at: https://hf.co/pyannote/speaker-diarization-3.1

Usage:
    python diarize_and_split.py --audio <path_to_audio>
    python diarize_and_split.py --audio interview.mp3 --min-duration 1.0
"""

import argparse
from pathlib import Path as _Path

# Auto-load .env from the script's own directory (keeps token out of the command line)
try:
    from dotenv import load_dotenv
    load_dotenv(_Path(__file__).parent / ".env")
except ImportError:
    pass  # python-dotenv not installed; fall back to env var or --token flag
import os
import sys
import json
from pathlib import Path
from collections import defaultdict

try:
    from pyannote.audio import Pipeline
    import torch
except ImportError:
    print("[ERROR] pyannote.audio is not installed.")
    print("Run: pip install pyannote.audio")
    sys.exit(1)

try:
    from pydub import AudioSegment
except ImportError:
    print("[ERROR] pydub is not installed.")
    print("Run: pip install pydub")
    sys.exit(1)

try:
    from tqdm import tqdm
except ImportError:
    # Fallback if tqdm not installed
    def tqdm(iterable, **kwargs):
        return iterable


def load_pipeline(hf_token: str) -> Pipeline:
    """Load the pyannote speaker diarization pipeline."""
    print("[INFO] Loading pyannote/speaker-diarization-3.1 pipeline...")
    print("       (First run downloads ~1GB of model weights, be patient)")

    pipeline = Pipeline.from_pretrained(
        "pyannote/speaker-diarization-3.1",
        token=hf_token,
    )

    # Use GPU if available for faster processing
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    pipeline.to(device)
    print(f"[INFO] Running on: {device}")

    return pipeline


def run_diarization(pipeline: Pipeline, audio_path: str, num_speakers: int = None) -> object:
    """Run speaker diarization on the audio file.
    
    Loads audio via torchaudio and passes a waveform dict to the pipeline,
    bypassing pyannote's built-in torchcodec reader which is broken on
    Windows with certain PyTorch+CUDA combinations.
    """
    import torchaudio

    print(f"[INFO] Diarizing: {audio_path}")
    print(f"[INFO] Loading waveform with torchaudio...")
    waveform, sample_rate = torchaudio.load(audio_path)

    # pyannote expects mono audio
    if waveform.shape[0] > 1:
        waveform = waveform.mean(dim=0, keepdim=True)

    audio_input = {"waveform": waveform, "sample_rate": sample_rate}

    params = {}
    if num_speakers is not None:
        params["num_speakers"] = num_speakers

    diarization = pipeline(audio_input, **params)
    return diarization


def _get_annotation(diarization):
    """Extract the Annotation object, handling both pyannote 3.x and 4.x output."""
    # pyannote 4.x returns a DiarizeOutput dataclass
    if hasattr(diarization, 'speaker_diarization'):
        return diarization.speaker_diarization
    # pyannote 3.x returns an Annotation directly
    return diarization


def parse_diarization(diarization) -> list[dict]:
    """
    Parse diarization output into a list of segment dicts.
    Returns: [{"speaker": "SPEAKER_00", "start": 0.0, "end": 2.5}, ...]
    """
    annotation = _get_annotation(diarization)
    segments = []
    for turn, _, speaker in annotation.itertracks(yield_label=True):
        segments.append({
            "speaker": speaker,
            "start": round(turn.start, 3),
            "end": round(turn.end, 3),
            "duration": round(turn.end - turn.start, 3),
        })
    return segments


def load_audio(audio_path: str) -> AudioSegment:
    """Load audio file using pydub (supports mp3, wav, flac, ogg, etc.)."""
    print(f"[INFO] Loading audio file: {audio_path}")
    ext = Path(audio_path).suffix.lower().lstrip(".")

    format_map = {
        "mp3": "mp3", "wav": "wav", "flac": "flac",
        "ogg": "ogg", "m4a": "mp4", "aac": "aac",
        "wma": "asf", "opus": "ogg",
    }
    fmt = format_map.get(ext, ext)
    audio = AudioSegment.from_file(audio_path, format=fmt)
    print(f"[INFO] Audio duration: {len(audio) / 1000:.2f}s | Channels: {audio.channels} | Sample rate: {audio.frame_rate}Hz")
    return audio


def split_audio_by_speaker(
    audio: AudioSegment,
    segments: list[dict],
    output_dir: str,
    min_duration: float = 0.5,
    output_format: str = "wav",
    merge_consecutive: bool = True,
    merge_gap_threshold: float = 0.3,
) -> dict:
    """
    Split audio into clips based on diarization segments.

    Args:
        audio: pydub AudioSegment object
        segments: list of diarization segment dicts
        output_dir: root output directory
        min_duration: skip clips shorter than this (seconds)
        output_format: output audio format (wav, mp3, flac)
        merge_consecutive: merge back-to-back segments from the same speaker
        merge_gap_threshold: max silence gap (seconds) to merge consecutive segments

    Returns:
        dict mapping speaker -> list of output file paths
    """
    os.makedirs(output_dir, exist_ok=True)

    # Optionally merge consecutive same-speaker segments with small gaps
    if merge_consecutive:
        segments = merge_segments(segments, merge_gap_threshold)

    # Group segments by speaker for organized output
    speaker_clips = defaultdict(list)
    speaker_counters = defaultdict(int)
    all_output_files = defaultdict(list)

    # Create per-speaker subdirectories
    speakers = sorted(set(s["speaker"] for s in segments))
    for speaker in speakers:
        speaker_dir = os.path.join(output_dir, speaker)
        os.makedirs(speaker_dir, exist_ok=True)

    print(f"\n[INFO] Splitting audio into {len(segments)} segments...")
    print(f"       Skipping segments shorter than {min_duration}s")
    skipped = 0

    for seg in tqdm(segments, desc="Exporting clips", unit="clip"):
        duration = seg["duration"]
        if duration < min_duration:
            skipped += 1
            continue

        speaker = seg["speaker"]
        start_ms = int(seg["start"] * 1000)
        end_ms = int(seg["end"] * 1000)

        clip = audio[start_ms:end_ms]

        speaker_counters[speaker] += 1
        clip_num = speaker_counters[speaker]

        filename = f"{speaker}_clip{clip_num:04d}_{seg['start']:.2f}s-{seg['end']:.2f}s.{output_format}"
        filepath = os.path.join(output_dir, speaker, filename)

        export_params = {}
        if output_format == "mp3":
            export_params["bitrate"] = "192k"

        clip.export(filepath, format=output_format, **export_params)
        all_output_files[speaker].append(filepath)

    if skipped:
        print(f"[INFO] Skipped {skipped} clips shorter than {min_duration}s")

    return dict(all_output_files)


def merge_segments(segments: list[dict], gap_threshold: float) -> list[dict]:
    """Merge consecutive segments from the same speaker with small gaps."""
    if not segments:
        return segments

    merged = [segments[0].copy()]
    for seg in segments[1:]:
        last = merged[-1]
        gap = seg["start"] - last["end"]
        if seg["speaker"] == last["speaker"] and gap <= gap_threshold:
            # Extend the last segment
            last["end"] = seg["end"]
            last["duration"] = round(last["end"] - last["start"], 3)
        else:
            merged.append(seg.copy())

    print(f"[INFO] Merged {len(segments)} segments -> {len(merged)} (gap threshold: {gap_threshold}s)")
    return merged


def print_summary(segments: list[dict], output_files: dict):
    """Print a summary of the diarization results."""
    print("\n" + "=" * 60)
    print("DIARIZATION SUMMARY")
    print("=" * 60)

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
        print(f"    Segments  : {stats['count']}")
        print(f"    Duration  : {stats['total_duration']:.2f}s ({pct:.1f}% of total)")
        print(f"    Clips saved: {clips_saved}")

    print(f"\n  Total speaking time: {total_duration:.2f}s")
    print(f"  Total clips exported: {sum(len(v) for v in output_files.values())}")
    print("=" * 60)


def save_diarization_json(segments: list[dict], output_dir: str, audio_name: str):
    """Save the raw diarization data as JSON for reference."""
    json_path = os.path.join(output_dir, f"{audio_name}_diarization.json")
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(segments, f, indent=2)
    print(f"\n[INFO] Diarization data saved to: {json_path}")


def save_diarization_rttm(diarization, output_dir: str, audio_name: str):
    """Save diarization in RTTM format (standard for speaker diarization evaluation)."""
    annotation = _get_annotation(diarization)
    rttm_path = os.path.join(output_dir, f"{audio_name}.rttm")
    with open(rttm_path, "w", encoding="utf-8") as f:
        annotation.write_rttm(f)
    print(f"[INFO] RTTM file saved to: {rttm_path}")


def main():
    parser = argparse.ArgumentParser(
        description="Diarize an audio file and split it by speaker",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument(
        "--audio", "-a",
        required=True,
        help="Path to the input audio file (mp3, wav, flac, ogg, m4a, etc.)",
    )
    parser.add_argument(
        "--token", "-t",
        default=os.environ.get("HF_TOKEN", ""),
        help="Hugging Face API token (or set HF_TOKEN env variable)",
    )
    parser.add_argument(
        "--output", "-o",
        default=None,
        help="Output directory (default: <audio_name>_diarized/)",
    )
    parser.add_argument(
        "--num-speakers", "-n",
        type=int,
        default=None,
        help="Known number of speakers (leave blank to auto-detect)",
    )
    parser.add_argument(
        "--min-duration",
        type=float,
        default=0.5,
        help="Minimum clip duration in seconds to export (default: 0.5)",
    )
    parser.add_argument(
        "--format", "-f",
        choices=["wav", "mp3", "flac"],
        default="wav",
        help="Output audio format (default: wav)",
    )
    parser.add_argument(
        "--no-merge",
        action="store_true",
        help="Disable merging of consecutive same-speaker segments",
    )
    parser.add_argument(
        "--merge-gap",
        type=float,
        default=0.3,
        help="Max gap (seconds) between same-speaker segments to merge (default: 0.3)",
    )
    args = parser.parse_args()

    # Validate inputs
    if not os.path.isfile(args.audio):
        print(f"[ERROR] Audio file not found: {args.audio}")
        sys.exit(1)

    if not args.token:
        print("[ERROR] Hugging Face token is required.")
        print("        Set --token <hf_token> or export HF_TOKEN=<token>")
        print("        Get your token at: https://hf.co/settings/tokens")
        print("        Accept model terms at: https://hf.co/pyannote/speaker-diarization-3.1")
        sys.exit(1)

    audio_path = Path(args.audio)
    audio_stem = audio_path.stem

    output_dir = args.output or str(audio_path.parent / f"{audio_stem}_diarized")
    os.makedirs(output_dir, exist_ok=True)

    print("\n" + "=" * 60)
    print("SPEAKER DIARIZATION + AUDIO SPLITTER")
    print("=" * 60)
    print(f"  Input   : {args.audio}")
    print(f"  Output  : {output_dir}")
    print(f"  Speakers: {'auto-detect' if args.num_speakers is None else args.num_speakers}")
    print(f"  Format  : {args.format}")
    print("=" * 60 + "\n")

    # Step 1: Load audio and convert to WAV if necessary (avoids pyannote io bugs)
    print(f"[INFO] Preparing audio: {args.audio}")
    audio = load_audio(args.audio)
    
    # Use a temporary wav file for diarization to bypass pyannote's torchcodec dependency issues
    temp_wav = Path(output_dir) / f"{audio_stem}_temp_processing.wav"
    if not temp_wav.exists():
        print(f"[INFO] Converting to temporary WAV for diarization...")
        audio.export(str(temp_wav), format="wav")
    
    # Step 2: Load diarization pipeline
    pipeline = load_pipeline(args.token)

    # Step 3: Run diarization on the WAV file
    diarization = run_diarization(pipeline, str(temp_wav), args.num_speakers)

    # Step 4: Parse results
    segments = parse_diarization(diarization)
    print(f"[INFO] Found {len(segments)} speech segments")

    # Step 5: Save RTTM and JSON
    save_diarization_rttm(diarization, output_dir, audio_stem)
    save_diarization_json(segments, output_dir, audio_stem)

    # Step 6: Split audio (we already have the 'audio' object loaded)
    output_files = split_audio_by_speaker(
        audio=audio,
        segments=segments,
        output_dir=output_dir,
        min_duration=args.min_duration,
        output_format=args.format,
        merge_consecutive=not args.no_merge,
        merge_gap_threshold=args.merge_gap,
    )

    # Clean up temp file
    if temp_wav.exists():
        try:
            os.remove(temp_wav)
        except:
            pass

    # Step 6: Print summary
    print_summary(segments, output_files)

    print(f"\n[DONE] All clips saved under: {output_dir}")
    print("       Each speaker has their own subfolder.")


if __name__ == "__main__":
    main()
